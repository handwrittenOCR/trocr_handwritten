import argparse
import inspect
import json
import math
from pathlib import Path

import torch
from PIL import Image
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from trocr_handwritten.llm.finetune.settings import LoraSettings
from trocr_handwritten.utils.prompt import load_prompt


def load_manifest(path, data_root):
    """
    Load a jsonl manifest, resolve absolute image paths, and attach a `length`
    proxy (image pixel count via PIL header read) so group_by_length can batch
    similarly-sized images and cut padding waste.
    """
    recs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        r["image_path"] = str(Path(data_root) / r["image"])
        try:
            with Image.open(r["image_path"]) as im:
                w, h = im.size
            r["length"] = w * h
        except Exception:
            r["length"] = 0
        recs.append(r)
    return recs


def cap_visual_tokens(processor, min_pixels, max_pixels):
    """
    Bound Qwen3-VL/3.5 dynamic-resolution visual tokens to cap VRAM. Handles both
    processor conventions: flat min_pixels/max_pixels and a size dict with
    shortest_edge/longest_edge.
    """
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        return
    for attr, val in (("min_pixels", min_pixels), ("max_pixels", max_pixels)):
        if getattr(ip, attr, None) is not None:
            setattr(ip, attr, val)
    size = getattr(ip, "size", None)
    if size is None:
        return
    if hasattr(size, "longest_edge"):
        size.longest_edge = max_pixels
        size.shortest_edge = min_pixels
    elif isinstance(size, dict):
        size["longest_edge"] = max_pixels
        size["shortest_edge"] = min_pixels


def build_messages(prompt, answer=None):
    """Build a chat conversation with one image and the OCR prompt."""
    msgs = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }
    ]
    if answer is not None:
        msgs.append(
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        )
    return msgs


class Collator:
    """Collate VLM SFT batches, masking the prompt prefix and pad tokens."""

    def __init__(self, processor, prompt):
        self.processor = processor
        self.prompt = prompt
        self.tok = processor.tokenizer
        self._logged = False

    def __call__(self, batch):
        images = [Image.open(b["image_path"]).convert("RGB") for b in batch]
        full = [
            self.processor.apply_chat_template(
                build_messages(self.prompt, b["text"]), tokenize=False
            )
            for b in batch
        ]
        enc = self.processor(
            text=full,
            images=[[im] for im in images],
            return_tensors="pt",
            padding=True,
        )
        labels = enc["input_ids"].clone()
        labels[labels == self.tok.pad_token_id] = -100
        image_token_id = getattr(self.processor, "image_token_id", None)
        if image_token_id is not None:
            labels[enc["input_ids"] == image_token_id] = -100
        for i, b in enumerate(batch):
            prefix = self.processor.apply_chat_template(
                build_messages(self.prompt), tokenize=False, add_generation_prompt=True
            )
            plen = self.processor(
                text=[prefix], images=[[images[i]]], return_tensors="pt"
            )["input_ids"].shape[1]
            labels[i, :plen] = -100
        enc["labels"] = labels
        if not self._logged:
            self._logged = True
            print(
                f"[collator] input_ids shape={tuple(enc['input_ids'].shape)} "
                f"(seq_len is the visual-token cap witness)",
                flush=True,
            )
        return enc


def main():
    """
    LoRA fine-tune a VLM for OCR. Every hyper-parameter lives in `LoraSettings`;
    the CLI only carries per-run I/O (model, manifest, roots, output, stage-2 adapter).
    """
    s = LoraSettings()
    parser = argparse.ArgumentParser(description="LoRA fine-tune a VLM for OCR.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True, help="train manifest jsonl")
    parser.add_argument(
        "--eval",
        default=None,
        help="dev manifest jsonl; enables per-epoch eval + early stopping.",
    )
    parser.add_argument("--data-root", default=s.data_root)
    parser.add_argument("--prompt-path", default=s.prompt_path)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--init-adapter",
        default=None,
        help="Path to an existing LoRA adapter to continue training (stage-2).",
    )
    parser.add_argument(
        "--min-visual-tokens",
        type=int,
        default=s.image_min_pixels // 1024,
        help="Lower bound on visual tokens per image (64 = Qwen default).",
    )
    parser.add_argument(
        "--max-visual-tokens",
        type=int,
        default=s.image_max_pixels // 1024,
        help="Upper bound on visual tokens per image (16384 = Qwen default).",
    )
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=s.target_modules,
        help="LoRA target modules, or the single token 'all-linear' to also "
        "adapt the vision tower and the merger.",
    )
    parser.add_argument("--batch-size", type=int, default=s.batch_size)
    parser.add_argument("--grad-accum", type=int, default=s.grad_accum)
    parser.add_argument(
        "--gradient-checkpointing",
        type=int,
        default=int(s.gradient_checkpointing),
        help="1 recomputes activations in backward (saves VRAM, ~2x slower); "
        "0 trades VRAM for speed on models small enough to fit.",
    )
    args = parser.parse_args()

    min_px = args.min_visual_tokens * 1024
    max_px = args.max_visual_tokens * 1024
    targets = args.target_modules
    if isinstance(targets, list) and len(targets) == 1 and targets[0] == "all-linear":
        targets = "all-linear"

    prompt = load_prompt(args.prompt_path)
    single_gpu = s.device_map in (None, "", "cuda", "none", "single")
    load_kwargs = dict(
        dtype=getattr(torch, s.torch_dtype),
        attn_implementation=s.attn_implementation,
    )
    if not single_gpu:
        load_kwargs["device_map"] = s.device_map
    model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kwargs)
    if single_gpu:
        model = model.to("cuda")
    processor = AutoProcessor.from_pretrained(
        args.model,
        size={"shortest_edge": min_px, "longest_edge": max_px},
    )
    cap_visual_tokens(processor, min_px, max_px)
    print(
        f"[resolution] {args.min_visual_tokens}-{args.max_visual_tokens} visual "
        f"tokens ({min_px}-{max_px} px)",
        flush=True,
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    train_recs = load_manifest(args.train, args.data_root)

    model.config.use_cache = False
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        print(f"Continuing adapter {args.init_adapter}")
    else:
        lora = LoraConfig(
            r=s.lora_rank,
            lora_alpha=s.lora_rank * s.lora_alpha_ratio,
            lora_dropout=s.lora_dropout,
            bias=s.lora_bias,
            target_modules=targets,
            task_type=s.task_type,
        )
        model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    dataset = Dataset.from_list(train_recs)
    eval_ds = None
    if args.eval:
        eval_ds = Dataset.from_list(load_manifest(args.eval, args.data_root))
    steps = math.ceil(len(train_recs) / (args.batch_size * args.grad_accum)) * s.epochs
    cand = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=s.epochs,
        learning_rate=s.lr,
        seed=s.seed,
        data_seed=s.seed,
        bf16=s.bf16,
        logging_steps=s.logging_steps,
        save_strategy=s.save_strategy,
        warmup_ratio=s.warmup_ratio,
        lr_scheduler_type=s.lr_scheduler_type,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        remove_unused_columns=s.remove_unused_columns,
        report_to=s.report_to,
        dataloader_num_workers=s.dataloader_num_workers,
        dataloader_pin_memory=True,
        use_liger_kernel=s.use_liger_kernel,
        group_by_length=s.group_by_length,
        length_column_name=s.length_column_name,
    )
    if eval_ds is not None:
        cand.update(
            eval_strategy="epoch",
            save_strategy="epoch",
            per_device_eval_batch_size=args.batch_size,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=1,
        )
    allowed = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "warmup_ratio" not in allowed and "warmup_steps" in allowed:
        cand.pop("warmup_ratio", None)
        cand["warmup_steps"] = int(s.warmup_ratio * steps)
    dropped = sorted(k for k in cand if k not in allowed)
    if dropped:
        import transformers

        print(
            f"transformers {transformers.__version__}: dropping unsupported "
            f"TrainingArguments kwargs {dropped}"
        )
    targs = TrainingArguments(**{k: v for k, v in cand.items() if k in allowed})
    callbacks = None
    if eval_ds is not None:
        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=s.early_stopping_patience,
                early_stopping_threshold=s.early_stopping_threshold,
            )
        ]
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=dataset,
        eval_dataset=eval_ds,
        data_collator=Collator(processor, prompt),
        callbacks=callbacks,
    )
    print(f"Training {args.model} on {len(train_recs)} samples (~{int(steps)} steps)")
    trainer.train()

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"SAVED_ADAPTER {args.output_dir}")


if __name__ == "__main__":
    main()
