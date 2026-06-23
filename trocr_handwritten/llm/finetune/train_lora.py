import argparse
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
    Trainer,
    TrainingArguments,
)

from trocr_handwritten.llm.finetune.settings import LoraSettings
from trocr_handwritten.utils.prompt import load_prompt


def load_manifest(path, data_root):
    """Load a jsonl manifest and resolve absolute image paths."""
    recs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        r["image_path"] = str(Path(data_root) / r["image"])
        recs.append(r)
    return recs


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
        return enc


def main():
    s = LoraSettings()
    parser = argparse.ArgumentParser(description="LoRA fine-tune a VLM for OCR.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True, help="train manifest jsonl")
    parser.add_argument("--data-root", default=s.data_root)
    parser.add_argument("--prompt-path", default=s.prompt_path)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--init-adapter",
        default=None,
        help="Path to an existing LoRA adapter to continue training (stage-2).",
    )
    parser.add_argument("--epochs", type=float, default=s.epochs)
    parser.add_argument("--lr", type=float, default=s.lr)
    parser.add_argument("--lora-rank", type=int, default=s.lora_rank)
    parser.add_argument("--batch-size", type=int, default=s.batch_size)
    parser.add_argument("--grad-accum", type=int, default=s.grad_accum)
    parser.add_argument("--seed", type=int, default=s.seed)
    args = parser.parse_args()

    prompt = load_prompt(args.prompt_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    train_recs = load_manifest(args.train, args.data_root)

    model.config.use_cache = False
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        print(f"Continuing adapter {args.init_adapter}")
    else:
        lora = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * s.lora_alpha_ratio,
            lora_dropout=s.lora_dropout,
            bias="none",
            target_modules=s.target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    dataset = Dataset.from_list(train_recs)
    steps = (
        math.ceil(len(train_recs) / (args.batch_size * args.grad_accum)) * args.epochs
    )
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        data_seed=args.seed,
        bf16=True,
        logging_steps=s.logging_steps,
        save_strategy="no",
        warmup_ratio=s.warmup_ratio,
        lr_scheduler_type=s.lr_scheduler_type,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=dataset,
        data_collator=Collator(processor, prompt),
    )
    print(f"Training {args.model} on {len(train_recs)} samples (~{int(steps)} steps)")
    trainer.train()

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"SAVED_ADAPTER {args.output_dir}")


if __name__ == "__main__":
    main()
