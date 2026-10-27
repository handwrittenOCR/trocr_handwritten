import argparse
import os
import time

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from trocr_handwritten.llm.finetune.settings import LoraSettings
from trocr_handwritten.llm.finetune.train_lora import (
    Collator,
    cap_visual_tokens,
    load_manifest,
)
from trocr_handwritten.utils.prompt import load_prompt


def build_model(model_id, s, attn, gc):
    """Load the base VLM, wrap it with LoRA, and set the train-time knobs."""
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=getattr(torch, s.torch_dtype),
        attn_implementation=attn,
    ).to("cuda")
    model.config.use_cache = False
    lora = LoraConfig(
        r=s.lora_rank,
        lora_alpha=s.lora_rank * s.lora_alpha_ratio,
        lora_dropout=s.lora_dropout,
        bias=s.lora_bias,
        target_modules=s.target_modules,
        task_type=s.task_type,
    )
    model = get_peft_model(model, lora)
    if gc:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    return model


def apply_liger(model_type):
    """Monkey-patch the model class with Liger fused kernels (incl. FLCE)."""
    from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN

    fn = MODEL_TYPE_TO_APPLY_LIGER_FN.get(model_type)
    if fn is None:
        print(f"no liger patcher for model_type={model_type}")
        return
    fn()


def bench_model(model, batch, n_params, steps=20):
    """Repeat fwd+bwd on a fixed real batch; return per-step time, tokens/s, MFU%."""
    batch = {k: v.to("cuda") for k, v in batch.items()}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    b, seq = batch["input_ids"].shape
    tokens = b * seq
    for _ in range(3):
        opt.zero_grad()
        model(**batch).loss.backward()
        opt.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    for _ in range(steps):
        opt.zero_grad()
        model(**batch).loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = (time.time() - t) / steps
    tok_s = tokens / dt
    tflops = 6 * n_params * tokens / dt / 1e12
    peak = torch.cuda.max_memory_allocated() / 1e9
    return dt, tok_s, tflops, peak, (b, seq)


def bench_loader(dataset, collator, nw, batches=40, bs=4):
    """Iterate the DataLoader without a model; return images/second."""
    dl = DataLoader(
        dataset,
        batch_size=bs,
        collate_fn=collator,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
    )
    it = iter(dl)
    n = next(it)["input_ids"].shape[0]
    t = time.time()
    for _ in range(batches - 1):
        n += next(it)["input_ids"].shape[0]
    return n / (time.time() - t)


def main():
    """
    Isolate the two halves of training speed for a VLM LoRA run:
      * dataloader-only images/s (is the CPU pipeline the bottleneck?)
      * model-only tokens/s + backbone MFU across (attn, gradient_checkpointing,
        batch_size) so we can pick the fastest protocol-neutral config.
    """
    s = LoraSettings()
    ap = argparse.ArgumentParser(description="Speed micro-bench for VLM LoRA training.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--data-root", default=s.data_root)
    ap.add_argument("--prompt-path", default=s.prompt_path)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument(
        "--min-visual-tokens",
        type=int,
        default=s.image_min_pixels // 1024,
        help="Lower bound on visual tokens per image; mirrors train_lora so the "
        "bench measures the same regime as the real run.",
    )
    ap.add_argument(
        "--max-visual-tokens",
        type=int,
        default=s.image_max_pixels // 1024,
        help="Upper bound on visual tokens per image.",
    )
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 16, 24])
    ap.add_argument("--gc", nargs="+", type=int, default=[1, 0])
    ap.add_argument("--liger", nargs="+", type=int, default=[0, 1])
    args = ap.parse_args()

    prompt = load_prompt(args.prompt_path)
    processor = AutoProcessor.from_pretrained(
        args.model,
        size={"shortest_edge": s.image_min_pixels, "longest_edge": s.image_max_pixels},
    )
    cap_visual_tokens(processor, s.image_min_pixels, s.image_max_pixels)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    collator = Collator(processor, prompt)

    recs = load_manifest(args.train, args.data_root)
    dataset = Dataset.from_list(recs)

    print("=== dataloader-only (images/s) ===")
    for nw in (8, 16):
        try:
            ips = bench_loader(dataset, collator, nw)
            print(f"num_workers={nw:2d}: {ips:6.1f} img/s")
        except Exception as exc:
            print(f"num_workers={nw:2d}: FAILED ({exc})")

    model_type = AutoConfig.from_pretrained(args.model).model_type
    recs_ws = sorted(recs, key=lambda r: os.path.getsize(r["image_path"]), reverse=True)
    print(f"\n=== model-only, worst-case seq (model_type={model_type}) ===")
    print(
        f"{'liger':6s} {'gc':3s} {'bs':4s} {'s/step':7s} {'tok/s':8s} "
        f"{'peakGB':7s} {'shape'}"
    )
    for lg in args.liger:
        if lg:
            apply_liger(model_type)
        for gc in args.gc:
            try:
                model = build_model(args.model, s, args.attn, bool(gc))
            except Exception as exc:
                print(f"liger={lg} skip (build failed: {str(exc)[:50]})")
                break
            for bs in args.batch_sizes:
                try:
                    batch = collator(recs_ws[:bs])
                    dt, tok_s, _, peak, shape = bench_model(model, batch, 1)
                    print(
                        f"{lg:<6d} {gc:<3d} {bs:<4d} {dt:7.3f} {tok_s:8.0f} "
                        f"{peak:7.1f} {shape}"
                    )
                except torch.OutOfMemoryError:
                    print(f"{lg:<6d} {gc:<3d} {bs:<4d} OOM")
                    torch.cuda.empty_cache()
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
