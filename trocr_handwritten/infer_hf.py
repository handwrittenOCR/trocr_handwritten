import argparse
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from trocr_handwritten.llm.finetune.settings import LoraSettings
from trocr_handwritten.llm.finetune.train_lora import cap_visual_tokens
from trocr_handwritten.utils.logging_config import get_logger
from trocr_handwritten.utils.prompt import load_prompt

logger = get_logger(__name__)


def build_prompt_text(processor, prompt):
    """Render the chat template with one image placeholder and the OCR prompt."""
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }
    ]
    return processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


def transcribe_batch(
    model, processor, prompt_text, images, max_new_tokens, temperature
):
    """Generate transcriptions for a batch of PIL images (left-padded)."""
    inputs = processor(
        text=[prompt_text] * len(images),
        images=[[im] for im in images],
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    start = inputs["input_ids"].shape[1]
    return [
        processor.decode(out[i, start:], skip_special_tokens=True).strip()
        for i in range(len(images))
    ]


def main():
    """
    Transcribe images with a HuggingFace VLM (no vLLM), writing one .md per image
    mirroring the input subfolder structure. Use for models vLLM cannot serve
    (e.g. Gemma-4's heterogeneous per-layer config). Output matches `llm.ocr`, so
    `llm.metrics` scores it unchanged. Batched generation for speed.
    """
    ap = argparse.ArgumentParser(description="HF VLM OCR runner (transformers).")
    ap.add_argument("--model", required=True, help="HF repo id (base model)")
    ap.add_argument(
        "--adapter",
        default=None,
        help="Path to a trained LoRA adapter dir; applies it and matches its "
        "capped image resolution for train/test consistency.",
    )
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--pattern", default="*/*.jpg")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--prompt-path", default="config/ocr.prompt")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--device-map",
        default="auto",
        help="'auto' shards a big model across all visible GPUs; 'cuda:0' pins one.",
    )
    ap.add_argument("--output-ext", default=".md")
    s = LoraSettings()
    ap.add_argument(
        "--min-visual-tokens",
        type=int,
        default=s.image_min_pixels // 1024,
        help="Lower bound on visual tokens; must match the adapter's training run.",
    )
    ap.add_argument(
        "--max-visual-tokens",
        type=int,
        default=s.image_max_pixels // 1024,
        help="Upper bound on visual tokens; must match the adapter's training run.",
    )
    args = ap.parse_args()

    prompt = load_prompt(args.prompt_path)
    logger.info(f"Loading {args.model} (device_map={args.device_map})")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    if args.adapter:
        logger.info(f"Applying adapter {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    proc_src = args.adapter or args.model
    processor = AutoProcessor.from_pretrained(proc_src, trust_remote_code=True)
    cap_visual_tokens(
        processor, args.min_visual_tokens * 1024, args.max_visual_tokens * 1024
    )
    logger.info(
        f"Resolution: {args.min_visual_tokens}-{args.max_visual_tokens} visual tokens"
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"
    prompt_text = build_prompt_text(processor, prompt)

    input_dir = Path(args.input_dir)
    images = sorted(input_dir.glob(args.pattern))
    logger.info(f"Found {len(images)} images")
    base = Path(args.output_dir)

    for r in range(args.repeat):
        out_root = base / f"run{r + 1}" if args.repeat > 1 else base

        def dest_for(img_path):
            rel = img_path.relative_to(input_dir).with_suffix(args.output_ext)
            return out_root / rel

        pending = [p for p in images if not dest_for(p).exists()]
        failures = 0
        with tqdm(total=len(pending), desc=f"run{r + 1}/{args.repeat}") as bar:
            for i in range(0, len(pending), args.batch_size):
                chunk = pending[i : i + args.batch_size]
                imgs = [Image.open(p).convert("RGB") for p in chunk]
                try:
                    texts = transcribe_batch(
                        model,
                        processor,
                        prompt_text,
                        imgs,
                        args.max_new_tokens,
                        args.temperature,
                    )
                except Exception as exc:
                    logger.warning(
                        f"batch at {chunk[0]} failed ({exc}); retrying singly"
                    )
                    texts = []
                    for im in imgs:
                        try:
                            texts.append(
                                transcribe_batch(
                                    model,
                                    processor,
                                    prompt_text,
                                    [im],
                                    args.max_new_tokens,
                                    args.temperature,
                                )[0]
                            )
                        except Exception as single_exc:
                            failures += 1
                            logger.error(f"single image failed: {single_exc}")
                            texts.append(None)
                for p, text in zip(chunk, texts):
                    if text is None:
                        continue
                    dest = dest_for(p)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(text, encoding="utf-8")
                bar.update(len(chunk))
        logger.info(f"run{r + 1}: done ({failures} failures)")


if __name__ == "__main__":
    main()
