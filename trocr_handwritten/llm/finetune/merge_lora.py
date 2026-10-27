import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from trocr_handwritten.utils.logging_config import get_logger

logger = get_logger(__name__)


def visual_target_name(adapter_dir):
    """
    Return the base-model parameter name of one adapted vision-tower linear.

    Read from the adapter's own keys rather than guessed from the model: only
    modules the adapter actually wraps change during the merge, so anything else
    (pos_embed, patch_embed conv) would report a zero delta and mean nothing.
    """
    from safetensors import safe_open

    path = Path(adapter_dir) / "adapter_model.safetensors"
    with safe_open(path, "pt") as f:
        for key in f.keys():
            if "visual" in key and ".lora_A." in key:
                base = key.split("base_model.model.")[-1].split(".lora_A")[0]
                return f"{base}.weight"
    return None


def main():
    """
    Fold a LoRA adapter into its base weights and write a standalone model.

    vLLM applies LoRA to the language model only for architectures whose vision
    tower has no registered PunicaWrapper (Qwen3.5 among them), so an adapter
    trained with `all-linear` loses its visual weights at serving time. A merged
    checkpoint is an ordinary model: every weight is in place and no LoRA flag is
    needed. The merge is exact for standard bf16 LoRA -- W <- W + BA * alpha/r.

    The processor is copied from the adapter directory so the merged model keeps
    the resolution bounds it was trained with.
    """
    ap = argparse.ArgumentParser(description="Merge a LoRA adapter into its base.")
    ap.add_argument("--base", required=True, help="HF repo id of the base model")
    ap.add_argument("--adapter", required=True, help="Trained LoRA adapter dir")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--device",
        default="cpu",
        help="'cpu' keeps the GPU free; a 9B merge needs ~20GB of RAM either way.",
    )
    args = ap.parse_args()

    logger.info(f"Loading {args.base} on {args.device}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True
    )
    name = visual_target_name(args.adapter)
    before = None
    if name is not None:
        params = dict(model.named_parameters())
        if name in params:
            before = params[name].detach().clone().float()
        else:
            logger.warning(f"Adapter targets {name}, absent from the base model")

    logger.info(f"Applying adapter {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()

    if before is not None:
        after = dict(model.named_parameters())[name].detach().float()
        delta = (after - before).abs().max().item()
        logger.info(f"Visual weight {name}: max |delta| = {delta:.3e}")
        if delta == 0.0:
            logger.warning(
                "Vision tower unchanged by the merge -- the adapted weight did "
                "not move, so the merge did not take."
            )
    elif name is None:
        logger.warning("Adapter carries no visual LoRA weights")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    try:
        processor = AutoProcessor.from_pretrained(args.adapter, trust_remote_code=True)
    except Exception:
        logger.warning(
            f"No processor in {args.adapter} (mid-training checkpoint); "
            f"falling back to {args.base}"
        )
        processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    processor.save_pretrained(out)
    logger.info(f"SAVED_MERGED {out}")


if __name__ == "__main__":
    main()
