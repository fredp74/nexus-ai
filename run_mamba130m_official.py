"""CPU generation with the official Mamba-130M."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, MambaForCausalLM, TextStreamer

MODEL = "state-spaces/mamba-130m-hf"
REVISION = "1e76775f628fbf1350fbe4dbb3d971ba64af25a1"
WEIGHTS_SHA256 = (
    "1a5ed29c492ef4d485df3b7c2c8109771696589855b2162ad1ba618b6067cbea"
)
FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
]
PROMPTS = [
    "Once upon a time,",
    "The history of science",
    "The unemployment rate",
]


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(8 * 1024 * 1024), b""
        ):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path,
        default=Path("runs/_models/mamba130m_1e76775f")
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/mamba130m_official/generations.json")
    )
    parser.add_argument("--tokens", type=int, default=160)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--prompt", action="append")
    args = parser.parse_args()
    if args.tokens < 1 or args.threads < 1:
        parser.error("Positive tokens/threads required")

    torch.set_num_threads(args.threads)
    print(
        "Official model:", MODEL,
        "| revision:", REVISION, flush=True
    )
    if not all((args.cache/name).exists() for name in FILES):
        print(
            "Downloading model and tokenizer (about 520 MB)...",
            flush=True
        )
        snapshot_download(
            MODEL,
            revision=REVISION,
            local_dir=str(args.cache),
            allow_patterns=FILES
        )

    print("Verifying weights...", flush=True)
    if (
        digest(args.cache/"model.safetensors")
        != WEIGHTS_SHA256
    ):
        raise ValueError("Official model checksum mismatch")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.cache),
        local_files_only=True,
        trust_remote_code=False
    )
    print("Loading float32 CPU model...", flush=True)
    model = MambaForCausalLM.from_pretrained(
        str(args.cache),
        local_files_only=True,
        use_safetensors=True,
        torch_dtype=torch.float32
    ).to("cpu").eval().requires_grad_(False)

    result = dict(
        model=MODEL,
        revision=REVISION,
        weights_sha256=WEIGHTS_SHA256,
        parameters=sum(p.numel() for p in model.parameters()),
        device="cpu",
        dtype="float32",
        max_new_tokens=args.tokens,
        temperature=.7,
        top_k=40,
        seed=42,
        generations=[]
    )
    print("Parameters:", result["parameters"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for prompt in args.prompt or PROMPTS:
        print("\n--- " + prompt + " ---", flush=True)
        torch.manual_seed(42)
        inputs = tokenizer(prompt, return_tensors="pt")
        start_length = inputs["input_ids"].shape[1]
        started = time.monotonic()

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.tokens,
                do_sample=True,
                temperature=.7,
                top_k=40,
                top_p=1.,
                repetition_penalty=1.,
                num_beams=1,
                use_cache=True,
                pad_token_id=model.config.eos_token_id,
                eos_token_id=model.config.eos_token_id,
                streamer=TextStreamer(
                    tokenizer,
                    skip_prompt=False,
                    skip_special_tokens=True
                )
            )

        elapsed = time.monotonic()-started
        result["generations"].append(dict(
            prompt=prompt,
            text=tokenizer.decode(
                output[0], skip_special_tokens=True
            ),
            continuation=tokenizer.decode(
                output[0, start_length:],
                skip_special_tokens=True
            ),
            generated_tokens=int(
                output.shape[1]-start_length
            ),
            seconds=elapsed
        ))
        pending = args.output.with_suffix(".tmp")
        pending.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        pending.replace(args.output)

    print("\nSaved:", args.output, flush=True)


if __name__ == "__main__":
    main()
