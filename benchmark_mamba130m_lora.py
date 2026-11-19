"""Bounded CPU LoRA benchmark on genuine DART training examples."""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
VENDOR = (
    Path(__file__).resolve().parent
    / "runs/_deps/mambapy_reference"
)
sys.path.insert(0, str(VENDOR))

import torch
import transformers
import peft
from transformers import AutoTokenizer, MambaConfig, MambaForCausalLM
from transformers.utils.import_utils import is_mambapy_available
from peft import LoraConfig, get_peft_model

REVISION = "1e76775f628fbf1350fbe4dbb3d971ba64af25a1"
MODEL_HASH = (
    "1a5ed29c492ef4d485df3b7c2c8109771696589855b2162ad1ba618b6067cbea"
)
DART_COMMIT = "bb644b12ec1d0264f96a31145534b9a94f6ff5f5"
DART_HASH = (
    "92c8594979c05f508f5739047079ec2ffe5a244e58bfa2b50a9cb8b9c65f5a2b"
)
DART_URL = (
    "https://raw.githubusercontent.com/Yale-LILY/dart/"
    + DART_COMMIT
    + "/data/v1.1.1/dart-v1.1.1-full-train.json"
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8*1024*1024), b""
        ):
            h.update(chunk)
    return h.hexdigest()


def batch_examples(path, tokenizer):
    selected = []
    data = json.loads(path.read_text(encoding="utf-8"))
    for index, row in enumerate(data):
        facts = "\n".join(
            " | ".join(triple)
            for triple in row["tripleset"]
        )
        prompt = "Facts:\n" + facts + "\nText:"
        for annotation in row["annotations"]:
            text = " ".join(annotation["text"].split())
            if (
                not 8 <= len(text.split()) <= 40
                or not text.endswith((".", "!", "?"))
            ):
                continue
            prefix = tokenizer.encode(prompt)
            answer = tokenizer.encode(" "+text)
            if (
                tokenizer.encode(prompt+" "+text)
                != prefix+answer
            ):
                continue
            ids = prefix+answer+[tokenizer.eos_token_id]
            if not 16 <= len(ids) <= 96:
                continue
            labels = (
                [-100]*len(prefix)
                + answer
                + [tokenizer.eos_token_id]
            )
            selected.append(dict(
                index=index,
                prompt=prompt,
                reference=text,
                input_ids=ids,
                labels=labels
            ))
            break
        if len(selected) == 4:
            return selected
    raise ValueError("Not enough short DART examples")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path,
        default=Path("runs/_models/mamba130m_1e76775f")
    )
    parser.add_argument(
        "--run", type=Path,
        default=Path("runs/mamba130m_lora_benchmark")
    )
    parser.add_argument(
        "--dart", type=Path,
        default=Path("data/dart_cache/train.json")
    )
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1 or args.run.exists():
        parser.error(
            "Positive threads and a new run directory required"
        )
    torch.set_num_threads(args.threads)

    if not is_mambapy_available():
        raise ValueError(
            "Pinned mambapy dependency not available"
        )
    if digest(args.cache/"model.safetensors") != MODEL_HASH:
        raise ValueError(
            "Official weights checksum mismatch"
        )
    if not args.dart.exists():
        print(
            "Downloading DART training data (23 MB)...",
            flush=True
        )
        args.dart.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(
            DART_URL, timeout=90
        ) as response:
            raw = response.read()
        if hashlib.sha256(raw).hexdigest() != DART_HASH:
            raise ValueError(
                "DART download checksum mismatch"
            )
        args.dart.write_bytes(raw)
    if digest(args.dart) != DART_HASH:
        raise ValueError("DART cache checksum mismatch")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.cache), local_files_only=True
    )
    rows = batch_examples(args.dart, tokenizer)
    config = MambaConfig.from_pretrained(
        str(args.cache), local_files_only=True
    )
    config.use_mambapy = True
    config.use_cache = False
    print(
        "Loading official CPU model and rank-8 out_proj LoRA...",
        flush=True
    )
    model = MambaForCausalLM.from_pretrained(
        str(args.cache),
        config=config,
        local_files_only=True,
        torch_dtype=torch.float32,
        use_safetensors=True
    )
    torch.manual_seed(42)
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=.1,
            target_modules=["out_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )
    )
    parameters = [
        p for p in model.parameters()
        if p.requires_grad
    ]
    count = sum(p.numel() for p in parameters)
    if (
        count != 442368
        or any(
            "lora_" not in name
            for name, p in model.named_parameters()
            if p.requires_grad
        )
    ):
        raise ValueError(
            "Unexpected trainable parameter selection"
        )

    args.run.mkdir(parents=True)
    model.save_pretrained(
        args.run/"initial_adapter",
        safe_serialization=True
    )
    optimizer = torch.optim.AdamW(
        parameters, lr=.0001, weight_decay=0.
    )
    model.train()
    print(
        "Trainable parameters:", count,
        "| lengths:",
        [len(r["input_ids"]) for r in rows],
        flush=True
    )
    timings, losses = [], []

    for update in range(5):
        row = rows[update % len(rows)]
        ids = torch.tensor([row["input_ids"]])
        labels = torch.tensor([row["labels"]])
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = model(
            input_ids=ids,
            labels=labels,
            use_cache=False
        ).loss
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        if not any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in parameters
        ):
            raise ValueError("No adapter gradient")
        if any(
            p.grad is not None
            for p in model.parameters()
            if not p.requires_grad
        ):
            raise ValueError(
                "Frozen base received gradients"
            )
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, 1.
        )
        if not torch.isfinite(norm):
            raise ValueError(
                "Nonfinite adapter gradient"
            )
        optimizer.step()
        seconds = time.perf_counter()-started
        timings.append(seconds)
        losses.append(loss.item())
        print(
            f"Update {update+1}/5 "
            f"| loss={loss.item():.4f} "
            f"| {seconds:.2f} s",
            flush=True
        )

    model.save_pretrained(
        args.run/"smoke_adapter",
        safe_serialization=True
    )
    mean = statistics.mean(timings[2:])
    result = dict(
        base_revision=REVISION,
        base_weights_sha256=MODEL_HASH,
        dart_url=DART_URL,
        dart_sha256=DART_HASH,
        examples=rows,
        transformers=transformers.__version__,
        peft=peft.__version__,
        trainable_parameters=count,
        rank=8,
        alpha=8,
        dropout=.1,
        target_modules=["out_proj"],
        dtype="float32",
        batch=1,
        use_mambapy=True,
        threads=args.threads,
        update_seconds=timings,
        losses=losses,
        mean_seconds=mean,
        estimated_100_updates_minutes=mean*100/60,
        official_weights_unchanged=(
            digest(args.cache/"model.safetensors")
            == MODEL_HASH
        ),
        note=(
            "Five updates on four training examples measure "
            "feasibility, not quality. Initial and smoke "
            "adapters are separate."
        )
    )
    (args.run/"result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8"
    )
    print(
        "Mean seconds (last 3):",
        round(mean, 2), flush=True
    )
    print(
        "100 updates, training-only minutes:",
        round(mean*100/60, 1), flush=True
    )
    print(
        "Official weights unchanged:",
        result["official_weights_unchanged"],
        flush=True
    )
    print(
        "Result:", args.run/"result.json",
        flush=True
    )


if __name__ == "__main__":
    main()
