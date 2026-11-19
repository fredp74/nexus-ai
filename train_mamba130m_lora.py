"""Resumable short-example DART pilot on frozen Mamba-130M."""
import argparse
import json
import math
import random
import signal
import time
import urllib.request
from pathlib import Path

import benchmark_mamba130m_lora as bench
import torch
from peft import (
    PeftModel,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, MambaConfig, MambaForCausalLM

HASHES = dict(
    train=bench.DART_HASH,
    dev="56606eac12baa7f0ddb81c61890f9f1a95bace4df8f8989852786358fe5d2b88",
    test="984be50fa46d0dbfce1ecfdad4a5c5a5cf82f1be0b124fe94f9f9b175d2a5045",
)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    temporary.replace(path)


def save(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def key(row):
    triples = sorted(
        tuple(
            " ".join(str(v).split()).casefold()
            for v in triple
        )
        for triple in row["tripleset"]
    )
    return json.dumps(triples, ensure_ascii=True)


def prepare(args, tokenizer):
    raw = {}
    for split, expected in HASHES.items():
        path = args.dart/(split+".json")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            url = bench.DART_URL.replace(
                "-train.json", "-"+split+".json"
            )
            print("Downloading DART", split, flush=True)
            with urllib.request.urlopen(
                url, timeout=90
            ) as response:
                payload = response.read()
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(payload)
            if bench.digest(temporary) != expected:
                raise ValueError(
                    "DART download checksum: "+split
                )
            temporary.replace(path)
        if bench.digest(path) != expected:
            raise ValueError("DART checksum: "+split)
        raw[split] = json.loads(
            path.read_text(encoding="utf-8")
        )

    reserved = {key(r) for r in raw["test"]}
    blocked = dict(
        test=set(),
        dev=reserved,
        train=reserved | {key(r) for r in raw["dev"]}
    )
    result = {}
    for split in ("train", "dev", "test"):
        indices = list(range(len(raw[split])))
        random.Random(42).shuffle(indices)
        selected, seen = [], set()
        count = (
            args.train_examples
            if split == "train"
            else args.eval_examples
        )
        for index in indices:
            row = raw[split][index]
            record_key = key(row)
            if (
                record_key in blocked[split]
                or record_key in seen
            ):
                continue
            prompt = (
                "Facts:\n"
                + "\n".join(
                    " | ".join(t) for t in row["tripleset"]
                )
                + "\nText:"
            )
            prefix = tokenizer.encode(prompt)
            for annotation in row["annotations"]:
                text = " ".join(annotation["text"].split())
                if (
                    not 8 <= len(text.split()) <= 40
                    or not text.endswith((".", "!", "?"))
                ):
                    continue
                answer = tokenizer.encode(" "+text)
                if (
                    tokenizer.encode(prompt+" "+text)
                    != prefix+answer
                ):
                    continue
                ids = prefix+answer+[tokenizer.eos_token_id]
                if not 16 <= len(ids) <= 96:
                    continue
                selected.append(dict(
                    index=index,
                    prompt=prompt,
                    reference=text,
                    key=record_key,
                    ids=ids,
                    labels=(
                        [-100]*len(prefix)
                        + answer
                        + [tokenizer.eos_token_id]
                    )
                ))
                seen.add(record_key)
                break
            if len(selected) == count:
                break
        if len(selected) < count:
            raise ValueError(
                "Not enough short examples: "+split
            )
        result[split] = selected
    return result


def loss_for(model, row):
    ids = torch.tensor([row["ids"]])
    labels = torch.tensor([row["labels"]])
    return model(
        input_ids=ids, labels=labels, use_cache=False
    ).loss


@torch.no_grad()
def evaluate(model, rows):
    model.eval()
    total, tokens = 0., 0
    for row in rows:
        n = sum(t != -100 for t in row["labels"][1:])
        total += loss_for(model, row).item()*n
        tokens += n
    model.train()
    return total/tokens


@torch.no_grad()
def generate(model, tokenizer, rows, limit):
    model.eval()
    result = []
    for row in rows:
        ids = tokenizer(
            row["prompt"], return_tensors="pt"
        )["input_ids"]
        output = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=limit,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
        result.append(dict(
            index=row["index"],
            facts=row["prompt"],
            reference=row["reference"],
            generated=tokenizer.decode(
                output[0, ids.shape[1]:],
                skip_special_tokens=True
            )
        ))
    model.train()
    return result


def train(args):
    if (
        bench.digest(args.cache/"model.safetensors")
        != bench.MODEL_HASH
    ):
        raise ValueError("Official weights changed")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.cache), local_files_only=True
    )
    data = prepare(args, tokenizer)
    identity = dict(
        format="mamba130m-dart-lora-v1",
        model_revision=bench.REVISION,
        model_hash=bench.MODEL_HASH,
        dart_hashes=HASHES,
        initial_adapter_hash=bench.digest(
            args.initial/"adapter_model.safetensors"
        ),
        initial_config_hash=bench.digest(
            args.initial/"adapter_config.json"
        ),
        script_hash=bench.digest(Path(__file__)),
        steps=args.steps,
        accumulation=args.accumulation,
        train_examples=args.train_examples,
        eval_examples=args.eval_examples,
        generation_examples=args.generation_examples,
        max_new_tokens=args.max_new_tokens,
        lr=.0001,
        seed=42,
        max_length=96
    )
    config = MambaConfig.from_pretrained(
        str(args.cache), local_files_only=True
    )
    config.use_mambapy = True
    config.use_cache = False
    backbone = MambaForCausalLM.from_pretrained(
        str(args.cache),
        config=config,
        local_files_only=True,
        use_safetensors=True,
        torch_dtype=torch.float32
    )
    model = PeftModel.from_pretrained(
        backbone, str(args.initial), is_trainable=True
    )
    parameters = [
        p for p in model.parameters()
        if p.requires_grad
    ]
    if sum(p.numel() for p in parameters) != 442368:
        raise ValueError(
            "Unexpected adapter parameter count"
        )
    optimizer = torch.optim.AdamW(
        parameters, lr=.0001, weight_decay=0.
    )
    rng = random.Random(42)
    torch.manual_seed(42)
    checkpoint = args.run/"last.pt"
    step, best, history = 0, float("inf"), []

    if checkpoint.exists():
        state = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True
        )
        if state["identity"] != identity:
            raise ValueError(
                "Resume data, settings or script changed"
            )
        set_peft_model_state_dict(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["random_state"])
        torch.set_rng_state(state["torch_rng"])
        step = state["step"]
        best = state["best"]
        history = state["history"]
    elif args.run.exists() and any(args.run.iterdir()):
        raise FileExistsError("Use a new run directory")
    args.run.mkdir(parents=True, exist_ok=True)

    def snapshot():
        return dict(
            identity=identity,
            step=step,
            best=best,
            history=history,
            adapter=get_peft_model_state_dict(model),
            optimizer=optimizer.state_dict(),
            random_state=rng.getstate(),
            torch_rng=torch.get_rng_state()
        )

    stopped = [False]
    previous = signal.signal(
        signal.SIGINT,
        lambda *_: stopped.__setitem__(0, True)
    )
    try:
        if not checkpoint.exists():
            best = evaluate(model, data["dev"])
            history.append(dict(
                step=0, validation_nll=best
            ))
            write_json(
                args.run/"task.json",
                dict(
                    identity=identity,
                    rows={
                        split: [
                            dict(index=r["index"], key=r["key"])
                            for r in rows
                        ]
                        for split, rows in data.items()
                    }
                )
            )
            model.save_pretrained(
                args.run/"best_adapter",
                safe_serialization=True
            )
            save(checkpoint, snapshot())
            print(
                "Initial validation NLL:", best,
                flush=True
            )

        started, start_step = time.monotonic(), step
        print(
            "LoRA pilot:", step, "/", args.steps,
            "| effective batch:", args.accumulation,
            flush=True
        )
        while step < args.until and not stopped[0]:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logged = 0.
            for row in rng.choices(
                data["train"], k=args.accumulation
            ):
                loss = (
                    loss_for(model, row)/args.accumulation
                )
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite loss")
                loss.backward()
                logged += loss.item()

            norm = torch.nn.utils.clip_grad_norm_(
                parameters, 1.
            )
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite gradient")
            warmup = min(
                50, max(1, args.steps//10)
            )
            progress = max(
                0.,
                (step+1-warmup)/max(1, args.steps-warmup)
            )
            lr = (
                .0001
                * min(1., (step+1)/warmup)
                * (.1+.45*(1+math.cos(math.pi*progress)))
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            step += 1

            if step % 10 == 0:
                seconds = (
                    (time.monotonic()-started)/(step-start_step)
                )
                print(
                    f"Step {step}/{args.steps} "
                    f"| loss={logged:.4f} "
                    f"| ETA={(args.until-step)*seconds/60:.1f} min",
                    flush=True
                )
            if step % 50 == 0 or step == args.until:
                value = evaluate(model, data["dev"])
                history.append(dict(
                    step=step, validation_nll=value
                ))
                if value < best:
                    best = value
                    model.save_pretrained(
                        args.run/"best_adapter",
                        safe_serialization=True
                    )
                print(
                    "Validation NLL:", value,
                    "| best:", best, flush=True
                )
            if (
                step % 25 == 0
                or step == args.until
                or stopped[0]
            ):
                save(checkpoint, snapshot())

        save(checkpoint, snapshot())
        if step == args.steps and not stopped[0]:
            model.load_adapter(
                str(args.run/"best_adapter"),
                adapter_name="selected",
                is_trainable=False
            )
            model.set_adapter("selected")
            with model.disable_adapter():
                before = evaluate(model, data["test"])
                texts_before = generate(
                    model, tokenizer,
                    data["test"][:args.generation_examples],
                    args.max_new_tokens
                )
            after = evaluate(model, data["test"])
            texts_after = generate(
                model, tokenizer,
                data["test"][:args.generation_examples],
                args.max_new_tokens
            )
            write_json(
                args.run/"comparison.json",
                dict(
                    test_nll_before=before,
                    test_nll_after=after,
                    history=history,
                    before=texts_before,
                    after=texts_after,
                    identity=identity
                )
            )
            print(
                "Test NLL before:", before,
                "| after:", after, flush=True
            )

        write_json(
            args.run/"progress.json",
            dict(
                step=step,
                best_validation_nll=best,
                history=history
            )
        )
        if (
            bench.digest(args.cache/"model.safetensors")
            != bench.MODEL_HASH
        ):
            raise ValueError(
                "Official checkpoint changed externally"
            )
        print(
            "Saved:", checkpoint,
            "| step:", step, flush=True
        )
    finally:
        signal.signal(signal.SIGINT, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path,
        default=Path("runs/_models/mamba130m_1e76775f")
    )
    parser.add_argument(
        "--initial", type=Path,
        default=Path(
            "runs/mamba130m_lora_benchmark/initial_adapter"
        )
    )
    parser.add_argument(
        "--dart", type=Path,
        default=Path("data/dart_cache")
    )
    parser.add_argument(
        "--run", type=Path,
        default=Path("runs/mamba130m_dart_lora")
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--until", type=int)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--train-examples", type=int, default=2000)
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--generation-examples", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    args.until = (
        args.steps if args.until is None else args.until
    )
    if (
        min(
            args.steps, args.accumulation,
            args.train_examples, args.eval_examples,
            args.threads, args.generation_examples,
            args.max_new_tokens
        ) < 1
        or not 1 <= args.until <= args.steps
    ):
        parser.error(
            "Positive dimensions and 1 <= until <= steps required"
        )
    torch.set_num_threads(args.threads)
    train(args)


if __name__ == "__main__":
    main()
