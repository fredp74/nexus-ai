"""Compute-matched SFT versus unlikelihood on controlled facts."""
import argparse
import difflib
import hashlib
import itertools
import json
import math
import random
import signal
import time
from pathlib import Path

import train_mamba130m_lora as shared
import audit_mamba_faithfulness as audit
import torch
import torch.nn.functional as F
from peft import (
    PeftModel,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, MambaConfig, MambaForCausalLM


def encode(tokenizer, prompt, sentence):
    prefix = tokenizer.encode(prompt)
    answer = tokenizer.encode(" "+sentence)
    if tokenizer.encode(prompt+" "+sentence) != prefix+answer:
        raise ValueError("Token boundary changed")
    return prefix, answer+[tokenizer.eos_token_id]


def record(tokenizer, facts, positive, negative, other_positive, omission):
    prompt = (
        "Facts:\n"
        + "\n".join(" | ".join(t) for t in facts)
        + "\nText:"
    )
    prefix, good = encode(tokenizer, prompt, positive)
    _, bad = encode(tokenizer, prompt, negative)
    _, other = encode(tokenizer, prompt, other_positive)
    if max(len(prefix)+len(good), len(prefix)+len(bad)) > 128:
        raise ValueError("Controlled example exceeds 128 tokens")

    wrong = [False]*len(bad)
    if omission:
        wrong[-1] = True
    else:
        matcher = difflib.SequenceMatcher(
            a=good, b=bad, autojunk=False
        )
        for op, _, _, a, b in matcher.get_opcodes():
            if op != "equal":
                wrong[a:b] = [True]*(b-a)
    if not any(wrong):
        raise ValueError("Missing incorrect target span")

    value_at = next(
        i for i, (a, b) in enumerate(zip(good, other))
        if a != b
    )
    bad_at = next(
        i for i, (a, b) in enumerate(zip(good, bad))
        if a != b
    )
    return dict(
        prompt=prompt,
        positive=positive,
        negative=negative,
        prefix=prefix,
        good=good,
        bad=bad,
        wrong=wrong,
        value_at=value_at,
        other_token=other[value_at],
        bad_at=bad_at
    )


def dataset(tokenizer, train_units, heldout_units):
    first = (
        "Ada Leah Owen Iris Milo Ruth Theo Elsa "
        "Dina Noel Luca Sana Arlo June Lena Otto"
    ).split()
    last = (
        "Bennett Foster Hayes Nash Rowan Spencer Warren Moss "
        "Hayden Bishop Ellis Fox Grant West Blake Walsh"
    ).split()
    adjectives = (
        "Copper Silver Crimson Golden Quiet Hidden Rustic Sunny "
        "Misty Green Blue Red Purple White Black Maple"
    ).split()
    nouns = (
        "Oak Pine Elm Birch Cove Ridge Valley Brook "
        "Hill Meadow Garden Grove Falcon Eagle Sparrow Heron"
    ).split()
    cities = (
        "Naples Genoa Turin Verona Ghent Bruges Bergen Oslo "
        "Lund Uppsala Malmo Aarhus Toledo Seville Valencia Bilbao"
    ).split()
    companies = [a+" Voyages" for a in adjectives]
    engines = [a+" Motors" for a in nouns]
    forbidden = {
        r["entity"].casefold() for r in audit.cases()
    }
    pools = dict(
        birthplace=[
            " ".join(p)
            for p in itertools.product(first, last)
        ],
        price=[
            "The "+" ".join(p)
            for p in itertools.product(adjectives, nouns)
        ],
        owner=[
            "MV "+" ".join(p)
            for p in itertools.product(adjectives, nouns)
        ]
    )
    result = {s: [] for s in ("train", "dev", "test")}

    for family, names in pools.items():
        names = [
            n for n in names if n.casefold() not in forbidden
        ]
        random.Random(91).shuffle(names)
        need = train_units+2*heldout_units
        if len(names) < need:
            raise ValueError("Too many requested entity groups")

        for index, name in enumerate(names[:need]):
            split = (
                "train" if index < train_units
                else "dev" if index < train_units+heldout_units
                else "test"
            )
            rng = random.Random(
                7000+index+list(pools).index(family)*1000
            )
            subtype = index % 4
            style = index % 3

            if family == "birthplace":
                values = rng.sample(cities, 2)
                facts = [
                    [(name, "BIRTH_PLACE", v)]
                    for v in values
                ]
                templates = [
                    lambda v: f"{name} was born in {v}.",
                    lambda v: f"{name}'s birthplace is {v}.",
                    lambda v: f"{name} has {v} as a birthplace.",
                ]
                positives = [
                    templates[style](v) for v in values
                ]
            elif family == "price":
                numbers = rng.sample(list(range(5, 100, 5)), 2)
                values = [
                    f"less than \u00a3{numbers[0]}",
                    f"more than \u00a3{numbers[1]}"
                ]
                facts = [
                    [
                        (name, "eatType", "restaurant"),
                        (name, "priceRange", v)
                    ]
                    for v in values
                ]
                templates = [
                    lambda v: (
                        f"{name} is a restaurant with a price range of {v}."
                    ),
                    lambda v: f"{name} is a restaurant priced at {v}.",
                    lambda v: (
                        f"The restaurant {name} has a price range of {v}."
                    ),
                ]
                positives = [
                    templates[style](v) for v in values
                ]
            else:
                values = rng.sample(companies, 2)
                engine = rng.choice(engines)
                facts = [
                    [
                        (name, "OWNER", v),
                        (name, "POWER_TYPE", engine)
                    ]
                    for v in values
                ]
                templates = [
                    lambda v: (
                        f"{name} is owned by {v} and powered by {engine}."
                    ),
                    lambda v: (
                        f"{v} owns {name}, which is powered by {engine}."
                    ),
                    lambda v: (
                        f"{name} is powered by {engine} and owned by {v}."
                    ),
                ]
                positives = [
                    templates[style](v) for v in values
                ]

            rows = []
            for variant in (0, 1):
                positive = positives[variant]
                if subtype == 0:
                    negative = positives[1-variant]
                elif subtype == 1:
                    if family == "birthplace":
                        negative = (
                            positive
                            .replace("was born in", "is a birthplace in")
                            .replace("'s birthplace", "'s workplace")
                            .replace("as a birthplace", "as a workplace")
                        )
                    elif family == "price":
                        negative = positive.replace(
                            "restaurant", "city"
                        )
                    else:
                        negative = (
                            positive
                            .replace(values[variant], "<SWAP>")
                            .replace(engine, values[variant])
                            .replace("<SWAP>", engine)
                        )
                elif subtype == 2:
                    extra = (
                        " is located in London."
                        if family == "price"
                        else " was born in 1987."
                        if family == "birthplace"
                        else " is operated by another company."
                    )
                    negative = positive+" "+name+extra
                else:
                    negative = (
                        f"{name} was born."
                        if family == "birthplace"
                        else f"{name} is a restaurant."
                        if family == "price"
                        else f"{name} is powered by {engine}."
                    )
                rows.append(record(
                    tokenizer,
                    facts[variant],
                    positive,
                    negative,
                    positives[1-variant],
                    subtype == 3
                ))
            result[split].append(dict(
                family=family,
                entity=name,
                negative_type=subtype,
                cases=rows
            ))
    return result


def tensors(row):
    sequences = [
        row["prefix"]+row[k] for k in ("good", "bad")
    ]
    length = max(map(len, sequences))
    x = torch.zeros(2, length, dtype=torch.long)
    labels = torch.full_like(x, -100)
    attention = torch.zeros_like(x)
    for i, seq in enumerate(sequences):
        x[i, :len(seq)] = torch.tensor(seq)
        attention[i, :len(seq)] = 1
        labels[i, len(row["prefix"]):len(seq)] = torch.tensor(
            row["good" if i == 0 else "bad"]
        )
    wrong = torch.zeros(length-1, dtype=torch.bool)
    for i, flag in enumerate(row["wrong"]):
        if flag:
            wrong[len(row["prefix"])+i-1] = True
    return x, labels, attention, wrong


def losses(model, row):
    x, labels, attention, wrong = tensors(row)
    logits = model(
        input_ids=x,
        attention_mask=attention,
        use_cache=False
    ).logits[:, :-1]
    targets = labels[:, 1:]
    logp = F.log_softmax(logits, dim=-1)
    chosen = logp.gather(
        -1, targets.clamp_min(0).unsqueeze(-1)
    ).squeeze(-1)
    positive = -chosen[0][targets[0] != -100].mean()
    bad_p = chosen[1][wrong].exp().clamp(max=1-1e-6)
    unlikelihood = -torch.log1p(-bad_p).mean()

    offset = len(row["prefix"])-1
    v = row["value_at"]
    b = row["bad_at"]
    value_ok = (
        logits[0, offset+v, row["good"][v]]
        > logits[0, offset+v, row["other_token"]]
    )
    negative_ok = (
        logits[0, offset+b, row["good"][b]]
        > logits[0, offset+b, row["bad"][b]]
    )
    return (
        positive,
        unlikelihood,
        bool(value_ok),
        bool(negative_ok),
        bad_p.mean().item()
    )


@torch.no_grad()
def evaluate(model, units):
    model.eval()
    nll = bad = 0.
    passed = value_pairs = 0
    for unit in units:
        values, checks = [], []
        for row in unit["cases"]:
            ce, _, v, c, p = losses(model, row)
            nll += ce.item()
            bad += p
            values.append(v)
            checks.append(v and c)
        passed += int(all(checks))
        value_pairs += int(all(values))
    model.train()
    return dict(
        units=len(units),
        both_preferences=passed,
        value_pairs=value_pairs,
        positive_nll=nll/(2*len(units)),
        wrong_token_probability=bad/(2*len(units))
    )


@torch.no_grad()
def generations(model, tokenizer, units):
    model.eval()
    result = []
    for family in ("birthplace", "price", "owner"):
        unit = next(
            u for u in units if u["family"] == family
        )
        for row in unit["cases"]:
            x = tokenizer(
                row["prompt"], return_tensors="pt"
            )["input_ids"]
            out = model.generate(
                input_ids=x,
                attention_mask=torch.ones_like(x),
                max_new_tokens=64,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
            result.append(dict(
                family=family,
                facts=row["prompt"],
                reference=row["positive"],
                generated=tokenizer.decode(
                    out[0, x.shape[1]:],
                    skip_special_tokens=True
                )
            ))
    model.train()
    return result


def arm(
    model, initial, params, data, args,
    name, weight, identity, stopped
):
    directory = args.run/name
    directory.mkdir(parents=True, exist_ok=True)
    set_peft_model_state_dict(model, initial)
    optimizer = torch.optim.AdamW(
        params, lr=.0001, weight_decay=0.
    )
    rng = random.Random(42)
    torch.manual_seed(42)
    step, best, history = 0, None, []
    checkpoint = directory/"last.pt"
    if checkpoint.exists():
        saved = torch.load(
            checkpoint, map_location="cpu", weights_only=True
        )
        if (
            saved["identity"] != identity
            or saved["weight"] != weight
        ):
            raise ValueError("Resume settings changed")
        set_peft_model_state_dict(model, saved["adapter"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.setstate(saved["rng"])
        torch.set_rng_state(saved["torch_rng"])
        step = saved["step"]
        best = saved["best"]
        history = saved["history"]

    def snapshot():
        return dict(
            identity=identity,
            weight=weight,
            step=step,
            best=best,
            history=history,
            adapter=get_peft_model_state_dict(model),
            optimizer=optimizer.state_dict(),
            rng=rng.getstate(),
            torch_rng=torch.get_rng_state()
        )

    def validate():
        nonlocal best
        metric = evaluate(model, data["dev"])
        history.append(dict(step=step, metric=metric))
        score = (
            metric["both_preferences"],
            -metric["positive_nll"]
        )
        if best is None or score > tuple(best):
            best = score
            shared.save(
                directory/"best.pt",
                dict(
                    step=step,
                    adapter=get_peft_model_state_dict(model),
                    metric=metric
                )
            )
        print(
            name, "step", step,
            "validation:", metric, flush=True
        )

    if not checkpoint.exists():
        validate()
        shared.save(checkpoint, snapshot())

    started, start_step = time.monotonic(), step
    while step < args.until and not stopped[0]:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        unit = rng.choice(data["train"])
        ce_sum = ul_sum = 0.
        for row in unit["cases"]:
            ce, ul, _, _, _ = losses(model, row)
            loss = (ce+weight*ul)/2
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            loss.backward()
            ce_sum += ce.item()/2
            ul_sum += ul.item()/2

        norm = torch.nn.utils.clip_grad_norm_(params, 1.)
        if not torch.isfinite(norm):
            raise ValueError("Nonfinite gradient")
        warmup = min(20, max(1, args.updates//10))
        progress = max(
            0.,
            (step+1-warmup)/max(1, args.updates-warmup)
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
                f"{name} {step}/{args.updates} "
                f"| CE={ce_sum:.4f} UL={ul_sum:.4f} "
                f"| arm ETA={(args.until-step)*seconds/60:.1f} min",
                flush=True
            )
        if step % 25 == 0 or step == args.until:
            validate()
        if (
            step % 10 == 0
            or step == args.until
            or stopped[0]
        ):
            shared.save(checkpoint, snapshot())

    shared.save(checkpoint, snapshot())
    if step == args.updates and not stopped[0]:
        selected = torch.load(
            directory/"best.pt",
            map_location="cpu",
            weights_only=True
        )
        set_peft_model_state_dict(
            model, selected["adapter"]
        )
        model.save_pretrained(
            directory/"best_adapter",
            safe_serialization=True
        )
    return step, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path,
        default=Path("runs/_models/mamba130m_1e76775f")
    )
    parser.add_argument(
        "--source", type=Path,
        default=Path("runs/mamba130m_dart_lora/best_adapter")
    )
    parser.add_argument(
        "--run", type=Path,
        default=Path("runs/mamba130m_faithfulness_training")
    )
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--until", type=int)
    parser.add_argument("--train-units", type=int, default=64)
    parser.add_argument("--heldout-units", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    args.until = (
        args.updates if args.until is None else args.until
    )
    if (
        min(
            args.updates, args.train_units,
            args.heldout_units, args.threads
        ) < 1
        or not 1 <= args.until <= args.updates
    ):
        parser.error(
            "Positive dimensions and until <= updates required"
        )
    torch.set_num_threads(args.threads)
    if (
        shared.bench.digest(args.cache/"model.safetensors")
        != shared.bench.MODEL_HASH
    ):
        raise ValueError("Official weights changed")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.cache), local_files_only=True
    )
    data = dataset(
        tokenizer, args.train_units, args.heldout_units
    )
    payload = json.dumps(
        data, sort_keys=True, ensure_ascii=True
    ).encode()
    identity = dict(
        format="mamba-faithfulness-paired-v1",
        data_hash=hashlib.sha256(payload).hexdigest(),
        source_hash=shared.bench.digest(
            args.source/"adapter_model.safetensors"
        ),
        source_config_hash=shared.bench.digest(
            args.source/"adapter_config.json"
        ),
        script_hash=shared.bench.digest(Path(__file__)),
        updates=args.updates,
        training=(
            "synthetic controlled references; paired contexts; "
            "identical compute graphs"
        ),
        contrastive_weight=.5
    )
    args.run.mkdir(parents=True, exist_ok=True)
    data_path = args.run/"data.json"
    if (
        data_path.exists()
        and data_path.read_bytes() != payload
    ):
        raise ValueError("Existing dataset differs")
    if not data_path.exists() and any(args.run.iterdir()):
        raise FileExistsError("Use a new run directory")
    data_path.write_bytes(payload)

    config = MambaConfig.from_pretrained(
        str(args.cache), local_files_only=True
    )
    config.use_mambapy = True
    config.use_cache = False
    backbone = MambaForCausalLM.from_pretrained(
        str(args.cache),
        config=config,
        local_files_only=True,
        torch_dtype=torch.float32,
        use_safetensors=True
    )
    model = PeftModel.from_pretrained(
        backbone, str(args.source), is_trainable=True
    )
    params = [
        p for p in model.parameters()
        if p.requires_grad
    ]
    if sum(p.numel() for p in params) != 442368:
        raise ValueError("Unexpected trainable parameters")
    initial = {
        k: v.detach().clone()
        for k, v in get_peft_model_state_dict(model).items()
    }
    stopped = [False]
    previous = signal.signal(
        signal.SIGINT,
        lambda *_: stopped.__setitem__(0, True)
    )
    try:
        complete = True
        for name, weight in [
            ("control", 0.),
            ("contrastive", .5)
        ]:
            step, _ = arm(
                model, initial, params, data, args,
                name, weight, identity, stopped
            )
            complete = complete and step == args.updates
            if stopped[0]:
                break

        if complete and not stopped[0]:
            results = {}
            for name in ("before", "control", "contrastive"):
                if name == "before":
                    set_peft_model_state_dict(model, initial)
                else:
                    selected = torch.load(
                        args.run/name/"best.pt",
                        map_location="cpu",
                        weights_only=True
                    )
                    set_peft_model_state_dict(
                        model, selected["adapter"]
                    )
                results[name] = dict(
                    test=evaluate(model, data["test"]),
                    generations=generations(
                        model, tokenizer, data["test"]
                    )
                )
                print(
                    name, "TEST:",
                    results[name]["test"], flush=True
                )
            shared.write_json(
                args.run/"comparison.json",
                dict(identity=identity, results=results)
            )

        if (
            shared.bench.digest(
                args.source/"adapter_model.safetensors"
            ) != identity["source_hash"]
        ):
            raise ValueError(
                "Source adapter changed externally"
            )
        print("Saved:", args.run, flush=True)
    finally:
        signal.signal(signal.SIGINT, previous)


if __name__ == "__main__":
    main()
