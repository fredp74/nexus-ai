"""Reserved counterfactual audit: preference and free generation."""
import argparse
import csv
from contextlib import nullcontext
import json
from pathlib import Path

import benchmark_mamba130m_lora as bench
import torch
from peft import PeftModel
from transformers import AutoTokenizer, MambaConfig, MambaForCausalLM


def cases():
    result = []
    for name, a, b in [
        ("Nora Vale", "Oxford", "Dover"),
        ("Mira Stone", "Bristol", "York"),
        ("Evan Reed", "Durham", "Exeter"),
    ]:
        result.append(dict(
            kind="birthplace",
            entity=name,
            values=[a, b],
            facts=[
                f"{name} | BIRTH_PLACE | {v}"
                for v in (a, b)
            ],
            sentences=[
                f"{name} was born in {v}."
                for v in (a, b)
            ]
        ))
    for name, lo, hi in [
        ("The Kestrel Lantern", 20, 30),
        ("The Amber Harbor", 15, 40),
        ("The Cedar Orchard", 25, 50),
    ]:
        values = [
            f"less than \u00a3{lo}",
            f"more than \u00a3{hi}"
        ]
        result.append(dict(
            kind="price",
            entity=name,
            values=values,
            facts=[
                f"{name} | eatType | restaurant\n"
                f"{name} | priceRange | {v}"
                for v in values
            ],
            sentences=[
                f"{name} is a restaurant with a price range of {v}."
                for v in values
            ]
        ))
    for name, a, b in [
        ("Cedarwind", "Atlas Cruises", "Beacon Cruises"),
        ("Amberwake", "Cobalt Cruises", "Harbor Cruises"),
        ("Silverharbor", "Cedar Cruises", "Mariner Cruises"),
    ]:
        result.append(dict(
            kind="owner",
            entity=name,
            values=[a, b],
            facts=[
                f"{name} | OWNER | {v}\n"
                f"{name} | POWER_TYPE | Meridian Engines"
                for v in (a, b)
            ],
            sentences=[
                f"{name} is owned by {v} and powered by Meridian Engines."
                for v in (a, b)
            ]
        ))
    return result


@torch.no_grad()
def nll(model, tokenizer, prompt, sentence):
    prefix = tokenizer.encode(prompt)
    answer = (
        tokenizer.encode(" "+sentence)
        + [tokenizer.eos_token_id]
    )
    if (
        tokenizer.encode(prompt+" "+sentence)
        != prefix+answer[:-1]
    ):
        raise ValueError("Unexpected token boundary")
    ids = torch.tensor([prefix+answer])
    labels = torch.tensor([
        [-100]*len(prefix)+answer
    ])
    loss = model(
        input_ids=ids,
        labels=labels,
        use_cache=False
    ).loss
    return loss.item()*len(answer)


@torch.no_grad()
def continuation(model, tokenizer, prompt, limit):
    ids = tokenizer(
        prompt, return_tensors="pt"
    )["input_ids"]
    output = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=limit,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(
        output[0, ids.shape[1]:],
        skip_special_tokens=True
    ).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path,
        default=Path("runs/_models/mamba130m_1e76775f")
    )
    parser.add_argument(
        "--adapter", type=Path,
        default=Path("runs/mamba130m_dart_lora/best_adapter")
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/mamba130m_faithfulness_audit")
    )
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--pairs", type=int, default=9)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if (
        not 1 <= args.pairs <= 9
        or min(args.tokens, args.threads) < 1
        or args.output.exists()
    ):
        parser.error(
            "Require 1-9 pairs, positive tokens/threads "
            "and a new output directory"
        )
    torch.set_num_threads(args.threads)
    if (
        bench.digest(args.cache/"model.safetensors")
        != bench.MODEL_HASH
    ):
        raise ValueError("Official weights checksum mismatch")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.cache), local_files_only=True
    )
    config = MambaConfig.from_pretrained(
        str(args.cache), local_files_only=True
    )
    config.use_mambapy = True
    backbone = MambaForCausalLM.from_pretrained(
        str(args.cache),
        config=config,
        local_files_only=True,
        use_safetensors=True,
        torch_dtype=torch.float32
    )
    model = PeftModel.from_pretrained(
        backbone,
        str(args.adapter),
        is_trainable=False
    )
    model.eval().requires_grad_(False)
    adapter_hash = bench.digest(
        args.adapter/"adapter_model.safetensors"
    )
    selected = cases()[:args.pairs]
    args.output.mkdir(parents=True)
    result = dict(
        model_revision=bench.REVISION,
        adapter_sha256=adapter_hash,
        purpose=(
            "Reserved audit only; keep these entities and "
            "pairs out of subsequent training."
        ),
        note=(
            "Forced-choice success is not free-generation "
            "factual accuracy. Lexical checks are only hints."
        ),
        cases=selected,
        modes={}
    )
    review = []

    for mode in ("base", "lora"):
        rows, summaries = [], {}
        manager = (
            model.disable_adapter()
            if mode == "base"
            else nullcontext()
        )
        with manager, torch.inference_mode():
            for pair_id, pair in enumerate(selected):
                pair_rows = []
                for variant in (0, 1):
                    prompt = (
                        "Facts:\n"
                        + pair["facts"][variant]
                        + "\nText:"
                    )
                    correct = nll(
                        model, tokenizer, prompt,
                        pair["sentences"][variant]
                    )
                    wrong = nll(
                        model, tokenizer, prompt,
                        pair["sentences"][1-variant]
                    )
                    text = continuation(
                        model, tokenizer, prompt, args.tokens
                    )
                    row = dict(
                        pair=pair_id,
                        kind=pair["kind"],
                        variant=variant,
                        facts=prompt,
                        expected_sentence=pair["sentences"][variant],
                        alternative_sentence=pair["sentences"][1-variant],
                        correct_sentence_nll=correct,
                        alternative_sentence_nll=wrong,
                        correct_sentence_preferred=correct < wrong,
                        margin=wrong-correct,
                        generated=text,
                        current_value_literal_present=(
                            pair["values"][variant].casefold()
                            in text.casefold()
                        ),
                        opposite_value_literal_present=(
                            pair["values"][1-variant].casefold()
                            in text.casefold()
                        )
                    )
                    pair_rows.append(row)
                    review.append(dict(
                        model=mode,
                        pair=pair_id,
                        variant=variant,
                        kind=pair["kind"],
                        facts=prompt,
                        expected_sentence=row["expected_sentence"],
                        generated=text,
                        missing_facts="",
                        unsupported_claims="",
                        wrong_relations=""
                    ))

                rows.extend(pair_rows)
                item = summaries.setdefault(
                    pair["kind"],
                    dict(
                        pairs=0,
                        both_sentences_preferred=0,
                        changed_generations=0
                    )
                )
                item["pairs"] += 1
                item["both_sentences_preferred"] += int(
                    all(
                        r["correct_sentence_preferred"]
                        for r in pair_rows
                    )
                )
                item["changed_generations"] += int(
                    pair_rows[0]["generated"]
                    != pair_rows[1]["generated"]
                )
                result["modes"][mode] = dict(
                    summary=summaries, rows=rows
                )
                (args.output/"result.json").write_text(
                    json.dumps(
                        result, indent=2, ensure_ascii=False
                    ),
                    encoding="utf-8"
                )
                print(
                    mode, pair["kind"],
                    pair_id+1, "/", len(selected),
                    "| correct forced choice:",
                    sum(
                        r["correct_sentence_preferred"]
                        for r in pair_rows
                    ),
                    "/2",
                    flush=True
                )
        print(
            mode, "SUMMARY:", summaries,
            flush=True
        )

    with (args.output/"review.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(review[0])
        )
        writer.writeheader()
        writer.writerows(review)

    if (
        bench.digest(args.adapter/"adapter_model.safetensors")
        != adapter_hash
    ):
        raise ValueError("Adapter changed externally")
    print(
        "Saved:", args.output/"result.json",
        flush=True
    )
    print(
        "Human review:", args.output/"review.csv",
        flush=True
    )


if __name__ == "__main__":
    main()
