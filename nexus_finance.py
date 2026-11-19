"""Source-controlled financial summaries with the frozen V3 adapter."""
import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import rescore_finance as verifier

NEW = {"amazon", "netflix", "adobe"}
COMPANIES = [
    "msft", "aapl", "googl", "meta",
    "amazon", "netflix", "adobe",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decide(row, text):
    checked = verifier.check(row["reference"], text)
    same_template = (
        verifier.name(verifier.normalize(text))
        == verifier.name(row["reference"])
    )
    errors = list(checked["errors"])
    if not same_template:
        errors.append("outside_controlled_template")

    scale = Decimal(
        1000000 if row["unit"] == "million" else 1000000000
    )
    table = {
        key: format(Decimal(value) / scale, ",f")
        for key, value in row["values_usd"].items()
    }
    return dict(
        status=(
            "source_aligned_template"
            if not errors else "source_table"
        ),
        generated=text,
        displayed_text=text.strip() if not errors else None,
        errors=errors,
        source_values=table,
        unit="USD " + row["unit"],
        company=row["entity"],
        end=row["end"],
        source=row["source"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--company", choices=COMPANIES, required=True
    )
    parser.add_argument(
        "--end", required=True,
        help="Exact ISO fiscal year-end date",
    )
    parser.add_argument(
        "--unit", choices=["million", "billion"],
        default="million",
    )
    parser.add_argument(
        "--saved", action="store_true",
        help="Use a saved holdout response; no model load",
    )
    args = parser.parse_args()

    if args.saved and args.company not in NEW:
        parser.error(
            "--saved supports amazon, netflix and adobe"
        )

    import train_nexus_finance as finance
    base = finance.base
    base.torch.set_num_threads(8)

    root = Path(
        "data/nexus_sec_holdout_2024"
        if args.company in NEW
        else ".migration_archive/source_data/data"
    )
    cache = Path("runs/_models/mamba130m_1e76775f")
    adapter = Path("runs/nexus_finance_v3/best_adapter")
    fingerprints = {
        filename: digest(adapter / filename)
        for filename in (
            "adapter_config.json",
            "adapter_model.safetensors",
        )
    }

    tokenizer = base.AutoTokenizer.from_pretrained(
        str(cache), local_files_only=True
    )
    rows = finance.real_rows(
        SimpleNamespace(
            reports_root=root, companies=[args.company]
        ),
        tokenizer,
    )
    matches = [
        row for row in rows
        if row["end"] == args.end
        and row["unit"] == args.unit
    ]
    if len(matches) != 1:
        available = ", ".join(
            sorted({row["end"] for row in rows})
        )
        parser.error(
            "Requested period unavailable. Dates: " + available
        )
    row = matches[0]

    if args.saved:
        path = Path(
            "runs/nexus_sec_holdout_2024/result.json"
        )
        saved = json.loads(
            path.read_text(encoding="utf-8")
        )
        if saved["identity"]["adapters"]["v3"] != fingerprints:
            raise ValueError(
                "Cached response belongs to another adapter"
            )
        items = [
            item for item in saved["models"]["v3"]
            if item["index"] == row["index"]
        ]
        if (
            len(items) != 1
            or sorted(items[0]["facts"].splitlines()) != sorted(row["prompt"].splitlines())
            or items[0]["reference"] != row["reference"]
            or items[0]["source"] != row["source"]
        ):
            raise ValueError("Cached response source differs")

        text = items[0]["generated"]
        evidence = dict(
            mode="saved_generation",
            file=str(path), sha256=digest(path), facts=items[0]["facts"],
        )
    else:
        if (
            digest(cache / "model.safetensors")
            != base.bench.MODEL_HASH
        ):
            raise ValueError(
                "Official backbone checksum differs"
            )
        config = base.MambaConfig.from_pretrained(
            str(cache), local_files_only=True
        )
        config.use_mambapy = True
        config.use_cache = False
        backbone = base.MambaForCausalLM.from_pretrained(
            str(cache), config=config,
            local_files_only=True,
            torch_dtype=base.torch.float32,
            use_safetensors=True,
        )
        model = base.PeftModel.from_pretrained(
            backbone, str(adapter), is_trainable=False
        )
        model.eval().requires_grad_(False)
        text = base.generate(
            model, tokenizer, [row], 96
        )[0]["generated"]
        evidence = dict(
            mode="live_generation",
            tokens=96, decoding="greedy",
            model_sha256=base.bench.MODEL_HASH,
        )

    record = decide(row, text)
    record.update(
        adapter=dict(
            path=str(adapter), sha256=fingerprints
        ),
        generation=evidence,
        checker_sha256=digest(Path(verifier.__file__)),
        script_sha256=digest(Path(__file__)),
        scope=(
            "Exact source fields and controlled sentence template; "
            "not general factual verification"
        ),
    )

    output = Path("runs/nexus_summaries")
    output.mkdir(parents=True, exist_ok=True)
    filename = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    ) + ".json"
    path = output / filename
    base.write_json(path, record)

    print("\n" + row["entity"] + " | " + row["end"])
    if record["displayed_text"]:
        print("Resume V3 - champs et format conformes a la source:")
        print(record["displayed_text"])
    else:
        print("Resume V3 non retenu. Valeurs de la source SEC:")
        labels = {
            "revenue": "Chiffre d'affaires",
            "operating_income": "Resultat operationnel",
            "net_income": "Resultat net",
        }
        for key, label in labels.items():
            print(
                label + " | "
                + record["source_values"][key]
                + " " + record["unit"]
            )

    print("Source:", row["source"]["filing_url"])
    print("Rapport et reponse brute:", path)


if __name__ == "__main__":
    main()
