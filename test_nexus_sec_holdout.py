"""Frozen V1/V3 comparison on new SEC companies. No training."""
import argparse
from datetime import date
from decimal import Decimal
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.request

import train_nexus_finance as finance
import rescore_finance as verifier

PLAN = [
    ("amazon", "0001018724", "2024-12-31"),
    ("netflix", "0001065280", "2024-12-31"),
    ("adobe", "0000796343", "2024-11-29"),
]
CUTOFF = "2025-04-01"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(payload, end):
    fields = {}
    for field, tags in finance.TAGS.items():
        records = {}
        for tag in sorted(tags):
            items = (
                payload["facts"]["us-gaap"]
                .get(tag, {}).get("units", {}).get("USD", [])
            )
            for item in items:
                if (
                    item.get("form") != "10-K"
                    or not item.get("start")
                    or item.get("filed", "9999") > CUTOFF
                ):
                    continue
                days = (
                    date.fromisoformat(item["end"])
                    - date.fromisoformat(item["start"])
                ).days
                if not 330 <= days <= 400:
                    continue

                value = Decimal(str(item["val"]))
                if not value.is_finite():
                    raise ValueError("Nonfinite source fact")

                key = (
                    item["filed"], item["accn"],
                    item["start"], item["end"],
                )
                record = dict(value_usd=str(value), tag=tag)
                if (
                    key in records
                    and records[key]["value_usd"]
                    != record["value_usd"]
                ):
                    raise ValueError(
                        "Conflicting source aliases: " + field
                    )
                records.setdefault(key, record)
        fields[field] = records

    common = set.intersection(
        *(set(records) for records in fields.values())
    )
    available = sorted(k for k in common if k[3] == end)
    if not available:
        raise ValueError(
            "Missing compatible annual facts: "
            + payload["entityName"]
        )
    current = available[-1]
    prior = sorted(
        k for k in common
        if k[:2] == current[:2]
        and 0 <= (
            date.fromisoformat(current[2])
            - date.fromisoformat(k[3])
        ).days <= 14
    )
    if len(prior) != 1:
        raise ValueError(
            "Expected one compatible previous annual period"
        )

    def period(key):
        return dict(
            start=key[2], end=key[3],
            facts={
                field: records[key]
                for field, records in fields.items()
            },
        )

    cik = str(payload["cik"]).zfill(10)
    accession = current[1]
    filing_url = (
        "https://www.sec.gov/Archives/edgar/data/"
        + str(int(cik)) + "/"
        + accession.replace("-", "") + "/"
        + accession + "-index.html"
    )
    return dict(
        entity=payload["entityName"], cik=cik,
        as_of=CUTOFF, filed=current[0],
        accession=accession, currency="USD",
        current=period(current), previous=period(prior[0]),
        filing_url=filing_url,
    )


def prepare(root):
    missing = []
    for symbol, cik, end in PLAN:
        folder = root / ("company_" + symbol + "2024")
        folder.mkdir(parents=True, exist_ok=True)
        source = folder / "companyfacts.json"
        url = (
            "https://data.sec.gov/api/xbrl/companyfacts/CIK"
            + cik + ".json"
        )
        if not source.exists():
            try:
                request = urllib.request.Request(
                    url, headers={
                        "User-Agent": (
                            "Nexus AI research "
                            "https://github.com/fredp74/nexus-ai"
                        )
                    },
                )
                with urllib.request.urlopen(
                    request, timeout=60
                ) as response:
                    raw = response.read(20000001)

                if len(raw) > 20000000:
                    raise ValueError("Source exceeds 20 MB")
                if str(json.loads(raw)["cik"]).zfill(10) != cik:
                    raise ValueError("Wrong SEC response")
                source.write_bytes(raw)
            except Exception as error:
                print("Download unavailable:", type(error).__name__)
                print("Save this URL from your browser:", url)
                print("As:", source.resolve())
                missing.append(symbol)
                continue

        payload = json.loads(source.read_bytes())
        if str(payload["cik"]).zfill(10) != cik:
            raise ValueError("Wrong company source: " + symbol)

        report = select(payload, end)
        report["source_url"] = url
        report["raw_sha256"] = digest(source)
        target = folder / "report.json"

        if target.exists():
            existing = json.loads(
                target.read_text(encoding="utf-8")
            )
            if existing != report:
                raise ValueError(
                    "Frozen report differs: " + symbol
                )
        else:
            target.write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )

    if missing:
        raise SystemExit(
            "Save the missing JSON files, then run "
            "the same command again."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    root = Path("data/nexus_sec_holdout_2024")
    prepare(root)
    base = finance.base
    cache = Path("runs/_models/mamba130m_1e76775f")
    base.torch.set_num_threads(8)

    tokenizer = base.AutoTokenizer.from_pretrained(
        str(cache), local_files_only=True
    )
    rows = finance.real_rows(
        SimpleNamespace(
            reports_root=root,
            companies=[item[0] for item in PLAN],
        ),
        tokenizer,
    )
    assert len(rows) == 12
    for row in rows:
        assert verifier.check(
            row["reference"], row["reference"]
        )["passed"]

    print(
        "Prepared: 12 cases, 3 companies, 2 periods, 2 units.",
        flush=True,
    )
    if args.prepare_only:
        return

    adapters = {
        version: Path(
            "runs/nexus_finance_" + version + "/best_adapter"
        )
        for version in ("v1", "v3")
    }
    weights = digest(cache / "model.safetensors")
    if weights != base.bench.MODEL_HASH:
        raise ValueError("Official backbone checksum differs")

    identity = dict(
        script=digest(Path(__file__)),
        checker=digest(Path(verifier.__file__)),
        builder=digest(Path(finance.__file__)),
        weights=weights, tokens=96,
        adapters={
            version: {
                filename: digest(folder / filename)
                for filename in (
                    "adapter_config.json",
                    "adapter_model.safetensors",
                )
            }
            for version, folder in adapters.items()
        },
        data=hashlib.sha256(
            json.dumps(rows, sort_keys=True).encode()
        ).hexdigest(),
    )

    output = Path("runs/nexus_sec_holdout_2024/result.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(identity=identity, models={})
    if output.exists():
        result = json.loads(
            output.read_text(encoding="utf-8")
        )
        if result["identity"] != identity:
            raise ValueError("Frozen evaluation inputs changed")

    for version, adapter in adapters.items():
        saved = result["models"].setdefault(version, [])
        done = {row["index"] for row in saved}

        if len(done) != len(rows):
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

            for row in rows:
                if row["index"] in done:
                    continue
                generated = base.generate(
                    model, tokenizer, [row], 96
                )[0]
                checked = verifier.check(
                    row["reference"], generated["generated"]
                )
                saved.append(dict(
                    **generated,
                    source=row["source"],
                    check=checked,
                ))
                base.write_json(output, result)
                print(
                    version, len(saved), "/12",
                    row["index"], checked, flush=True,
                )

            del model, backbone
            gc.collect()

        print(
            version,
            sum(row["check"]["passed"] for row in saved),
            "/12", flush=True,
        )

    print("Saved:", output)


if __name__ == "__main__":
    main()
