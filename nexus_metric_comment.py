"""Per-metric SEC commentary. V5 is opt-in and experimental."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path

import nexus_compare as compare
import nexus_metric as metric


def digest(path):
    return compare.digest(path.read_bytes())


def choose(key, value, generated=None):
    expected = metric.sentence(key, value)
    accepted = (
        generated is not None
        and metric.conforms(generated, expected)
    )
    if accepted:
        reason = "controlled_template_match"
    elif value is None or Decimal(value) == 0:
        reason = "zero_or_undefined"
    elif generated is None:
        reason = "no_neural_request"
    else:
        reason = "controlled_template_mismatch"

    return dict(
        metric=key, value=value,
        generated=generated, expected=expected,
        displayed=generated.strip() if accepted else expected,
        origin="experimental_v5" if accepted else "calculated",
        reason=reason,
    )


def load_experimental():
    import train_nexus_finance_v5 as trainer
    base = trainer.base
    base.torch.set_num_threads(8)

    run = Path("runs/nexus_finance_v5")
    checkpoint = run / "last.pt"
    checksum = digest(checkpoint)
    saved = base.torch.load(
        checkpoint, map_location="cpu", weights_only=True
    )
    identity = saved["identity"]

    audit_path = run / (
        "audit_last_step" + str(saved["step"]) + ".json"
    )
    audit = json.loads(
        audit_path.read_text(encoding="utf-8")
    )
    if audit["identity"]["checkpoint_sha256"] != checksum:
        raise ValueError("Checkpoint differs from audited V5")

    counts = Counter(row["kind"] for row in audit["rows"])
    if counts != {
        "synthetic-comparison": 80,
        "sec-comparison-regression": 35,
        "sec-summary-regression": 28,
    }:
        raise ValueError("Expected completed V5 audit")

    if digest(Path(trainer.__file__)) != identity["script"]:
        raise ValueError("V5 training script changed")

    for module in (
        trainer.old, trainer.comment,
        trainer.compare, trainer.verifier, base,
    ):
        path = Path(module.__file__)
        if digest(path) != identity["scripts"][path.name]:
            raise ValueError(
                "V5 dependency changed: " + path.name
            )

    initial = Path("runs/nexus_finance_v3/best_adapter")
    for name, value in identity["source"].items():
        if digest(initial / name) != value:
            raise ValueError("Initial V3 differs")

    cache = Path("runs/_models/mamba130m_1e76775f")
    if (
        digest(cache / "model.safetensors")
        != base.bench.MODEL_HASH
    ):
        raise ValueError("Official backbone checksum differs")

    tokenizer = base.AutoTokenizer.from_pretrained(
        str(cache), local_files_only=True
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
        backbone, str(initial), is_trainable=False
    )
    base.set_peft_model_state_dict(
        model, saved["adapter"]
    )
    loaded = base.get_peft_model_state_dict(model)
    if (
        set(loaded) != set(saved["adapter"])
        or any(
            not loaded[key].equal(value)
            for key, value in saved["adapter"].items()
        )
    ):
        raise ValueError("V5 checkpoint not loaded exactly")

    model.eval().requires_grad_(False)
    provenance = dict(
        checkpoint=str(checkpoint),
        sha256=checksum,
        step=saved["step"],
        audit_sha256=digest(audit_path),
        selection=json.loads(
            (run / "selection.json").read_text(encoding="utf-8")
        ),
        role=(
            "Experimental specialist; not a globally selected "
            "replacement for V3"
        ),
    )
    return trainer, tokenizer, model, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--company", choices=compare.CIKS, required=True
    )
    parser.add_argument(
        "--experimental-v5", action="store_true"
    )
    args = parser.parse_args()

    root = Path(
        "data/nexus_sec_holdout_2024"
        if args.company in ("amazon", "netflix", "adobe")
        else ".migration_archive/source_data/data"
    )
    report, report_hash = compare.verified(
        root / ("company_" + args.company + "2024"),
        compare.CIKS[args.company],
    )
    metrics = compare.calculate(
        report["current"], report["previous"]
    )
    values = {
        key + "_growth": item["change_percent"]
        for key, item in metrics["changes"].items()
    }
    values.update({
        key + "_change": item["change_percentage_points"]
        for key, item in metrics["margins"].items()
    })
    rounded = {
        key: (
            compare.display(value)
            .replace(" ", "").replace(",", ".")
            if value is not None else None
        )
        for key, value in values.items()
    }

    specialist = None
    if args.experimental_v5:
        print("Chargement du checkpoint V5 audite...", flush=True)
        trainer, tokenizer, model, specialist = load_experimental()

    sentences = []
    for key, value in rounded.items():
        generated = None
        if (
            args.experimental_v5
            and value is not None
            and Decimal(value) != 0
        ):
            row = metric.sample(
                tokenizer, trainer.pack,
                args.company + ":" + key,
                key, value, "sec-comparison-regression",
            )
            generated = trainer.base.generate(
                model, tokenizer, [row], 48
            )[0]["generated"]
        sentences.append(choose(key, value, generated))

    accession = report["accession"]
    url = (
        "https://www.sec.gov/Archives/edgar/data/"
        + str(int(report["cik"])) + "/"
        + accession.replace("-", "") + "/"
        + accession + "-index.html"
    )
    result = dict(
        company=report["entity"],
        current=report["current"],
        previous=report["previous"],
        sentences=sentences,
        metrics=metrics,
        specialist=specialist,
        source=dict(
            filing_url=url,
            filed=report["filed"],
            accession=accession,
            raw_sha256=report["raw_sha256"],
            report_sha256=report_hash,
        ),
        scripts={
            Path(path).name: digest(Path(path))
            for path in (
                __file__, compare.__file__, metric.__file__
            )
        },
        scope=(
            "Per-metric controlled sentences; names/dates "
            "from source; not open-ended analysis"
        ),
        rounding=(
            "2 decimals, ROUND_HALF_UP; percent growth differs "
            "from percentage-point margin change"
        ),
    )

    folder = Path("runs/nexus_metric_comments")
    folder.mkdir(parents=True, exist_ok=True)
    filename = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    ) + ".json"
    path = folder / filename
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)

    print(
        "\n" + report["entity"] + " | "
        + report["previous"]["end"] + " -> "
        + report["current"]["end"]
    )
    print(
        "Mode V5 experimental avec controle et secours."
        if args.experimental_v5
        else "Mode calcule, sans inference neuronale."
    )
    for item in sentences:
        origin = (
            "V5 controle"
            if item["origin"] == "experimental_v5"
            else "Calcule"
        )
        print("[" + origin + "] " + item["displayed"])

    print(
        "Phrases V5 retenues:",
        sum(
            item["origin"] == "experimental_v5"
            for item in sentences
        ),
        "/5",
    )
    print(
        "Valeurs affichees arrondies a deux decimales; "
        "aucune cause deduite."
    )
    print("Source:", url)
    print("Rapport et propositions brutes:", path)


if __name__ == "__main__":
    main()
