"""Controlled annual-change commentary. No adapter training."""
import argparse
from datetime import datetime, timezone
from decimal import Decimal, localcontext, ROUND_HALF_UP
import json
from pathlib import Path
import re

import nexus_compare as compare


def rounded(value):
    if value is None:
        return None
    with localcontext() as context:
        context.prec = 60
        number = Decimal(value).quantize(
            Decimal(".01"), rounding=ROUND_HALF_UP
        )
    return format(
        abs(number) if number == 0 else number, ".2f"
    )


def build(report):
    metrics = compare.calculate(
        report["current"], report["previous"]
    )
    values = {
        key + "_growth_percent": rounded(item["change_percent"])
        for key, item in metrics["changes"].items()
    }
    values.update({
        key + "_change_points": rounded(
            item["change_percentage_points"]
        )
        for key, item in metrics["margins"].items()
    })

    def growth(key):
        value = values[key + "_growth_percent"]
        return (
            "not defined" if value is None
            else value + " percent"
        )

    def margin(key):
        value = values[key + "_change_points"]
        return (
            "not defined" if value is None
            else value + " percentage points"
        )

    text = (
        f"For the year ended {report['current']['end']}, "
        f"{report['entity']} reported "
        f"revenue growth of {growth('revenue')}, "
        f"operating income growth of {growth('operating_income')} "
        f"and net income growth of {growth('net_income')}. "
        f"The operating margin change was "
        f"{margin('operating_margin')}, "
        f"and the net margin change was {margin('net_margin')}."
    )
    entity = report["entity"]
    facts = [
        (entity, "YEAR_END", report["current"]["end"]),
        (entity, "PRIOR_YEAR_END", report["previous"]["end"]),
    ]
    facts += [
        (
            entity, key.upper(),
            value if value is not None else "NOT_DEFINED"
        )
        for key, value in values.items()
    ]
    prompt = (
        "Facts:\n"
        + "\n".join(" | ".join(item) for item in facts)
        + "\nText:"
    )
    return metrics, values, prompt, text


def conforms(generated, reference):
    def clean(text):
        return re.sub(r"\s+", " ", text).strip().casefold()
    return clean(generated) == clean(reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--company", choices=compare.CIKS, required=True
    )
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()

    root = Path(
        "data/nexus_sec_holdout_2024"
        if args.company in ("amazon", "netflix", "adobe")
        else ".migration_archive/source_data/data"
    )
    folder = root / ("company_" + args.company + "2024")
    report, report_hash = compare.verified(
        folder, compare.CIKS[args.company]
    )
    metrics, values, prompt, reference = build(report)
    generated, adapter_hashes = None, None

    if not args.source_only:
        import train_nexus_finance as finance
        base = finance.base
        base.torch.set_num_threads(8)

        cache = Path("runs/_models/mamba130m_1e76775f")
        adapter = Path("runs/nexus_finance_v3/best_adapter")
        adapter_hashes = {
            name: compare.digest((adapter / name).read_bytes())
            for name in (
                "adapter_config.json",
                "adapter_model.safetensors",
            )
        }
        if (
            compare.digest(
                (cache / "model.safetensors").read_bytes()
            ) != base.bench.MODEL_HASH
        ):
            raise ValueError("Official model checksum differs")

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
            backbone, str(adapter), is_trainable=False
        )
        model.eval().requires_grad_(False)

        generated = base.generate(
            model, tokenizer,
            [dict(
                index=args.company,
                prompt=prompt,
                reference=reference,
            )],
            128,
        )[0]["generated"]

        if any(
            compare.digest((adapter / name).read_bytes()) != value
            for name, value in adapter_hashes.items()
        ):
            raise ValueError("Adapter changed")

    accepted = (
        generated is not None
        and conforms(generated, reference)
    )
    status = (
        "controlled_v3_comment" if accepted
        else (
            "calculated_comment" if args.source_only
            else "calculated_fallback"
        )
    )
    result = dict(
        company=report["entity"],
        current_end=report["current"]["end"],
        prior_end=report["previous"]["end"],
        status=status,
        generated=generated,
        displayed_text=(
            generated.strip() if accepted else reference
        ),
        expected_text=reference,
        prompt=prompt,
        rounded_metrics=values,
        metrics=metrics,
        rounding=(
            "2 decimals, ROUND_HALF_UP; growth undefined "
            "if previous <= 0; margin undefined if revenue <= 0"
        ),
        adapter_sha256=adapter_hashes,
        report_sha256=report_hash,
        raw_sha256=report["raw_sha256"],
        scripts={
            name: compare.digest(Path(name).read_bytes())
            for name in (__file__, compare.__file__)
        },
        method=(
            "Exact controlled-template comparison; "
            "no general semantic judgment or causal interpretation"
        ),
    )
    accession = report["accession"]
    result["source_url"] = (
        "https://www.sec.gov/Archives/edgar/data/"
        + str(int(report["cik"])) + "/"
        + accession.replace("-", "") + "/"
        + accession + "-index.html"
    )

    folder = Path("runs/nexus_comments")
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
        "\nCommentaire V3 conforme:"
        if accepted
        else "\nCommentaire calcule depuis les faits SEC:"
    )
    if generated is not None and not accepted:
        print(
            "(Proposition V3 hors format controle; non retenue.)"
        )
    print(result["displayed_text"])
    print(
        "Pourcentages et ecarts en points arrondis "
        "a deux decimales; aucune cause deduite."
    )
    print("Source:", result["source_url"])
    print("Rapport et proposition brute:", path)


if __name__ == "__main__":
    main()
