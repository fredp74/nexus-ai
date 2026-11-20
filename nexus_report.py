"""One SEC report: source facts, V3 summary, optional V5 commentary."""
import argparse
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import gc
import json
from pathlib import Path

import nexus_compare as compare
import nexus_finance as summaries
import nexus_metric as metric
import nexus_metric_comment as comments


def digest(path):
    return compare.digest(path.read_bytes())


def source_row(report, unit):
    period = report["current"]
    values = {
        key: period["facts"][key]["value_usd"]
        for key in compare.FIELDS
    }
    scale = Decimal(
        1000000 if unit == "million" else 1000000000
    )
    with localcontext() as context:
        context.prec = 50
        amounts = {}
        for key, value in values.items():
            text = format(Decimal(value) / scale, ",f")
            amounts[key] = (
                text.rstrip("0").rstrip(".")
                if "." in text else text
            )

    reference = (
        f"For the year ended {period['end']}, "
        f"{report['entity']} reported "
        f"revenue of USD {amounts['revenue']} {unit}, "
        f"operating income of USD {amounts['operating_income']} {unit}, "
        f"and net income of USD {amounts['net_income']} {unit}."
    )
    return dict(
        entity=report["entity"], end=period["end"],
        unit=unit, values_usd=values, reference=reference,
        source=dict(
            raw_sha256=report["raw_sha256"],
            filing_url=report["filing_url"],
        ),
    )


def live_summary(report, unit):
    import train_nexus_finance as finance
    base = finance.base
    base.torch.set_num_threads(8)

    cache = Path("runs/_models/mamba130m_1e76775f")
    adapter = Path("runs/nexus_finance_v3/best_adapter")
    hashes = {
        name: digest(adapter / name)
        for name in (
            "adapter_config.json",
            "adapter_model.safetensors",
        )
    }
    if (
        digest(cache / "model.safetensors")
        != base.bench.MODEL_HASH
    ):
        raise ValueError("Official backbone checksum differs")

    tokenizer = base.AutoTokenizer.from_pretrained(
        str(cache), local_files_only=True
    )
    current = report["current"]
    row = finance.row(
        tokenizer, report["entity"], current["end"],
        {
            key: current["facts"][key]["value_usd"]
            for key in finance.FIELDS
        },
        unit, 0, 0 if unit == "million" else 1,
        dict(
            raw_sha256=report["raw_sha256"],
            filing_url=report["filing_url"],
        ),
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
        model, tokenizer, [row], 96
    )[0]["generated"]

    decision = summaries.decide(row, generated)
    decision.update(
        prompt=row["prompt"],
        reference=row["reference"],
        adapter_sha256=hashes,
        model_sha256=base.bench.MODEL_HASH,
        source_builder_sha256=digest(Path(finance.__file__)),
    )
    for name, value in hashes.items():
        if digest(adapter / name) != value:
            raise ValueError("V3 adapter changed")

    del model, backbone, tokenizer
    gc.collect()
    return decision


def metric_comments(metrics, experimental):
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
    if experimental:
        print("Chargement V5 experimental...", flush=True)
        trainer, tokenizer, model, specialist = (
            comments.load_experimental()
        )

    sentences = []
    for key, value in rounded.items():
        generated, prompt = None, None
        if (
            experimental
            and value is not None
            and Decimal(value) != 0
        ):
            row = metric.sample(
                tokenizer, trainer.pack, key,
                key, value, "sec-comparison-regression",
            )
            prompt = row["prompt"]
            generated = trainer.base.generate(
                model, tokenizer, [row], 48
            )[0]["generated"]

        item = comments.choose(key, value, generated)
        item["prompt"] = prompt
        sentences.append(item)

    if experimental:
        del model, tokenizer
        gc.collect()
    return sentences, specialist


def safe(text):
    return (
        str(text).replace("&", "&amp;")
        .replace("<", "&lt;").replace(">", "&gt;")
        .replace("|", "\\|").replace("\n", " ")
    )


def render(record, json_name):
    report = record["report"]
    before, now = report["previous"], report["current"]
    scale = Decimal(
        1000000 if record["unit"] == "million"
        else 1000000000
    )
    unit = (
        "millions USD" if record["unit"] == "million"
        else "milliards USD"
    )
    lines = [
        "# Nexus AI - " + safe(report["entity"]), "",
        "## Periodes et synthese", "",
        "- Exercice precedent : "
        + before["start"] + " au " + before["end"] + ".",
        "- Exercice courant : "
        + now["start"] + " au " + now["end"] + ".",
        "- Unite du tableau : " + unit + ".", "",
    ]

    summary = record["summary"]
    if summary.get("displayed_text"):
        lines += [
            "**Resume V3 : champs et format conformes a la source.**",
            "", safe(summary["displayed_text"]), "",
        ]
    elif record["mode"] == "source_only":
        lines += [
            "**Synthese calculee, sans inference neuronale.**",
            "", safe(summary["reference"]), "",
        ]
    else:
        lines += [
            "**Resume V3 non retenu : les chiffres sources "
            "sont presentes ci-dessous.**", "",
        ]

    lines += [
        "## Chiffres et variations", "",
        "| Champ | Precedent | Courant | Variation | Variation (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in compare.FIELDS.items():
        item = record["metrics"]["changes"][key]
        cells = [
            label,
            compare.display(item["previous_usd"], scale),
            compare.display(item["current_usd"], scale),
            compare.display(item["change_usd"], scale),
            compare.display(item["change_percent"]),
        ]
        lines.append(
            "| " + " | ".join(safe(cell) for cell in cells) + " |"
        )

    lines += [
        "", "## Marges", "",
        "| Marge | Precedente (%) | Courante (%) | Ecart (points) |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("operating_margin", "Operationnelle"),
        ("net_margin", "Nette"),
    ):
        item = record["metrics"]["margins"][key]
        cells = [
            label,
            compare.display(item["previous_percent"]),
            compare.display(item["current_percent"]),
            compare.display(item["change_percentage_points"]),
        ]
        lines.append(
            "| " + " | ".join(safe(cell) for cell in cells) + " |"
        )

    lines += ["", "## Commentaire des variations", ""]
    for item in record["sentences"]:
        origin = (
            "V5 experimental, controle"
            if item["origin"] == "experimental_v5"
            else "Calcule depuis les faits"
        )
        lines.append(
            "- **" + origin + " :** " + safe(item["displayed"])
        )

    lines += [
        "", "## Source et perimetre", "",
        "- [Depot SEC 10-K]("
        + record["source"]["filing_url"]
        + "), depose le " + report["filed"] + ".",
        "- Date limite de depot retenue : "
        + report["as_of"] + ".",
        "- [Rapport JSON et provenance](" + json_name + ").",
        "",
        "Les tableaux et phrases de variation sont arrondis "
        "a deux decimales. Le resume conserve les montants fournis. "
        "Les variations sont calculees avant arrondi. "
        "Une variation de marge est exprimee en points, "
        "pas en pourcentage.",
        "",
        "n/d : croissance non definie si le montant precedent "
        "est nul ou negatif ; marge non definie si le chiffre "
        "d'affaires est nul ou negatif.",
        "",
        "Exercices fiscaux compares sans ajustement de duree. "
        "Perimetre : chiffre d'affaires, resultats et marges. "
        "Cette fiche ne couvre pas encore le bilan, le cash-flow "
        "ou la valorisation et ne deduit aucune cause.",
        "",
        "V5 reste un specialiste experimental ; elle ne remplace "
        "pas V3 dans l'application.",
        "",
        "A research project by "
        "[Algotradingresearch.com](https://algotradingresearch.com).",
        "",
    ]
    return "\n".join(lines)


def atomic(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--company", choices=compare.CIKS, required=True
    )
    parser.add_argument(
        "--unit", choices=["million", "billion"],
        default="million",
    )
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--experimental-v5", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.source_only and args.experimental_v5:
        parser.error(
            "--source-only and --experimental-v5 "
            "cannot be combined"
        )

    root = Path(
        "data/nexus_sec_holdout_2024"
        if args.company in ("amazon", "netflix", "adobe")
        else ".migration_archive/source_data/data"
    )
    report_path = (
        root / ("company_" + args.company + "2024")
        / "report.json"
    )
    report, report_hash = compare.verified(
        report_path.parent, compare.CIKS[args.company]
    )

    accession = report["accession"]
    url = (
        "https://www.sec.gov/Archives/edgar/data/"
        + str(int(report["cik"])) + "/"
        + accession.replace("-", "") + "/"
        + accession + "-index.html"
    )
    report["filing_url"] = url
    metrics = compare.calculate(
        report["current"], report["previous"]
    )

    if args.source_only:
        row = source_row(report, args.unit)
        summary = dict(
            status="calculated_summary",
            reference=row["reference"],
            generated=None, displayed_text=None,
        )
    else:
        print(
            "Generation et controle du resume V3...",
            flush=True,
        )
        summary = live_summary(report, args.unit)

    sentences, specialist = metric_comments(
        metrics, args.experimental_v5
    )
    record = dict(
        company=args.company,
        mode="source_only" if args.source_only else "v3_summary",
        experimental_v5=args.experimental_v5,
        unit=args.unit,
        report=report,
        metrics=metrics,
        summary=summary,
        sentences=sentences,
        specialist=specialist,
        source=dict(
            report_path=str(report_path),
            report_sha256=report_hash,
            raw_sha256=report["raw_sha256"],
            filing_url=url,
        ),
        scripts={
            Path(path).name: digest(Path(path))
            for path in (
                __file__, compare.__file__,
                summaries.__file__, metric.__file__,
                comments.__file__,
            )
        },
        calculation_precision=50,
        display_decimals=2,
    )

    folder = Path("runs/nexus_reports")
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    stem = (
        args.company + "_" + report["current"]["end"]
        + "_" + stamp
    )
    md_path = folder / (stem + ".md")
    json_path = folder / (stem + ".json")
    if md_path.exists() or json_path.exists():
        raise FileExistsError("Report already exists")

    text = render(record, json_path.name)
    record["markdown_sha256"] = compare.digest(
        text.encode("utf-8")
    )
    record["created_utc"] = stamp
    atomic(
        json_path,
        json.dumps(record, indent=2, ensure_ascii=False),
    )
    atomic(md_path, text)

    print("\nFiche Nexus :", report["entity"])
    print("Resume :", summary["status"])
    print(
        "Phrases V5 retenues :",
        sum(
            item["origin"] == "experimental_v5"
            for item in sentences
        ),
        "/5",
    )
    print("Markdown :", md_path)
    print("JSON     :", json_path)
    if args.show:
        print("\n" + text)


if __name__ == "__main__":
    main()
