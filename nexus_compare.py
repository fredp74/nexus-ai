"""Compare annual SEC periods using local facts and Decimal."""
import argparse
from datetime import date, datetime, timezone
from decimal import Decimal, localcontext, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path

FIELDS = {
    "revenue": "Chiffre d'affaires",
    "operating_income": "Resultat operationnel",
    "net_income": "Resultat net",
}
TAGS = {
    "revenue": {
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues", "SalesRevenueNet",
    },
    "operating_income": {"OperatingIncomeLoss"},
    "net_income": {"NetIncomeLoss"},
}
CIKS = {
    "msft": "0000789019", "aapl": "0000320193",
    "googl": "0001652044", "meta": "0001326801",
    "amazon": "0001018724", "netflix": "0001065280",
    "adobe": "0000796343",
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def verified(folder, cik):
    report_bytes = (folder / "report.json").read_bytes()
    report = json.loads(report_bytes)
    raw_bytes = (folder / "companyfacts.json").read_bytes()
    if digest(raw_bytes) != report["raw_sha256"]:
        raise ValueError("SEC snapshot checksum differs")

    raw = json.loads(raw_bytes)
    if (
        str(raw["cik"]).zfill(10) != cik
        or str(report["cik"]).zfill(10) != cik
        or raw["entityName"] != report["entity"]
        or report["currency"] != "USD"
    ):
        raise ValueError("SEC company/currency mismatch")

    filed = date.fromisoformat(report["filed"])
    if filed > date.fromisoformat(report["as_of"]):
        raise ValueError("Filing exceeds cutoff")

    for name in ("current", "previous"):
        period = report.get(name)
        if not period:
            raise ValueError("Two annual periods required")
        start = date.fromisoformat(period["start"])
        end = date.fromisoformat(period["end"])
        if not 330 <= (end - start).days <= 400 or end > filed:
            raise ValueError("Unsupported annual period")

        for field in FIELDS:
            fact = period["facts"][field]
            if fact["tag"] not in TAGS[field]:
                raise ValueError("Unsupported SEC tag: " + field)

            value = Decimal(fact["value_usd"])
            items = (
                raw["facts"]["us-gaap"][fact["tag"]]
                ["units"]["USD"]
            )
            candidates = [
                Decimal(str(item["val"]))
                for item in items
                if item.get("form") == "10-K"
                and item.get("accn") == report["accession"]
                and item.get("filed") == report["filed"]
                and item.get("start") == period["start"]
                and item.get("end") == period["end"]
            ]
            if (
                not value.is_finite()
                or not candidates
                or any(v != value for v in candidates)
            ):
                raise ValueError(
                    "Report differs from raw SEC fact: " + field
                )

    return report, digest(report_bytes)


def calculate(current, previous):
    gap = (
        date.fromisoformat(current["start"])
        - date.fromisoformat(previous["end"])
    ).days
    if not 0 < gap <= 14:
        raise ValueError(
            "Expected consecutive, non-overlapping annual periods"
        )

    values = [
        {
            key: Decimal(period["facts"][key]["value_usd"])
            for key in FIELDS
        }
        for period in (current, previous)
    ]
    if not all(
        value.is_finite()
        for item in values for value in item.values()
    ):
        raise ValueError("Nonfinite amount")

    with localcontext() as context:
        context.prec = 50
        now, before = values
        changes = {}

        for key in FIELDS:
            change = now[key] - before[key]
            growth = (
                change / before[key] * 100
                if before[key] > 0 else None
            )
            changes[key] = dict(
                current_usd=str(now[key]),
                previous_usd=str(before[key]),
                change_usd=str(change),
                change_percent=(
                    str(growth) if growth is not None else None
                ),
                percent_unavailable_reason=(
                    None if before[key] > 0
                    else "Previous amount is zero or negative"
                ),
            )

        margins = {}
        for key, field in (
            ("operating_margin", "operating_income"),
            ("net_margin", "net_income"),
        ):
            a = (
                now[field] / now["revenue"] * 100
                if now["revenue"] > 0 else None
            )
            b = (
                before[field] / before["revenue"] * 100
                if before["revenue"] > 0 else None
            )
            margins[key] = dict(
                current_percent=str(a) if a is not None else None,
                previous_percent=str(b) if b is not None else None,
                change_percentage_points=(
                    str(a - b)
                    if a is not None and b is not None else None
                ),
            )

    return dict(changes=changes, margins=margins)


def display(value, scale=Decimal(1)):
    if value is None:
        return "n/d"
    with localcontext() as context:
        context.prec = 60
        amount = (Decimal(value) / scale).quantize(
            Decimal(".01"), rounding=ROUND_HALF_UP
        )
        text = format(amount, ",.2f")
    return text.replace(",", " ").replace(".", ",")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--company", choices=CIKS, required=True
    )
    parser.add_argument(
        "--unit", choices=["million", "billion"],
        default="million",
    )
    parser.add_argument("--end")
    parser.add_argument("--prior-end")
    args = parser.parse_args()

    root = Path(
        "data/nexus_sec_holdout_2024"
        if args.company in ("amazon", "netflix", "adobe")
        else ".migration_archive/source_data/data"
    )
    folder = root / ("company_" + args.company + "2024")
    report, report_hash = verified(folder, CIKS[args.company])
    current, previous = report["current"], report["previous"]

    if (
        (args.end and args.end != current["end"])
        or (args.prior_end and args.prior_end != previous["end"])
    ):
        parser.error(
            "Available comparison: "
            + previous["end"] + " -> " + current["end"]
        )

    metrics = calculate(current, previous)
    accession = report["accession"]
    url = (
        "https://www.sec.gov/Archives/edgar/data/"
        + str(int(CIKS[args.company])) + "/"
        + accession.replace("-", "") + "/"
        + accession + "-index.html"
    )
    result = dict(
        company=report["entity"],
        cik=CIKS[args.company],
        current=current, previous=previous,
        metrics=metrics, unit=args.unit,
        decimal_precision=50, display_decimals=2,
        method="Deterministic SEC calculations; no neural inference",
        definitions=dict(
            change="current - previous",
            growth="100 * change / previous; only if previous > 0",
            margin="100 * income / revenue; only if revenue > 0",
            margin_change=(
                "current margin - previous margin, "
                "in percentage points"
            ),
        ),
        source=dict(
            filed=report["filed"],
            accession=accession, filing_url=url,
            raw_sha256=report["raw_sha256"],
            report_sha256=report_hash,
            cutoff=report["as_of"],
            report_path=str(folder / "report.json"),
        ),
        script_sha256=digest(Path(__file__).read_bytes()),
    )

    output = Path("runs/nexus_comparisons")
    output.mkdir(parents=True, exist_ok=True)
    filename = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    ) + ".json"
    path = output / filename
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)

    scale = Decimal(
        1000000 if args.unit == "million" else 1000000000
    )
    unit = (
        "millions USD"
        if args.unit == "million" else "milliards USD"
    )
    print(
        "\n" + report["entity"]
        + " | Comparaison annuelle | " + unit
    )
    print("Precedent:", previous["start"], "au", previous["end"])
    print("Courant  :", current["start"], "au", current["end"])

    print("\nChamp | Precedent | Courant | Variation | Variation %")
    for key, label in FIELDS.items():
        item = metrics["changes"][key]
        print(" | ".join([
            label,
            display(item["previous_usd"], scale),
            display(item["current_usd"], scale),
            display(item["change_usd"], scale),
            display(item["change_percent"]),
        ]))

    print("\nMarge | Precedente % | Courante % | Ecart en points")
    for key, label in (
        ("operating_margin", "Operationnelle"),
        ("net_margin", "Nette"),
    ):
        item = metrics["margins"][key]
        print(" | ".join([
            label,
            display(item["previous_percent"]),
            display(item["current_percent"]),
            display(item["change_percentage_points"]),
        ]))

    print("\nVariations % : n/d si le montant precedent est nul ou negatif.")
    print("Exercices fiscaux compares sans ajustement de duree.")
    print("Affichage arrondi a 2 decimales.")
    print("Calculs deterministes depuis les faits SEC; aucune inference neuronale.")
    print("Source:", url)
    print("Rapport:", path)


if __name__ == "__main__":
    main()
