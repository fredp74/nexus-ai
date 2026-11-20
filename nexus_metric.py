"""One source metric per sentence. Names/dates stay outside neural text."""
from decimal import Decimal
import random
import re
from pathlib import Path

import nexus_compare as compare

FIELDS = {
    "revenue_growth": ("Revenue", "percent"),
    "operating_income_growth": ("Operating income", "percent"),
    "net_income_growth": ("Net income", "percent"),
    "operating_margin_change": (
        "Operating margin", "percentage points"
    ),
    "net_margin_change": ("Net margin", "percentage points"),
}


def sentence(metric, value):
    subject, unit = FIELDS[metric]
    if value is None:
        noun = (
            " growth" if metric.endswith("_growth")
            else " change"
        )
        return subject + noun + " is not defined."

    number = Decimal(value)
    if number == 0:
        return subject + " was unchanged."

    verb = (
        " increased by " if number > 0
        else " decreased by "
    )
    return (
        subject + verb + format(abs(number), ".2f")
        + " " + unit + "."
    )


def sample(tokenizer, pack, index, metric, value, kind):
    prompt = (
        "Facts:\nMETRIC | " + metric.upper()
        + "\nVALUE | "
        + (value if value is not None else "NOT_DEFINED")
        + "\nUNIT | " + FIELDS[metric][1]
        + "\nText:"
    )
    item = pack(
        tokenizer, index, prompt,
        sentence(metric, value), kind,
    )
    item.update(metric=metric, value=value)
    return item


def synthetic_data(tokenizer, pack):
    result = {
        split: [] for split in ("train", "dev", "test")
    }
    seen = set()

    for split, count, seed in (
        ("train", 128, 81601),
        ("dev", 16, 81602),
        ("test", 16, 81603),
    ):
        rng = random.Random(seed)
        for i in range(count):
            for j, metric in enumerate(FIELDS):
                case = (i + j) % 10

                if case == 0:
                    value = None
                elif case == 1:
                    value = "0.00"
                else:
                    while True:
                        negative = case % 2 == 0
                        maximum = (
                            9999
                            if metric == "revenue_growth" and negative
                            else 120000
                        )
                        number = Decimal(
                            rng.randrange(1, maximum + 1)
                        ) / 100
                        if negative:
                            number = -number
                        value = format(number, ".2f")
                        if (metric, value) not in seen:
                            seen.add((metric, value))
                            break

                item = sample(
                    tokenizer, pack,
                    f"metric:{split}:{i}:{metric}",
                    metric, value, "synthetic-comparison",
                )
                item["source"] = dict(
                    kind="synthetic-precomputed-metric",
                    split=split,
                )
                result[split].append(item)

    sets = {
        split: {
            (row["metric"], row["value"])
            for row in rows
            if row["value"] not in (None, "0.00")
        }
        for split, rows in result.items()
    }
    assert not (
        sets["train"] & sets["dev"]
        or sets["train"] & sets["test"]
        or sets["dev"] & sets["test"]
    )
    return result


def conforms(generated, reference):
    def clean(text):
        return re.sub(r"\s+", " ", text).strip().casefold()
    return clean(generated) == clean(reference)


def real_tests(tokenizer, pack, old):
    rows = []
    for symbol, cik in compare.CIKS.items():
        root = Path(
            "data/nexus_sec_holdout_2024"
            if symbol in ("amazon", "netflix", "adobe")
            else ".migration_archive/source_data/data"
        )
        report, report_hash = compare.verified(
            root / ("company_" + symbol + "2024"), cik
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

        for metric, raw in values.items():
            value = (
                compare.display(raw)
                .replace(" ", "").replace(",", ".")
                if raw is not None else None
            )
            item = sample(
                tokenizer, pack, symbol + ":" + metric,
                metric, value, "sec-comparison-regression",
            )
            item["source"] = dict(
                company=report["entity"],
                end=report["current"]["end"],
                prior_end=report["previous"]["end"],
                report_sha256=report_hash,
                raw_sha256=report["raw_sha256"],
                filing_url=report["filing_url"],
                unrounded_value=raw,
            )
            rows.append(item)

        for period in (report["current"], report["previous"]):
            values_usd = {
                key: period["facts"][key]["value_usd"]
                for key in old.FIELDS
            }
            for unit in ("million", "billion"):
                item = old.row(
                    tokenizer, report["entity"], period["end"],
                    values_usd, unit, 0, len(rows),
                    dict(
                        kind="sec-summary-regression",
                        raw_sha256=report["raw_sha256"],
                        filing_url=report["filing_url"],
                    ),
                )
                item["kind"] = "sec-summary-regression"
                rows.append(item)
    return rows
