"""Rescore saved financial summaries. No model loading or training."""
import hashlib
import json
import re
from datetime import date
from decimal import Decimal
from pathlib import Path

NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
MONEY = re.compile(
    r"\bUSD\s*(" + NUMBER + r")\s*(million|billion)s?\b", re.I
)
LABEL = re.compile(
    r"\brevenues?\b|\boperating income\b|\bnet income\b", re.I
)


def normalize(text):
    months = (
        "January February March April May June July August "
        "September October November December"
    ).split()
    names = "|".join(months)

    def convert(match, reverse=False):
        month, day = (
            (match[2], match[1]) if reverse
            else (match[1], match[2])
        )
        try:
            return date(
                int(match[3]),
                months.index(month.title()) + 1,
                int(day),
            ).isoformat()
        except ValueError:
            return match[0]

    text = re.sub(
        r"\b(" + names + r")\s+(\d{1,2}),?\s+(\d{4})\b",
        convert, text, flags=re.I,
    )
    text = re.sub(
        r"\b(\d{1,2})\s+(" + names + r"),?\s+(\d{4})\b",
        lambda match: convert(match, True),
        text, flags=re.I,
    )
    text = re.sub(
        r"([+-])\s*(?:US\$|\$)\s*(?=\d)",
        r"USD \1", text,
    )
    return re.sub(
        r"(?:US\$|\$)\s*(?=[+-]?\d)", "USD ", text
    )


def name(text):
    aliases = {
        "corporation": "corp",
        "incorporated": "inc",
        "limited": "ltd",
    }
    return [
        aliases.get(word, word)
        for word in re.findall(r"[a-z0-9]+", text.lower())
    ]



def fields(text):
    amounts = list(MONEY.finditer(text))
    result, used = {}, set()
    labels = {
        "revenue": r"\brevenues?\b",
        "operating_income": r"\boperating income\b",
        "net_income": r"\bnet income\b",
    }
    connector = (
        r"\s*(?:at\s+[A-Za-z][A-Za-z .,'&-]*?\s+)?"
        r"(?:(?:of|was|is|were|at|reached|totaled|"
        r"totalled|stood|reported)\s*)*[:=]?\s*"
    )

    for index, amount in enumerate(amounts):
        left = amounts[index - 1].end() if index else 0
        right = (
            amounts[index + 1].start()
            if index + 1 < len(amounts) else len(text)
        )
        before = text[left:amount.start()]
        after = text[amount.end():right]
        candidates = []

        for key, label in labels.items():
            forward = re.search(
                label + connector + r"$", before, re.I
            )
            backward = re.match(
                r"\s+in\s+" + label, after, re.I
            )
            if forward or backward:
                candidates.append(key)

        if len(candidates) == 1:
            value = (
                Decimal(amount[1].replace(",", "")),
                amount[2].lower(),
            )
            result.setdefault(candidates[0], []).append(value)
            used.add(index)

    return result, amounts, used


def check(reference, generated):
    reference = normalize(reference)
    text = normalize(generated)
    match = re.fullmatch(
        r"For the year ended (\d{4}-\d{2}-\d{2}), "
        r"(.+?) reported revenue .*",
        reference,
    )
    if not match:
        raise ValueError(
            "Unexpected reference format: " + reference
        )

    end, company = match.groups()
    expected, _, _ = fields(reference)
    required = {"revenue", "operating_income", "net_income"}
    if set(expected) != required:
        raise ValueError("Reference fields could not be parsed")

    errors = []
    tokens, entity = name(text), name(company)
    if not any(
        tokens[index:index + len(entity)] == entity
        for index in range(len(tokens))
    ):
        errors.append("company")

    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
    if dates != [end]:
        errors.append("date")

    if re.search(
        r"\b(?:quarter\w*|three months|six months|nine months)\b",
        text, re.I,
    ):
        errors.append("annual_period")

    binding_text = re.sub(
        r"for (?:the |fiscal )?year end(?:ed|ing) "
        + re.escape(end) + r",?",
        "", text, flags=re.I,
    )
    actual, amounts, used = fields(binding_text)

    for key, value in expected.items():
        if actual.get(key) != value:
            errors.append(key)

    if len(amounts) != 3 or len(used) != 3:
        errors.append("amount_count_or_binding")

    remaining = MONEY.sub("", text)
    remaining = remaining.replace(end, "").replace(company, "")
    if re.search(r"\d", remaining):
        errors.append("extra_number")

    return {"passed": not errors, "errors": errors}


def main():
    source = Path("runs/nexus_qwen05b_comparison/result.json")
    raw = source.read_bytes()
    saved = json.loads(raw)
    baseline = saved["rows"]
    if not baseline:
        raise ValueError("No saved comparison outputs")

    models = {"Qwen saved outputs": baseline}
    hashes = {
        str(source): hashlib.sha256(raw).hexdigest()
    }

    for version in ("v1", "v2"):
        path = Path(
            "runs/nexus_finance_" + version + "/result.json"
        )
        if not path.exists():
            print("Missing:", path)
            continue

        data = path.read_bytes()
        rows = json.loads(data)["results"]["after"]["real_generations"]
        indexed = {row["index"]: row for row in rows}
        selected = [
            indexed[row["index"]] for row in baseline
        ]

        if any(
            first["facts"] != second["facts"]
            or first["reference"] != second["reference"]
            for first, second in zip(baseline, selected)
        ):
            raise ValueError(
                "Cached inputs differ: " + str(path)
            )

        models["Mamba " + version.upper()] = selected
        hashes[str(path)] = hashlib.sha256(data).hexdigest()

    results = {}
    for model, rows in models.items():
        evaluated = [
            {
                "index": row["index"],
                "generated": row["generated"],
                "reference": row["reference"],
                **check(row["reference"], row["generated"]),
            }
            for row in rows
        ]
        results[model] = evaluated
        passed = sum(row["passed"] for row in evaluated)
        print("\n" + model, str(passed) + "/" + str(len(evaluated)))

        for row in evaluated:
            if not row["passed"]:
                print(
                    "  Case", row["index"], ":",
                    ", ".join(row["errors"]),
                )

    output = Path("runs/finance_rescore_v1.json")
    report = {
        "input_sha256": hashes,
        "results": results,
        "scope": (
            "Conservative labelled amount/date checks; "
            "not full semantic verification. "
            "Previously inspected regression set."
        ),
    }
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\nSaved:", output)


if __name__ == "__main__":
    main()
