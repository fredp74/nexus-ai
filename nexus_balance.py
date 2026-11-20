"""Explicit SEC balance/cash-flow facts and derived ratios."""
from decimal import Decimal, localcontext
import json
from pathlib import Path

import nexus_compare as compare

FIELDS = {
    "cash": (
        "Tresorerie et equivalents",
        "CashAndCashEquivalentsAtCarryingValue", "instant"
    ),
    "assets": ("Actifs totaux", "Assets", "instant"),
    "liabilities": ("Passifs totaux", "Liabilities", "instant"),
    "equity": (
        "Capitaux propres des actionnaires",
        "StockholdersEquity", "instant"
    ),
    "operating_cash_flow": (
        "Flux de tresorerie operationnel",
        "NetCashProvidedByUsedInOperatingActivities", "annual"
    ),
    "capex": (
        "Achats d'immobilisations corporelles",
        "PaymentsToAcquirePropertyPlantAndEquipment", "annual"
    ),
}
RATIOS = {
    "equity_assets": "Capitaux propres / actifs",
    "cash_assets": "Tresorerie / actifs",
    "liabilities_assets": "Passifs / actifs",
    "capex_operating_cash_flow":
        "Achats d'immobilisations / flux operationnel",
    "fcf_margin":
        "Flux disponible proxy / chiffre d'affaires",
}


def unavailable(reason):
    return dict(status="unavailable", value=None, reason=reason)


def find_fact(payload, report, period, tag, kind):
    records = (
        payload["facts"]["us-gaap"]
        .get(tag, {}).get("units", {}).get("USD", [])
    )
    matches = [
        item for item in records
        if item.get("form") == "10-K"
        and item.get("accn") == report["accession"]
        and item.get("filed") == report["filed"]
        and item.get("end") == period["end"]
        and (
            item.get("start") == period["start"]
            if kind == "annual"
            else item.get("start") is None
        )
    ]
    if not matches:
        result = unavailable(
            "Fait explicite USD absent du depot "
            "et de la periode retenus"
        )
    else:
        values = [Decimal(str(item["val"])) for item in matches]
        if (
            not all(value.is_finite() for value in values)
            or len(set(values)) != 1
        ):
            result = unavailable(
                "Valeurs sources non finies ou contradictoires"
            )
        else:
            result = dict(
                status="found",
                value=str(values[0]),
                reason=None,
            )

    result.update(
        tag=tag, period_kind=kind, end=period["end"],
        start=period["start"] if kind == "annual" else None,
        accession=report["accession"],
        filed=report["filed"],
    )
    return result


def amount(item):
    return (
        Decimal(item["value"])
        if item["value"] is not None else None
    )


def derive(facts, revenue):
    with localcontext() as context:
        context.prec = 50
        cfo = amount(facts["operating_cash_flow"])
        capex = amount(facts["capex"])

        if cfo is None or capex is None:
            fcf = unavailable(
                "Flux operationnel ou achats "
                "d'immobilisations indisponibles"
            )
        elif capex < 0:
            fcf = unavailable(
                "Paiements d'immobilisations negatifs : "
                "aucune correction automatique de signe"
            )
        else:
            fcf = dict(
                status="calculated",
                value=str(cfo - capex),
                reason=None,
            )

        def ratio(numerator, denominator):
            if numerator is None or denominator is None:
                return unavailable(
                    "Composante source indisponible"
                )
            if denominator <= 0:
                return unavailable(
                    "Denominateur nul ou negatif"
                )
            return dict(
                status="calculated",
                value=str(numerator / denominator * 100),
                reason=None,
            )

        assets = amount(facts["assets"])
        ratios = {
            "equity_assets":
                ratio(amount(facts["equity"]), assets),
            "cash_assets":
                ratio(amount(facts["cash"]), assets),
            "liabilities_assets":
                ratio(amount(facts["liabilities"]), assets),
            "capex_operating_cash_flow": ratio(
                capex if capex is None or capex >= 0 else None,
                cfo,
            ),
            "fcf_margin": ratio(amount(fcf), revenue),
        }

    return dict(fcf=fcf, ratios=ratios)


def analyze(report_path, cik, expected_report_hash):
    report_path = Path(report_path)
    report, report_hash = compare.verified(
        report_path.parent, cik
    )
    if report_hash != expected_report_hash:
        raise ValueError("Report changed during generation")

    raw = (
        report_path.parent / "companyfacts.json"
    ).read_bytes()
    if compare.digest(raw) != report["raw_sha256"]:
        raise ValueError("SEC snapshot changed")

    payload = json.loads(raw)
    compare.calculate(report["current"], report["previous"])
    periods = {}

    for name in ("current", "previous"):
        period = report[name]
        facts = {
            key: find_fact(
                payload, report, period, tag, kind
            )
            for key, (_, tag, kind) in FIELDS.items()
        }
        revenue = Decimal(
            period["facts"]["revenue"]["value_usd"]
        )
        periods[name] = dict(
            start=period["start"], end=period["end"],
            facts=facts, derived=derive(facts, revenue),
        )

    return dict(
        periods=periods,
        report_sha256=report_hash,
        raw_sha256=report["raw_sha256"],
        filed=report["filed"],
        accession=report["accession"],
        precision=50,
        method=(
            "Explicit USD facts from the same 10-K accession; "
            "deterministic calculations only"
        ),
        definitions=dict(
            fcf=(
                "Operating cash flow minus "
                "PaymentsToAcquirePropertyPlantAndEquipment; "
                "acquisition payments must be nonnegative"
            ),
            fcf_scope=(
                "Proxy, not an issuer-defined non-GAAP free "
                "cash flow measure; no automatic adjustments "
                "for leases or business acquisitions"
            ),
            equity=(
                "StockholdersEquity only; no substitution "
                "with noncontrolling interests"
            ),
            cash=(
                "CashAndCashEquivalentsAtCarryingValue; "
                "not cash plus marketable securities"
            ),
            ratios=(
                "100 * numerator / denominator; "
                "denominator must be positive"
            ),
        ),
    )


def escape(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def render(data, unit):
    current = data["periods"]["current"]
    previous = data["periods"]["previous"]
    scale = Decimal(
        1000000 if unit == "million" else 1000000000
    )
    label = (
        "millions USD" if unit == "million"
        else "milliards USD"
    )
    lines = [
        "", "## Bilan - situations de fin d'exercice", "",
        "Unite : " + label + ".", "",
        "| Champ | " + previous["end"]
        + " | " + current["end"] + " | Variation |",
        "|---|---:|---:|---:|",
    ]

    def monetary_row(title, before, now):
        a, b = amount(before), amount(now)
        with localcontext() as context:
            context.prec = 50
            change = (
                b - a if a is not None and b is not None
                else None
            )
        return (
            "| " + escape(title) + " | "
            + " | ".join(
                compare.display(value, scale)
                for value in (a, b, change)
            )
            + " |"
        )

    for key, (title, _, kind) in FIELDS.items():
        if kind == "instant":
            lines.append(monetary_row(
                title,
                previous["facts"][key],
                current["facts"][key],
            ))

    lines += [
        "", "## Cash-flow - flux sur les exercices", "",
        "| Champ | Precedent | Courant | Variation |",
        "|---|---:|---:|---:|",
    ]
    for key in ("operating_cash_flow", "capex"):
        lines.append(monetary_row(
            FIELDS[key][0],
            previous["facts"][key],
            current["facts"][key],
        ))
    lines.append(monetary_row(
        "Flux disponible proxy "
        "(flux operationnel - achats d'immobilisations)",
        previous["derived"]["fcf"],
        current["derived"]["fcf"],
    ))

    lines += [
        "", "## Ratios complementaires", "",
        "| Ratio (%) | Precedent | Courant |",
        "|---|---:|---:|",
    ]
    for key, title in RATIOS.items():
        a = previous["derived"]["ratios"][key]["value"]
        b = current["derived"]["ratios"][key]["value"]
        lines.append(
            "| " + escape(title)
            + " | " + compare.display(a)
            + " | " + compare.display(b) + " |"
        )

    lines += [
        "",
        "Ces champs et ratios sont extraits ou calcules "
        "sans inference neuronale. Le flux disponible "
        "est un proxy defini ici, pas une mesure non-GAAP "
        "revendiquee par l'entreprise. Aucun ajustement "
        "automatique pour les contrats de location ou "
        "acquisitions d'entreprises. "
        "La dette totale n'est pas calculee.",
        "",
    ]

    missing = []
    for period in (previous, current):
        for key, item in period["facts"].items():
            if item["status"] != "found":
                missing.append(
                    "- " + FIELDS[key][0]
                    + " au " + period["end"]
                    + " : " + item["reason"] + "."
                )
        if period["derived"]["fcf"]["value"] is None:
            missing.append(
                "- Flux disponible proxy pour "
                + period["end"] + " : "
                + period["derived"]["fcf"]["reason"] + "."
            )

    if missing:
        lines += [
            "**Champs indisponibles (n/d)**", ""
        ] + missing + [""]

    lines += [
        "Ratios n/d si une composante manque ou si le "
        "denominateur est nul ou negatif. Les capitaux "
        "propres concernent les actionnaires ; la tresorerie "
        "ne comprend pas automatiquement les placements financiers.",
        "",
    ]
    return "\n".join(lines)
