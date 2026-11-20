"""Extend the Nexus report with explicit SEC balance/cash-flow facts."""
from pathlib import Path

import nexus_balance as balance
import nexus_report as report


def main():
    original = report.render

    def extended(record, json_name):
        data = balance.analyze(
            Path(record["source"]["report_path"]),
            str(record["report"]["cik"]).zfill(10),
            record["source"]["report_sha256"],
        )
        record["balance_cashflow"] = data

        for path in (__file__, balance.__file__):
            record["scripts"][Path(path).name] = (
                report.digest(Path(path))
            )

        text = original(record, json_name)
        marker = "\n## Source et perimetre\n"
        old_scope = (
            "Perimetre : chiffre d'affaires, resultats et marges. "
            "Cette fiche ne couvre pas encore le bilan, "
            "le cash-flow ou la valorisation et ne deduit aucune cause."
        )
        if (
            text.count(marker) != 1
            or text.count(old_scope) != 1
        ):
            raise ValueError("Base report structure changed")

        text = text.replace(
            marker,
            "\n" + balance.render(data, record["unit"]) + marker,
            1,
        )
        return text.replace(
            old_scope,
            "Perimetre : chiffre d'affaires, resultats, "
            "marges, bilan et cash-flow selon les faits "
            "explicites disponibles. Pas de valorisation "
            "ni de deduction de causes.",
            1,
        )

    report.render = extended
    try:
        report.main()
    finally:
        report.render = original


if __name__ == "__main__":
    main()
