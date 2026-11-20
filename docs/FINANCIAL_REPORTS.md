# Financial report exports

## Components

- nexus_metric_comment.py: opt-in V5 metric sentences with per-sentence
  checks and deterministic fallback.
- nexus_report.py: SEC results, V3 summary, annual changes, margins,
  optional V5 commentary, Markdown and JSON exports.
- nexus_balance.py: explicit balance-sheet and cash-flow facts,
  plus defined deterministic ratios.
- nexus_report_cashflow.py: extended report including those facts.

V3 remains the financial-summary adapter.
V5 uses the separately audited last checkpoint in experimental mode.
Its original selection threshold and selection status are unchanged.

## Usage

Fully calculated report, without model loading:

    python .\nexus_report_cashflow.py --company adobe --source-only --show

V3 summary and opt-in experimental V5 commentary:

    python .\nexus_report_cashflow.py --company adobe --experimental-v5 --show

Without --experimental-v5, variation commentary is calculated.
A 0/5 V5 count in source-only mode is expected.

Exports are saved under runs/nexus_reports/ as paired Markdown and JSON.

## Source verification

Facts come from local SEC companyfacts snapshots.
Company identity, source checksum, filing accession, filing date,
currency and annual periods are checked.

Balance-sheet values are instant facts at the period end.
Cash-flow values cover the matching annual period.
Missing or conflicting facts remain unavailable.

## Balance and cash-flow scope

Explicit tags cover:
- Cash and cash equivalents.
- Assets.
- Liabilities.
- Stockholders' equity.
- Operating cash flow.
- Payments to acquire property, plant and equipment.

The free-cash-flow proxy is operating cash flow minus those acquisition
payments. It is not presented as an issuer-defined non-GAAP measure.
Negative acquisition payments are not automatically converted
to positive amounts.

There is no automatic substitution with lease payments, business
acquisitions, marketable securities or noncontrolling interests.
Total debt and valuation are not calculated.

## Controls and provenance

Rejected model text is retained in the JSON diagnostic record.
It is not displayed as accepted prose in the Markdown report.

V5 phrases are checked independently. Zero and undefined metrics
use deterministic sentences without requesting V5 generation.

The JSON records source facts, calculations, available fact tags,
unavailability reasons, model/checkpoint provenance and script hashes.
It also contains the Markdown checksum.

These are controlled task-specific checks, not general semantic
factuality verification or open-ended financial reasoning.

## Observed integration check

Adobe FY2024 completed with:
- An accepted V3 summary.
- Five accepted V5 metric sentences.
- Balance-sheet and cash-flow tables from the same SEC filing.
- Operating cash flow: USD 8,056 million.
- Property, plant and equipment payments: USD 183 million.
- Free-cash-flow proxy: USD 7,873 million.

This integration check is not an independent model benchmark.

## Display

Tables and variation sentences are rounded to two decimals.
The summary preserves supplied amounts.
Changes are calculated before display rounding.
Margin changes are percentage points, not percentage growth.

Reports compare fiscal periods without duration adjustment.
They do not infer causes from the numerical facts.

All model weights, SEC snapshots, generated reports and diagnostic
JSON files remain local and excluded from Git.
