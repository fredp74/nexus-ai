# Financial verbalization V3

## Scope

Mamba-130M with a LoRA adapter verbalizes supplied annual SEC facts:
company, year-end date, revenue, operating income and net income.

This is a bounded verbalization experiment, not general financial
reasoning, forecasting or investment advice.

## Model

Candidate: runs/nexus_finance_v3/best_adapter
Baseline: runs/nexus_finance_v1/best_adapter

V3 starts from V1 and trains for 200 additional updates.
Peak learning rate: 0.00002.

Each update uses two financial examples, one synthetic relation
replay example and one reviewed DART replay example.

Selection checks 16 generated development responses and reference
likelihood. Candidates must meet the generated-response minimum and
retention checks. The initial adapter remains the fallback.

## Results

| Evaluation | V1 | V3 |
|---|---:|---:|
| Reserved synthetic financial examples | 33/64 | 50/64 |
| Previously inspected SEC regression cases | 10/16 | 15/16 |
| New SEC companies: Amazon, Netflix, Adobe | 5/12 | 10/12 |

The regression companies are Microsoft, Apple, Alphabet and Meta.
This previously inspected set is not an independent test.

The new-company set contains three companies, two annual periods
and two units per company. Its 12 cases are related observations.

Manual comparison of all 28 V3 SEC responses with their reference
facts confirmed the reported pass counts.

Remaining errors:
- Meta 2024, millions: operating income and net income swapped.
- Amazon 2024, millions: incorrect attribution and invented amount.
- Netflix 2023, millions: missing net income and repeated USD tokens.

DART replay NLL increased from 0.8390 to 0.8902, about 6.1%.

## Limits

The checker covers company, date, labelled exact amounts, units and
additional numbers. It is not a general semantic factuality verifier.

SEC snapshots retain source hashes and filing references.
The new-company filing-date cutoff is 2025-04-01.
These current snapshots are not independently archived point-in-time data.

Do not train on these test responses and continue describing the same
examples as an independent held-out test.

## Commands

Train or resume: python .\train_nexus_finance_v3.py

Evaluate new companies: python .\test_nexus_sec_holdout.py

Recheck initial cached comparison: python .\rescore_finance.py

## Local artifacts

Models, datasets, optimizer states and result JSON files remain local.
They are excluded from Git.

V1, V3 and the official Mamba backbone are retained.
Retired Qwen/V2 scripts are in the ignored migration archive.
V2 weights and its last optimizer checkpoint have been removed.
