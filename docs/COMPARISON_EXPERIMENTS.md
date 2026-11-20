# Annual comparisons and metric verbalization

## Working components

- nexus_compare.py calculates annual changes and margins from local SEC facts.
- nexus_comment.py attempts a V3 comparison comment, with an explicitly
  identified deterministic fallback when its controlled template is not met.
- V3 remains the adapter used by nexus_finance.py.
- No V4 or V5 adapter has been promoted into the application.

Calculations use Decimal. Percent growth is undefined when the previous
amount is zero or negative. Margin changes are expressed in percentage
points, not percent. Display values are rounded to two decimals.

## V4: full comparison paragraphs

V4 starts from V3 and trains for 200 updates with synthetic comparisons,
financial-summary replay and reviewed DART replay.

At step 200, development scores were:
- Comparisons: 11/16.
- Financial summaries: 16/16.
- Retention checks: passed.

The comparison selection minimum was 12/16. No trained candidate
qualified, so best_adapter remained the initial V3 copy.

A separate diagnostic evaluated the trained last.pt checkpoint:
- Reserved synthetic comparisons: 19/32.
- SEC comparison paragraphs: 3/7.
- SEC summaries: 28/28, versus 25/28 for the matching V3 baseline.

Errors included mixed metrics, modified dates, an incorrect company
and unsupported causal statements.

## V5: one metric per sentence

V5 starts independently from V3.
Company names and dates are outside the generated sentence.
Inputs contain one precomputed metric, its value and its unit.

The task covers revenue growth, operating-income growth, net-income
growth, operating-margin change and net-margin change.

Synthetic data:
- Training: 640 sentences.
- Validation: 80 sentences.
- Test: 80 sentences.

Nonzero metric/value pairs are disjoint across splits.
Zero and undefined states are intentionally shared.
The model verbalizes supplied metrics; it does not calculate them.

At step 200, development scores were:
- Metric sentences: 17/20.
- Financial summaries: 15/16.
- Retention checks: passed.

The metric selection minimum was 18/20.
No trained candidate qualified. best_adapter therefore remains V3.
The trained V5 weights are retained in last.pt.

A separate diagnostic of last.pt produced:
- Reserved synthetic metric sentences: 69/80.
- SEC metric sentences: 35/35.
- SEC summaries: 22/28, equal to the matching V3 baseline.

All 35 SEC metric sentences were manually compared with their
references; their values, directions and units matched.

## Interpretation and limits

The 35 SEC sentences represent five metrics for seven companies,
not 35 independent companies. These SEC cases were already inspected
during development and are regression checks.

V4 and V5 use different summary prompt ordering. Their summary counts
must be compared with their own matching V3 baselines, not directly
with one another.

Evaluation uses controlled reference forms. Exact-match failures can
include correct paraphrases. For example, Revenue growth was 0.00
percent is factually consistent with Revenue was unchanged.
Other failures are substantive, including an incorrect digit or a
decrease stated for a zero change.

The original scores and selection thresholds remain unchanged.
Diagnostic test results do not retroactively promote either adapter.

## Next integration step

Keep V3 for financial summaries.
Consider V5 only as an experimental metric-verbalization adapter,
with exact per-sentence checks and a deterministic fallback.

This experimental integration has not yet been implemented.
No new training or model promotion is performed by this publication.

## Local artifacts

Scripts are versioned. Models, datasets, optimizer states and cached
responses remain excluded from Git under runs/ and data/.

Key artifacts:
- runs/nexus_finance_v4/last.pt
- runs/nexus_finance_v4/audit_last_step200.json
- runs/nexus_finance_v5/last.pt
- runs/nexus_finance_v5/audit_last_step200.json
- runs/nexus_comparisons/
- runs/nexus_comments/

Do not modify training scripts or their hashed dependencies while
resuming an existing experiment.
