# Nexus AI

**Financial AI research connecting quantitative evidence, business value and macroeconomic risk.**

## The ambition

Build an AI research system that examines investments through three complementary perspectives:

- **Jim Simons — quantitative research:** statistical evidence, probabilities, systematic signals and rigorous out-of-sample validation.
- **Warren Buffett — fundamental investing:** business quality, durable competitive advantages, cash generation, intrinsic value and long-term capital allocation.
- **Ray Dalio — macroeconomics and risk:** economic cycles, debt, inflation, liquidity, diversification and portfolio exposures.

The goal is to connect these perspectives: understand the business, examine the data, and place both in their economic and risk context.

## What Nexus AI should deliver

Given financial documents, market data and an analysis date, the intended system should help answer:

- What does the quantitative evidence support, and how robust is it?
- What drives the company's earnings, cash flows and valuation?
- Which economic regimes and portfolio exposures matter?
- Where do the quantitative, fundamental and macroeconomic views agree or conflict?
- Which assumptions, missing information or risks could change the conclusion?

Sources, dates, calculations and uncertainty should remain visible. Research must use information available at the stated analysis date.

## Research directions

### Quantitative research

Explore systematic signals, market behaviour and statistical relationships. Evaluate results with temporal separation, transaction costs, turnover and checks against look-ahead bias and overfitting.

### Fundamental analysis

Read financial disclosures, connect accounting figures to business economics, and examine profitability, balance-sheet strength, capital allocation and valuation assumptions.

### Macroeconomic and portfolio analysis

Study growth, inflation, rates, liquidity and debt cycles. Connect scenarios to asset exposures, diversification and portfolio risk.

### Integrated financial reasoning

Combine evidence across these areas, explain disagreements and produce traceable research conclusions.

## Current implementation

The active prototype uses the official pretrained **Mamba-130M** model with **LoRA adapters** on its input and output projections.

Current work focuses on:

- CPU-compatible training and generation.
- Resumable adapter training with frozen base weights.
- Grounded data-to-text generation using DART and controlled examples.
- Source-aligned references, replay and evaluation on reserved examples.
- Inspection of generated text alongside language-model loss.

The base model is `state-spaces/mamba-130m-hf`, pinned to revision
`1e76775f628fbf1350fbe4dbb3d971ba64af25a1`.

Factual reliability remains incomplete: relation inversions, omitted facts and unsupported additions still occur. Financial specialization and the integrated investment-research workflow remain development goals. Investment performance has not been established.

## Development priorities

1. Improve factual consistency and use of context across varied documents.
2. Build financial corpora with source provenance, reporting dates and temporal evaluation splits.
3. Support verifiable financial calculations and fundamental analysis.
4. Evaluate quantitative signals, macroeconomic scenarios and portfolio risk.
5. Integrate these capabilities into a coherent financial research workflow.

## Working with the repository

Install dependencies with `python -m pip install -r requirements.txt`.

For the existing local experiment, the training entry point is
`python train_mamba_reviewed_dart.py`.

Resuming requires the local model cache, DART data and parent adapters. Weights, datasets, checkpoints and migration archives are excluded from Git.

## Research foundations

- [Official Mamba implementation](https://github.com/state-spaces/mamba)
- [Official Mamba-130M model](https://huggingface.co/state-spaces/mamba-130m-hf)
- [mamba.py](https://github.com/alxndrTL/mamba.py)
- [SSM-PEFT](https://github.com/furiosa-ai/ssm-peft)
- [DART](https://github.com/Yale-LILY/dart)

Upstream code, model weights and datasets remain subject to their respective licenses.

---

A research project by [Algotradingresearch.com](https://algotradingresearch.com).
