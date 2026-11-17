# Claim-level temporal citation verification — ATR Lab02

M.Tech AI Tools for Research, IIIT Allahabad (November 2026).

This repository is a **public-safe code and analysis extract** for an exploratory study comparing a strict historical-cutoff prompt, a weaker prompt, a deterministic citation-date filter, and a Qwen2.5 3B verifier. The historical cutoff is **2025-06-30 23:59:59 UTC**.

## Main reproducible result

The seven selected E2v4 local-model question runs generated 51 claims under strict instructions (16 citing post-cutoff evidence) and 39 under weak instructions (20 citing post-cutoff evidence). The deterministic date rule removed all 20 weak-arm date violations by construction. The Qwen verifier removed 4/16 strict and 7/20 weak date-violating claims but incorrectly marked 18 strict and 2 weak pre-only claims as post-cutoff. These are **citation-date judgments**, not independently adjudicated factual-support accuracy or gold-fact coverage.

`results/E2v4_citation_date_decisions.csv` provides per-claim citation IDs, date flags, and Qwen decisions without reproducing third-party passage text. Run:

```bash
python3 scripts/recompute_public_dates.py
python3 -m unittest discover -s scripts -p 'test_public_dates.py' -v
```

The first command recomputes five-method aggregated structural counts from the public audit CSV; the second checks expected totals. No model access is required for these summary-level checks.

## Full model replication

The original input passages, raw request/response bodies, and source-bearing run folders are **not released** here pending rights review of third-party excerpts. Experiment-generation scripts from the private package are provided for scrutiny under `scripts/`, but they **cannot run with this public extract alone**. To execute the original model comparison, researchers must obtain authorized copies of evidence passages, construct the same saved question contexts/source-ID registry, install Ollama, and set up the runner with the original directory structure. The configured local model was `atr-qwen3b` (Qwen2.5 3B), with temperature 0 and requested seed 42; identical responses are not guaranteed across builds.

Experiment 1 was a distinct 11-question ChatGPT-interface study. Four verifier outputs were archived as independent original JSONs, while seven decisions had weaker provenance. Original 85 claim-support judgments were provisional and nonblind. Do **not** treat this public data as ground-truth factual citation precision.

An additional E2v4 H02 completion exists; the chosen seven-run manifest and earlier-H02 sensitivity analysis are included. Run selection was based on the first *researcher-reported* completion and was not preregistered. See the manuscript for the selection limitation.

## Files

- `paper/`: revised 10-page manuscript (LaTeX, BibTeX, PDF).
- `results/`: selected-run manifest, claim-level citation-date audit, five-arm case results, H02 sensitivity and source-hash manifest.
- `scripts/recompute_public_dates.py`: complete public-data recomputation.
- `scripts/run_experiment.py`: original experiment runner (**requires private evidence inputs**).
- `configs/`: methodological settings and the E2v4 protocol extension.

No original full-text scientific PDFs, raw source-bearing prompts, reviewer's personal information, API keys, or confidential notes are intentionally included. Review and confirm repository contents before publishing.
