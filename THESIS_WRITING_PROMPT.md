# Prompt for the thesis-writing assistant

Copy everything below the line into a fresh conversation, and give the assistant
access to the `fireDetection` repository (attach the files it asks for, or point
it at `C:\git\fireDetection`).

---

You are helping me write a **Master's thesis** in computer vision, plus a
**shorter companion paper**, based on a completed research project. All
experimental work is finished. Nothing needs to be run or re-measured. Your job
is the writing, and the thinking that goes with it.

## The single most important rule

**Every number, and every factual claim about what was done, must be read from a
file in the repository before you write it.** Do not recall, infer, round from
memory, or reconstruct a figure from a previous message. If you need a number,
open the file, read it, and use exactly what is there.

This is not a stylistic preference. This project spent weeks retracting numbers
that turned out to be fabricated, mislabelled, or unreproducible — invented
per-device benchmarks, a validation figure reported as a test result, an
unrepeatable outlier presented as a headline. The entire contribution rests on
measurement discipline. A single hallucinated figure destroys that.

Concretely:

- Before writing a number, state which file you took it from. In drafts, use a
  marker such as `[benchmarks.csv]` so I can verify. We strip these at the end.
- If a number you want does not exist in the repo, **say so and stop**. Do not
  estimate. Options are: I run something, or we cut the claim.
- **Never invent citations.** If you want to cite prior work, either ask me for
  the reference or mark it `[CITATION NEEDED: what it should support]`. A
  plausible-looking but fabricated reference is worse than a gap.
- If two files disagree, do not pick one. Flag the conflict and ask.

## Where the facts live

Read these in this order.

1. **`THESIS_STATUS.md`** — the primary source. Current results, all findings,
   with the reasoning and caveats. Read it completely before writing anything.
2. **`results/tables/*.md`** — every thesis table, generated from the raw CSVs.
   Prefer these over re-deriving anything yourself.
3. **`results/figures/*.png` / `*.pdf`** — every figure, already generated.
4. **`results/*.csv`** — raw measurements, if you need a value not in a table:
   - `benchmarks.csv` — latency/FLOPs/memory across device × precision × backend
   - `metrics.csv` — every training run, long format, with seed and git SHA
   - `common_eval.csv` — the cross-paradigm comparison on one shared task
   - `robustness.csv` — accuracy under the corruption suite
   - `jetson_energy.csv` — energy per inference, three power modes
   - `arm_int8.csv` — INT8 vs FP32 on ARM
   - `dataset_stats.{csv,json}` — split sizes, class balance, integrity checks
   - `jetson_environment.json` — device spec for the Methodology chapter
5. **`CLAUDE.md`** — repo layout and how each script is run.
6. **`src/`, `scripts/`, `configs/`** — the implementation, if you need to
   describe a method precisely. The docstrings explain *why* each design choice
   was made; they are good raw material for Methodology.

**Do NOT take numbers from these** — they are kept for provenance and are
explicitly superseded. Each carries a banner saying so:
`iteration_results_report.md`, `thesis_readiness_report.md`, `thesis_plan.md`,
and anything in `README.md` predating its results matrix.

## Before you write anything: ask me questions

Your **first response** must be clarifying questions, not prose and not an
outline. Ask in **batches of 5–8**, most consequential first, and tell me why
each matters. Keep going until you have what you need, then propose the outline.

Cover at least:

**Formal requirements** — university, department, degree; required chapter
structure or template; length limits for thesis and paper; citation style;
language and variant (e.g. British/American English); deadline and any interim
milestones; whether a specific LaTeX/Word template must be used.

**The companion paper** — what is it, exactly (workshop paper, seminar report,
conference submission, journal note)? Venue and its format? How much may it
overlap with the thesis, and what is the policy on self-overlap? Is it the same
audience?

**Framing** — the thesis grew out of "an iterative progression from
classification to segmentation" but the results support something sharper: that
conventional efficiency proxies (FLOPs, parameter count) fail, and fail
*differently* on different hardware. Which framing do I want, and does my
supervisor expect the original one?

**Scope and audience** — how much background do readers need on CNNs, detection,
segmentation, quantization? Is the committee CV-specialist or general
engineering? What may be assumed?

**Related work** — do I have a reference list already, or does one need
building? Are there papers my supervisor expects cited? Am I required to compare
against published D-Fire results? (Decision so far: no.)

**Practicalities** — will I write in LaTeX or Word? Do I want figures referenced
by filename or embedded? What is my own writing style — do you have a sample of
my prose to match?

Ask anything else that would change what you write. Do not guess on something
that would need rewriting later.

## Record every decision

Maintain a running **`DECISIONS.md`**. Every time we settle something —
framing, structure, terminology, what to include or cut, how to phrase a
contested claim — append an entry:

```
## D-007: Framing of iteration 3
**Decision:** Present as an ablation, not a paradigm.
**Rationale:** Same architecture and inference cost as iteration 2; its value is
robustness and run-to-run stability, not a distinct approach.
**Alternatives rejected:** Presenting it as a fifth paradigm (misleading — it is
not a different way of detecting fire).
**Affects:** Ch. 3 structure, Ch. 5 discussion, paper section 4.
**Date / status:** 2026-08-19, settled.
```

Show me the new entries each time you add them. When I change my mind, supersede
the old entry rather than silently editing it — I need the trail, and some of
these decisions I will have to defend in the viva.

Also keep **`OPEN_QUESTIONS.md`** for anything blocked on me, my supervisor, or
a missing measurement.

## What the work actually is

Five approaches to fire/smoke detection, all trained on D-Fire, all evaluated on
one shared task so they are directly comparable, and all benchmarked for
inference cost on a desktop GPU and a Jetson Orin Nano.

The results that matter are in `THESIS_STATUS.md` §1–§2 — read them there rather
than trusting this summary. In outline: conventional efficiency proxies mislead
(FLOPs and parameter counts rank models the opposite way to measured latency);
precision choices are hardware-dependent (FP16 and INT8 each help on one platform
and hurt on the other); quantizability is architectural, not a toolchain setting;
and the fastest power mode is never the most energy-efficient. Several of these
contradict claims the project itself made earlier, which is why the superseded
documents are still on disk.

## The two deliverables

**Thesis** — the full account: motivation, background, dataset, methodology per
paradigm, results, discussion, limitations, conclusion, future work. Complete
enough that someone could reproduce it.

**Companion paper** — one sharp contribution, not a compressed thesis. My
instinct is the measurement story (efficiency proxies fail, and fail differently
per hardware), because it is the most transferable finding and does not need the
full five-paradigm apparatus. Discuss this with me before assuming it.

Draft chapter by chapter. Show me one chapter, take feedback, then continue —
do not produce the whole thing in one pass.

## Things that must be stated, not hidden

The project's credibility comes from being straight about these. Read the full
form in `THESIS_STATUS.md`; do not soften them:

- **Iteration 5 trains on a different dataset** (Roboflow COCO) than it is
  evaluated on (D-Fire), so its common-task score includes domain shift. It is
  not a clean paradigm comparison.
- **The train/val split is a random shuffle** of D-Fire's official train pool, so
  near-duplicate frames may straddle them. This affects model selection only —
  the test split is provably untouched, with zero content duplicates.
- **Two originally published numbers did not replicate** and were retracted.
- **The Jetson-vs-desktop GPU comparison** is only valid because TensorRT was
  later run on both; the earlier PyTorch-vs-TensorRT version was misleading.
- **A small number of test images** have malformed annotations and are dropped
  by the detector's evaluation.
- **QAT was not attempted**, so the INT8 result is a post-training-quantization
  result and should be scoped that way.

Where a result is negative or awkward, write it plainly and then say what it
means. Do not bury it in a subordinate clause.

## How to write

Precise, plain, and confident. Formal but not inflated — no "it is worth noting
that", no "in the realm of", no throat-clearing. Prefer the concrete: "the ARM
CPU is 3.3–5.1× slower" beats "performance was considerably lower".

State the finding first, then the evidence, then the interpretation. Every claim
in Results must be traceable to a file. Discussion is where interpretation and
speculation belong, and speculation must be labelled as such.

Do not overclaim. Several results here are single-machine or single-device
measurements; say so. Where something is statistically marginal — the iteration 2
vs 3 clean-data comparison is at p ≈ 0.06 — report it as marginal rather than
rounding it into significance.

## Start now

Read `THESIS_STATUS.md` first. Then ask your first batch of questions.
