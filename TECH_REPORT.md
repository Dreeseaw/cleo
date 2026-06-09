# Cleo: a 2B tool-using SQL analyst that discovers values before answering

*Technical report — June 2026. A personal research project.*

## TL;DR

Cleo is a ~2B model that answers natural-language questions over a SQL database. Unlike one-shot
text-to-SQL, it can issue read-only `gather` probes to **discover real values, codes, and conventions in
the data** before writing its answer — e.g. learning that *"current"* means `to_date = '9999-01-01'`, or
that an order status is `'O'` not `'open'`. These literals are exactly what a one-shot model has to
*guess*, and usually guesses wrong.

The shipped model (v1.0) was produced by **behavioral cloning on denotation-verified teacher
trajectories** — not RL, not full-logit distillation — for about **$1.30 of teacher inference**. It
beats its one-shot predecessor on the two axes that matter for using it on a *new* database:

| held-out suite | v0.9 (one-shot) | **v1.0 (tool-use)** |
|---|---|---|
| value-discovery (answer needs a discovered literal) | 13.6% | **51.5%** |
| general SQL, out-of-distribution DBs | 59.3% | **64.2%** |
| general SQL, in-distribution | 57.5% | 47.5% |

The first half of this report is the production training process that worked. The second half is an
honest account of the approaches that **didn't** — RL, clean-base starts, on-policy DAgger — and what
each failure taught us about the limits of a 2B model.

---

## Part I — The production training process

### 1. The problem: the value-discovery gate

A text-to-SQL model maps `(schema, question) → SQL`. It fails predictably when the *correct query
depends on a literal that lives in the data, not the schema*: a sentinel date for "current", a
single-letter status code, an enum's exact casing, a country code (`GB` not `UK`). The schema shows a
column named `status`; only the data reveals its values are `{'O','C','X'}`. A one-shot model must guess,
and a wrong guess returns the wrong rows (often silently — an empty result or a plausible-but-wrong
count). We call this the **value-discovery gate**, and at 2B it is the dominant error mode on realistic
databases.

The fix is structural: give the model a **tool** to look. Cleo runs a short loop —

```
{"tool":"gather","sql":"SELECT DISTINCT status FROM orders"}   → cols=[status] rows(3)=[["O"],["C"],["X"]]
{"tool":"final","sql":"SELECT region, COUNT(*) FROM orders WHERE status='C' GROUP BY region"}
```

— issuing up to a few read-only probes, then committing to a final read-only `SELECT` (or a
clarification when the request is ambiguous).

### 2. Starting point

v0.9 is a 2B `qwen3_5` model, SFT'd on ~55k `(schema, question → SQL/clarify)` examples (15k authored +
a 40k SynSQL sample). That SFT bought broad SQL generality but contained **zero gather-use signal** —
every training target handed the model the literal directly. So v0.9 is a competent one-shot analyst
that cannot discover values. The job for v1.0: teach the gather-and-use behavior **without** losing the
55k generality, and ship at minimal cost.

### 3. The method that shipped: teacher-trajectory behavioral cloning

The recipe is deliberately the cheapest thing that could work:

1. **Curate questions, not answers.** A leak-guarded pool of ~700 questions across 472 schemas
   (value-discovery depth + general breadth + clarify), drawn from execution-verified pools and held
   strictly disjoint from every eval suite (schema- and question-level).
2. **Let a teacher drive the harness.** A frontier-but-cheap teacher (`gemini-2.5-flash`, ~$0.30/Mtok)
   runs the gather→final loop against the real database for each question. We keep a trajectory **only if
   its final answer is denotation-correct** against gold (verified by executing both queries and
   comparing rows). Every kept trajectory contributes `(prompt_state → teacher_action)` pairs.
3. **Behavioral-clone the student on those pairs.** Standard supervised fine-tuning (LoRA, r=16), cross-
   entropy on the *action* tokens only (the schema/observation context is masked). No teacher logits are
   ever stored — just the chosen action text. The loss is computed with the LM head applied **only at
   label positions** (the vocabulary is 248k; a full-sequence head is the memory bottleneck).

Total teacher spend to build the trajectory set: **~$1.30**. Training is ~3.5 minutes on a single 16 GB
consumer GPU.

Why BC and not RL or logit-distillation? Because BC is *direct supervision of the exact behavior we
want*. RL has to discover the behavior by sampling (it didn't — see Part II), and full-logit
distillation is storage-prohibitive and was a marginal win in prior work. BC writes the gather-and-use
behavior straight into the model.

### 4. Calibration: teaching when *not* to gather

The first BC model gathered well but **over-gathered** — it probed on simple questions where the answer
was obvious from the schema, and the unnecessary probe sometimes derailed the final answer. Diagnostically
this was clear: when the model *didn't* gather, its general accuracy matched v0.9 (competence intact); the
loss was entirely unnecessary gathers.

The fix stayed in the same cheap regime: add a slice of **"confident → answer directly"** trajectories —
easy, short-schema questions where the teacher (prompted to prefer direct answers) one-shots correctly —
and upweight them. This re-balanced the gather decision: out-of-distribution gather rate fell from 36/81
to 11/81, and OOD accuracy rose from 48% to **63%**, past v0.9, while value-discovery held.

### 5. Packaging

The bf16 model is merged and quantized to **GGUF Q8_0**. Q8 is deliberate: lower quantization (Q4_K_M)
measurably erodes the trained deltas (a tool-use/RL gain of ~+44 points on one suite dropped by ~17
points under Q4), whereas Q8 reproduces the bf16 accuracy exactly (value-discovery 51.5% at both
precisions). The model ships as a small `cleo` Python package that runs the gather→final loop against
**any DB-API 2.0 connection** (Postgres, SQLite, DuckDB, SQLAlchemy) — no data pre-staging, every query
validated read-only and rolled back.

### 6. Results

On held-out suites (denotation-scored: execute predicted vs gold SQL, compare row-sets; suites are
schema-disjoint from all training data, gold verified by an LLM-judge swarm):

- **Value-discovery** (66 questions, 22 held-out schemas, answer requires a discovered literal):
  **13.6% → 51.5%.** A one-shot model is structurally capped here; the +38 points is the tool paying off.
- **General OOD** (new databases, the best predictor of "a fresh schema at work"): **59.3% → 64.2%.**
- **General in-distribution**: 57.5% → 47.5%. This is v0.9's over-fit home turf; v1.0's *one-shot*
  accuracy here is at parity (~55%), so the gap is residual over-gathering, not lost SQL competence.

---

## Part II — What didn't work (and what it taught us)

The shortest path to v1.0 was not the first path tried. Each detour was instructive.

### 7. Tool-use RL: the gather-but-ignore wall

The first instinct was reinforcement learning: roll out trajectories, reward denotation-correct finals,
GRPO. It plateaued at ~17% on value-discovery — barely above one-shot. The failure was specific and
diagnosable: warm-started from the strongly one-shot prior, the model learned to **gather (64/66) but
then ignore the observation**, reverting to its single-shot guess (`to_date IS NULL` instead of the
sentinel it had just seen). RL can only reinforce behaviors it *samples*; the policy almost never
sampled "bind the discovered value," so there was nothing to reinforce — an exploration wall that
reward-shaping (gather cost, literal-hit bonuses, prompt nudges) could not break.

**Lesson:** when the target behavior is far from the prior's support, exploration-based learning stalls.
Provide the behavior directly (supervision) instead of hoping to sample it. This is what redirected the
whole project to behavioral cloning.

### 8. Clean base vs. warm start: a rule that didn't generalize

A standing heuristic from prior work: *start new behaviors from the clean base, not the over-trained
champion* — because an over-trained prior fights newly-explored behavior. We tested it for BC and found
the opposite. BC from the clean base reached value-discovery parity but **lost ~15 points of general
SQL** (461 trajectories don't replace 55k SFT's breadth). BC warm-started from v0.9 **kept the generality
and learned the tool**, beating clean-base on all three suites.

**Lesson:** the clean-base rule is about *exploration* (RL/on-policy), where a prior competes with newly
sampled behavior. Under *direct supervision* (BC), the prior is an asset, not a liability — the gradient
points straight at the new behavior regardless of what the model already knows. Match the starting
checkpoint to the learning signal, not to a blanket rule.

### 9. The calibration⊥discovery tension (the 2B ceiling)

Value-discovery wants the model to gather aggressively; general SQL wants it to answer directly. Every
intervention that pushed one pushed the other the wrong way. Adding don't-gather calibration recovered
OOD but capped value-discovery; adding discovery signal lifted value-discovery but re-broke OOD
calibration. The Pareto frontier is real and it sits where v1.0 sits.

This echoes an earlier finding from capacity/coverage sweeps: **at 2B, out-of-distribution accuracy is
bound by capacity and data coverage, not by optimization.** No reward tweak or data re-weighting moves
the frontier; it's a property of the model size.

### 10. On-policy DAgger: works on its target, hits the same wall

The principled next step after BC is on-policy correction (DAgger): let the *student* drive the harness,
and have the teacher relabel the student's **own failure states** with verified-correct actions. We built
it (text-level, no logits, no pod — the student rolls out locally, the teacher relabels via API, ~$0.16).
It worked exactly as theory predicts on its target: value-discovery **50% → 57.6%**. But the on-policy
gather-corrections re-rotated the model toward gathering and **OOD calibration fell 63% → 52%** — the
same tension from §9, now seen from the other side. Restricting to only the verified *binding* finals
softened the trade but still didn't strictly beat the calibrated v1.0.

**Lesson:** on-policy correction is the right tool and it demonstrably moves the metric it targets — but
it cannot manufacture capacity. Breaking the frontier needs a bigger student (or a richer signal that
transfers the teacher's full calibration, i.e. logit-level KL — a confirmed-viable but pod-scale
escalation we chose not to spend on for a marginal-confidence bet).

### 11. Smaller potholes

- **Quantization:** Q4_K_M is a false economy here — it erodes precisely the small trained deltas that
  make v1.0 better than v0.9. Ship Q8.
- **Fidelity is fragile:** the inference harness must reproduce the *exact* training prompt — instruction
  string, observation format, budget cues — or accuracy silently drops. The production package shares one
  contract module to prevent drift.
- **Infra rabbit-holes:** chasing local generation speed-ups (custom attention/conv kernels) on a
  Blackwell consumer GPU was a dead end; the disciplined move was to accept the torch fallback and keep
  the experiment loop tight rather than yak-shave the kernel.

---

## Evaluation methodology

- **Denotation scoring**, not string match: execute predicted and gold SQL, compare row-sets (order-
  insensitive, cell-normalized). A query is correct iff it returns the right rows.
- **Leak-guarded suites**: every eval schema and question is held disjoint from all training pools
  (checked by schema id, db file, and question text).
- **Verified gold**: benchmark questions and gold SQL were authored and execution-verified by an
  LLM-judge swarm, then spot-checked.
- **The value-discovery suite is open-sourced** so the gather-use gap is independently measurable.

## Limitations & future work

- v1.0 sits on the 2B calibration⊥discovery frontier; the residual value-discovery errors are
  *wrong-binding* (gathers, then binds the wrong value). The high-confidence lever is **student
  capacity**, not more 2B optimization.
- **Logit-KL on-policy distillation** (soft targets from a vocab-aligned 27B teacher, computed live and
  never stored) may transfer the teacher's calibration in a way hard BC/DAgger can't — a pod-scale
  experiment, parked.
- **Verifiable RL polish** (denotation reward + a small gather cost) is the natural fix for the residual
  wrong-bindings and any remaining over-gather, now that the behavior already exists.

## Artifacts

- Model: `dreeseaw/cleo` (Q8_0 GGUF + bf16). Package: `cleo` (`Cleo.from_gguf(...).ask(question, conn)`).
- Benchmark: the value-discovery suite (open-sourced; schemas reference Spider, TPC-H/DS, and standard
  sample databases — not redistributed wholesale).
- Training schemas derive in part from **SynSQL** and other public text-to-SQL corpora, *referenced, not
  redistributed*.
