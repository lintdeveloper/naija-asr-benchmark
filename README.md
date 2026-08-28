# naija-asr-bench

Measuring Nigerian-language ASR under the conditions it is actually deployed in
— narrowband telephone audio, ambient noise, and code-switching — rather than
the clean read speech public leaderboards score.

Plan: `~/Documents/nigerian-asr-benchmark-plan.pdf`

**Status: Milestone 0 complete (2026-08-27).** Toolchain verified, FLEURS configs resolved, first
zero-shot baselines recorded below. Re-verified end to end on 2026-08-28 after the package
restructure: the same Hausa clip produces a byte-identical hypothesis, so the restructure is
behaviour-preserving.

---

## Milestone 0 — results

Ran 2026-08-27 on an M-series Mac, CPU, `datasets` 3.6 / `transformers` 4.57 / `torch` 2.13.

**FLEURS config names — resolved, no longer guesses:**

| Language | Config | Status |
|---|---|---|
| Hausa | `ha_ng` | confirmed — the plan listed this as an unverified guess |
| Yorùbá | `yo_ng` | confirmed |
| English | `en_us` | confirmed |
| Igbo | `ig_ng` | confirmed; the split did **not fetch here** — cause not isolated, see below |

All four config names are therefore confirmed against the live config list (103 configs;
`ig*` matches exactly `ig_ng`). The plan's guesses were all correct.

**Igbo did not fetch here, and the cause is NOT isolated.** `--lang ig` ran over 25 minutes with
no output and had to be killed, while `ha`, `yo` and `en` each completed in minutes. The obvious
reading — that `ig_ng` is unavailable — **is not supported**, because a later Hausa run failed the
same way:

```
'HTTPSConnectionPool(host='us.aws.cdn.hf.co', port=443): Read timed out.'
  … GET .../parquet-data/ha_ng/test-00000-of-00001.parquet
Retrying in 1s [Retry 1/5].
```

Both `huggingface.co` and `us.aws.cdn.hf.co` answer a plain request in under 1.5s, so the hosts are
reachable — it is the large parquet transfer that times out. **The network is a confound here, so
the Igbo result carries no information about `ig_ng`** until it is re-run on a connection that
reliably completes a Hausa fetch. It does not belong in §8 as a data-availability risk yet.

What it did establish is an engineering constraint, which is in the harness now: a fetch can block
in a way that **`signal.alarm` cannot interrupt** — a probe with a 90s alarm ran past 330s and
needed `SIGTERM` from outside. So the deadline must be enforced by a parent process watching a
child, not by the work watching itself. `load_samples` does that, with a generous ceiling
(`RESILIX_STREAM_TIMEOUT_S`, default 900s). Milestone 1 loops over four languages and several
models, and without this one bad shard or one flaky minute hangs the whole run.

**First zero-shot baselines, `openai/whisper-tiny`, one utterance per language:**

| Language | WER | CER | Behaviour |
|---|---:|---:|---|
| English (control) | 5.3% | 1.0% | 18/19 words correct |
| Hausa | 95.7% | 29.1% | phonetically close, orthographically English |
| Yorùbá | ~100% | — | emitted CJK characters (`羽毛` repeated) — language-ID failure |

### Read these numbers correctly

**This is not a measurement of Hausa or Yorùbá ASR.** `whisper-tiny` is 39M parameters, the
smallest model in the family, and this is a single utterance per language with no averaging. The
plan is explicit that Milestone 0 tests plumbing, not quality. What the numbers support is the
*shape* of the gap under one identical protocol — nothing about `whisper-large`, MMS, or any
fine-tuned model.

Two observations worth carrying into Milestone 1:

**Hausa's error is orthographic, not acoustic.** 95.7% WER against 29.1% CER is a large split:
the model hears the phonetics and renders them in English spelling — `ginshiƙi` → `deginshiki`,
`walƙiya` → `valkiya`. That is the mechanism §2.2 predicts, appearing on the first utterance.

**Yorùbá contradicts the §2.2 prediction, and the confound is model scale.** The plan expects
Yorùbá to lead, since tone is orthographically marked and it scores best on SSA-COMET (57.0).
Instead `whisper-tiny` fails hardest on it, switching script entirely rather than degrading. The
likely reading is that at 39M parameters language identification dominates and swamps any tone
effect. **The tone-orthography experiment therefore cannot be run at this model size** — it needs
a size at which all four languages are at least identified. Record in §8 as a scale confound and
choose Milestone 1's model floor accordingly.

---

## Running it

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                       # runtime deps
uv sync --extra dev           # + pytest, ruff, mypy

uv run naija-asr-bench                  # Hausa
uv run naija-asr-bench --lang yo        # Yorùbá, Igbo, English
uv run naija-asr-bench --help
```

On a slow connection, raise the fetch deadline:

```bash
NAIJA_ASR_FETCH_TIMEOUT_S=1800 uv run naija-asr-bench --lang ha
```

### What it does

1. Reports the toolchain and picks a device
2. **Resolves the FLEURS config against the live list** rather than trusting a recorded name
3. Streams 5 test samples — no multi-gigabyte download
4. Prints the reference transcripts
5. Runs one clip through `whisper-tiny` and prints hypothesis beside reference

### Definition of done

**A hypothesis printed next to a reference transcript.** That is all. `whisper-tiny` scores
terribly on Hausa — 80–100% WER, sometimes the wrong language entirely — and **that is the correct
outcome.** You are testing plumbing, not quality. Do not tune anything; that belongs in Milestone 1.

### Checks

```bash
uv run pytest       # 17 tests, no network needed
uv run ruff check .
uv run ruff format --check .
uv run mypy         # strict
```

The tests are deliberately network-free. FLEURS streaming leaves no local parquet cache, so there
is nothing to replay offline, and a flaky CDN produces false failures — synthetic fixtures are the
only honest option for the fetch path.

Two bugs were found by these checks rather than by running the thing:

- **A deadlock in the bounded fetch.** The first version joined the child before reading the
  queue, which hangs once the payload exceeds the pipe buffer — the child blocks in `put()` waiting
  for a reader while the parent blocks in `join()` waiting for the child. Three 64KB waveforms
  reproduced it; real FLEURS rows are ~1MB each, so **every language would have reported a bogus
  timeout.**
- **A union return from `transformers.pipeline`.** It yields a dict for a single input but a list
  of dicts in some versions. Assuming the dict form type-checks against `Any` and raises
  `AttributeError` at runtime on the other path. `mypy --strict` found it; no test would have.

## Also part of Milestone 0: the PazaBench check

Manual, and worth doing before Milestone 3 — it could remove a third of that
milestone's work.

Microsoft Research's **PazaBench** is a public African ASR leaderboard covering
39 African languages and 52 models, hosted on HuggingFace. Its initial depth is
on six Kenyan languages. Find out:

- [ ] Does it include Hausa, Yorùbá, Igbo or Nigerian Pidgin?
- [ ] Which models are scored for those languages?
- [ ] Which eval set does it use — FLEURS, Common Voice, or its own?
- [ ] Are the per-utterance outputs published, or only aggregate scores?
- [ ] Any degradation / noise conditions, or clean audio only?

**If it covers Nigerian languages on FLEURS:** adopt its clean-condition numbers
as the baseline instead of re-running them, cite it, and spend the saved compute
on the degradation and LLM-correction arms — which is where this project's
contribution actually lives.

**If it later adds degradation conditions:** the contribution narrows to the
tone-orthography experiment and entity/numeral accuracy. Those two need to stay
central regardless.

Record the answer in the plan's §8 risk table.

---

## Layout

```
pyproject.toml              deps, ruff, mypy and pytest config — single source
uv.lock                     locked environment
src/naija_asr_bench/
  cli.py                    argparse, orchestration, exit codes
  fleurs.py                 config resolution + the bounded fetch + dataclasses
  asr.py                    transcription
  environment.py            toolchain reporting and device choice
  console.py                presentation only, so logic is testable
  errors.py                 SmokeError — nothing below the CLI calls sys.exit
tests/                      pytest; network-free
```

`src` layout so tests run against the installed package rather than the working directory. Heavy
imports (`torch`, `transformers`, `datasets`) stay inside functions — `--help` should not cost
seconds, and a test asserts that.

## Licence

MIT — see [LICENSE](LICENSE).
