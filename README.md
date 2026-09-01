# naija-asr-benchmark

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

PazaBench check: **done**, see below — it covers Hausa, Yorùbá and Igbo on FLEURS, but has no
acoustic-degradation axis, so the contribution here is unaffected.

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

uv run naija-asr-benchmark                  # Hausa
uv run naija-asr-benchmark --lang yo        # Yorùbá, Igbo, English
uv run naija-asr-benchmark --help
```

On a slow connection, raise the fetch deadline:

```bash
NAIJA_ASR_FETCH_TIMEOUT_S=1800 uv run naija-asr-benchmark --lang ha
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

## The PazaBench check — done 2026-09-01

**PazaBench covers Hausa, Yorùbá and Igbo.** Answered from the leaderboard's own source rather
than its announcement posts, which name only the six Kenyan languages the Paza *models* target.
`src/data/language_to_countries_map.json` in
[microsoft/paza-bench](https://huggingface.co/spaces/microsoft/paza-bench) lists 59 languages, of
which seven map to Nigeria: **Hausa, Yorùbá, Igbo**, Adamawa Fulfulde, Borgu Fulfulde, Fula and
Kanuri. Nigerian Pidgin is **not** among them.

| Question | Answer |
|---|---|
| Includes Hausa / Yorùbá / Igbo / Pidgin? | **Yes / Yes / Yes / No** |
| Which models? | 51–52 models incl. Whisper, MMS-1B and the new Paza checkpoints |
| Which eval sets? | 11 dataset groups, incl. **Google FLEURS**, Mozilla Common Voice 23.0, ALFFA and **Naija Voices** |
| Per-utterance outputs? | **No** — aggregate WER / CER / RTFx only; results load from a private `RESULTS_REPO` |
| Degradation conditions? | **None.** Grouped by *speech style* — conversational, read-aloud, unscripted, broadcast, domain — not by acoustic condition. No noise, SNR, narrowband or bandwidth axis anywhere in the metadata |

### What this changes

**Milestone 3 shrinks.** They cover the clean-condition leaderboarding for all three languages on
FLEURS, with more models and more compute than this project has. Do not re-run it. Adopt their
clean numbers as the baseline, cite them, and spend the saved compute on the arms they do not
have.

**The contribution is unaffected, and better defined for it.** The plan already said not to
position this as "the Nigerian ASR leaderboard" because Microsoft Research had built one. That is
now confirmed rather than assumed, and the gap is sharper than expected: **PazaBench has no
acoustic-degradation axis at all.** It varies speech *style* and holds the channel constant. Every
one of the four open questions in §1.5 survives:

- degradation curves for Nigerian languages — **still nobody's**
- tone-orthography as a variable — **still nobody's**
- whether LLM correction helps or harms — **still nobody's**
- entity-level scoring — **still nobody's**

**Two things to lift from them.** Per-utterance outputs are not published, so any comparison
against their numbers is aggregate-to-aggregate — worth stating in the write-up. And **Naija
Voices** is a dataset the plan's §1.2 data landscape does not list; it should be assessed for the
degradation arm, since a Nigerian-collected corpus may be closer to deployment audio than FLEURS
read speech.

**Nigerian Pidgin remains entirely uncovered by anyone**, which makes §2.3's "include it if data
permits" more valuable than when it was written.

## Layout

```
pyproject.toml              deps, ruff, mypy and pytest config — single source
uv.lock                     locked environment
src/naija_asr_benchmark/
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
