# naija-asr-benchmark

Measuring Nigerian-language ASR under the conditions it is actually deployed in
— narrowband telephone audio, ambient noise, and code-switching — rather than
the clean read speech public leaderboards score.

**The full research plan is in this repo:** [`docs/plan.html`](docs/plan.html) —
14 pages, version 1.4. Prior work and what it forecloses, the two contributions, an evidential-status
section separating what is established elsewhere from what is tested here, per-sector WER
thresholds, cost, and the milestone breakdown. [`docs/plan.pdf`](docs/plan.pdf) is the same
document if you would rather read it that way.

**Scope, in one line:** this is an evaluation harness, not a machine-learning project. No training,
no fine-tuning, no novel architecture — it calls existing public models and computes standard error
metrics under acoustic conditions no public leaderboard measures.

**Status: Milestone 1 complete (2026-09-07).** Milestone 0 complete 2026-08-27.

## Milestone 1 — results

`whisper-tiny`, Hausa, 20 clips from a local FLEURS copy, raw text (no normalisation):

| | |
|---|---:|
| clips | 20 |
| reference words | 460 |
| **WER** | **436.1%** |
| **CER** | **316.3%** |
| repetition collapses | **7 of 20 (35%)** |
| WER excluding collapses | 106.9% |
| CER excluding collapses | 46.6% |

### The collapse rate is the result

Corpus WER above 100% does not mean every word is wrong — it means insertions
dwarf the references. **`whisper-tiny` collapsed into repetition loops on 7 of 20
Hausa clips**, emitting hypotheses many times longer than the reference. One
returned 444 characters of `1,2,1,0,1,0,…` against a 56-character reference.

So the headline figures are not quality measurements, and the harness says so
rather than printing them unqualified. On the 13 clips it did not collapse on:
**WER 106.9%, CER 46.6%** — a WER at the edge of what the plan predicts for a
39M-parameter model, and a CER far below it, which is the orthographic signature:
the model hearing Hausa and spelling it in English.

Both figures are always reported together. Dropping inconvenient utterances
quietly is how a benchmark becomes an opinion; the gap between them is itself a
finding.

### What the collapse clip was

Its reference: `kwatancin 802.11n na aiki duk akan mita 2.4ghz da 5.0ghz`. The
**numeral-heavy** clip is the one that failed hardest, which is unlooked-for
early support for §5.1's numeral-error thesis and for contribution 2 — on the
first twenty-clip run, before any degradation was applied.

### Caveats

One model, one language, 20 clips, no normalisation, clean audio. The plan scopes
Milestone 1 as plumbing rather than quality and expects 80–100% WER, so nothing
here is a claim about Hausa ASR in general — only about this checkpoint on this
sample.

---

## Running it

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                       # runtime deps
uv sync --extra dev           # + pytest, ruff, mypy
```

Two commands, one per milestone:

```bash
# Milestone 0 — does the toolchain work at all
uv run naija-asr-benchmark smoke --lang ha

# Milestone 1 — N clips, one model, one WER number
uv run naija-asr-benchmark evaluate --lang ha --clips 20
```

`--lang ha|yo|ig|en`, `--model <hf-checkpoint>`, `--clips N`, `--no-save`,
`--data-file <parquet>`.

**Use `--data-file` for anything you intend to cite.** `huggingface_hub`'s downloader stalled
repeatedly at 0 KB/s while plain HTTP to the same URL sustained 1.2 MB/s, and its resume logic
truncated a 674 MB partial back to 494 MB and corrupted it. Fetching the parquet with an
append-only range loop and pointing the harness at it worked first time:

```bash
curl -L -o data/ha_ng-test.parquet \
  "https://huggingface.co/datasets/google/fleurs/resolve/main/parquet-data/ha_ng/test-00000-of-00001.parquet"

uv run naija-asr-benchmark evaluate --lang ha --clips 20 --data-file data/ha_ng-test.parquet
```

It reads only the rows it needs with pyarrow and decodes them with soundfile.
`load_dataset("parquet", …)` materialises the whole 734 MB into an Arrow cache before any
`select()` applies, which timed out at 900 s for a twenty-clip run on a **local** file. Results are written to
`results/` as JSON including every utterance — Milestone 4 hand-categorises ~400 errors, and
re-running inference to recover them would be wasteful. `results/` is gitignored; those are
artifacts, not source.

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
