# naija-asr-bench

Measuring Nigerian-language ASR under the conditions it is actually deployed in
— narrowband telephone audio, ambient noise, and code-switching — rather than
the clean read speech public leaderboards score.

Plan: `~/Documents/nigerian-asr-benchmark-plan.pdf`

**Status: Milestone 0 complete (2026-08-27).** Toolchain verified, FLEURS configs resolved, first
zero-shot baselines recorded below.

---

## Milestone 0 — results

Ran 2026-08-27 on an M-series Mac, CPU, `datasets` 3.6 / `transformers` 4.57 / `torch` 2.13.

**FLEURS config names — resolved, no longer guesses:**

| Language | Config | Status |
|---|---|---|
| Hausa | `ha_ng` | confirmed — the plan listed this as an unverified guess |
| Yorùbá | `yo_ng` | confirmed |
| English | `en_us` | confirmed |
| Igbo | `ig_ng` | config exists; the **split would not stream** — see below |

All four config names are therefore confirmed against the live config list (103 configs;
`ig*` matches exactly `ig_ng`). The plan's guesses were all correct.

**Igbo is blocked on data access, not on naming.** `milestone0.py --lang ig` ran for over 25
minutes with no output and had to be killed, while `ha`, `yo` and `en` each completed in minutes.
Since `get_dataset_config_names` returns `ig_ng` immediately, the failure is in fetching the
split, not resolving it. Worth an entry in the plan's §8 risk table: a language can be listed in
FLEURS and still be effectively unavailable, which affects the four-language scope in §2.1. Next
step is a direct shard fetch with a timeout to distinguish a slow shard from a broken one.

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

## Milestone 0 — how to re-run

One evening. The only goal is to prove the toolchain works before any real work
starts. Most projects in an unfamiliar ecosystem die here, for boring reasons:
wrong config name, gated dataset, missing audio codec.

```bash
cd ~/Documents/naija-asr-bench
./setup.sh
./.venv/bin/python milestone0.py
```

Then confirm the other three languages resolve:

```bash
./.venv/bin/python milestone0.py --lang yo
./.venv/bin/python milestone0.py --lang ig
./.venv/bin/python milestone0.py --lang en
```

### What the script does

1. Reports Python / torch / transformers / datasets versions and picks a device
2. **Resolves the FLEURS config name** — the plan lists `ha_ng` as an unverified
   guess. If wrong, the script lists every `ha_*` config and tells you the fix.
3. Streams 5 test samples (no multi-GB download)
4. Prints the reference transcripts
5. Runs one clip through `whisper-tiny` and prints hypothesis beside reference

### Definition of done

**A hypothesis printed next to a reference transcript.** That is all.

`whisper-tiny` will score terribly on Hausa — likely 80–100% WER, possibly
outputting the wrong language entirely. **That is the correct outcome.** You are
testing plumbing, not quality. Do not tune anything. Do not reach for a larger
model. Both belong in Milestone 1 and later.

### If it fails

| Symptom | Cause | Fix |
|---|---|---|
| `python3.11 not found` | System python is 3.9 | `brew install python@3.11` |
| Config not found | The `ha_ng` guess was wrong | Script prints the real candidates — update `EXPECTED_CONFIGS` in `milestone0.py` |
| Audio backend error | libsndfile missing | `brew install libsndfile`, then re-run `./setup.sh` |
| `trust_remote_code` error | `datasets` too new | The `<4` pin should prevent it; check the install actually honoured it |
| Model download stalls | Network / disk | ~150MB for whisper-tiny; check free space |

---

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
LICENSE            MIT
requirements.txt   pinned dependency set (see the comment about datasets<4)
setup.sh           creates .venv with python3.11, installs deps
milestone0.py      the smoke test
```

## Licence

MIT — see [LICENSE](LICENSE).
