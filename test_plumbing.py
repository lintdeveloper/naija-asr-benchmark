"""Checks the parent/child plumbing in load_samples without touching the network.

`load_samples` was changed to fetch in a child process (see its docstring). That
moved every sample across a process boundary, which is the part most likely to
break silently: a numpy waveform has to pickle, `audio` has to stay nested for
the consumers, and a child that dies must not leave the parent hanging.

The network cannot verify this — FLEURS streaming does not populate a local
parquet cache, so there is nothing to replay offline, and a flaky CDN gives a
false failure. These checks are synthetic on purpose.

    ./.venv/bin/python test_plumbing.py
"""

import multiprocessing as mp
import queue as queue_mod
import sys

import numpy as np

import milestone0


def _good_child(_config, n, q):
    q.put(
        (
            "ok",
            [
                {
                    "audio": {
                        "array": np.linspace(-1, 1, 16000, dtype=np.float32),
                        "sampling_rate": 16000,
                    },
                    "transcription": f"utterance {i}",
                    "raw_transcription": "",
                    "_fields": ["audio", "id", "transcription"],
                }
                for i in range(n)
            ],
        )
    )


def _dying_child(_config, _n, _q):
    sys.exit(1)  # exits without putting anything on the queue


def run(target, config="xx_yy", n=3, timeout=20):
    """Mirror load_samples exactly: read the queue BEFORE joining.

    Joining first deadlocks once the payload exceeds the pipe buffer — the child
    blocks in put() waiting for a reader while the parent blocks in join()
    waiting for the child. That is what this file caught, and the bug was real:
    three synthetic 64KB waveforms were already enough to hang it, and five real
    FLEURS samples are several megabytes.
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    child = ctx.Process(target=target, args=(config, n, q), daemon=True)
    child.start()
    try:
        result = q.get(timeout=timeout)
    except queue_mod.Empty:
        result = None
    child.join(10)
    timed_out = child.is_alive()
    if timed_out:
        child.kill()
        child.join(5)
    return timed_out, result


def main():
    failures = []

    def check(name, cond, detail=""):
        print(f"  {'✓' if cond else '✗'} {name}{'' if cond else '  — ' + detail}")
        if not cond:
            failures.append(name)

    # 1. a numpy waveform survives the process boundary intact
    alive, result = run(_good_child)
    ok = result is not None and result[0] == "ok"
    check("child result crosses the boundary", ok and not alive)
    if ok:
        samples = result[1]
        check("all samples arrive", len(samples) == 3, f"got {len(samples)}")
        a = samples[0]["audio"]
        check("audio stays NESTED (consumers rely on it)", "array" in a and "sampling_rate" in a)
        check("waveform is still a numpy array", isinstance(a["array"], np.ndarray),
              f"got {type(a['array']).__name__}")
        check("waveform survives byte-exact", len(a["array"]) == 16000
              and np.isclose(a["array"][0], -1.0) and np.isclose(a["array"][-1], 1.0))
        check("duration maths works (show_references)",
              abs(len(a["array"]) / a["sampling_rate"] - 1.0) < 1e-9)
        check("transcription present", samples[0]["transcription"] == "utterance 0")

    # 2. a child that dies must not hang the parent, and must not look like success
    alive, result = run(_dying_child, timeout=10)
    check("dead child does not hang the parent", not alive)
    check("dead child yields no payload", result is None)

    # 3. the payload that caused the original deadlock now gets through
    alive, result = run(_good_child, n=8, timeout=25)
    check("8 samples (~512KB, well past a pipe buffer) do not deadlock",
          result is not None and not alive)

    # 3. the constant is overridable, since a slow connection needs a bigger ceiling
    check("STREAM_TIMEOUT_S is an int", isinstance(milestone0.STREAM_TIMEOUT_S, int),
          repr(milestone0.STREAM_TIMEOUT_S))
    check("STREAM_TIMEOUT_S is generous (>= 600s)", milestone0.STREAM_TIMEOUT_S >= 600,
          f"{milestone0.STREAM_TIMEOUT_S}s — 180s failed a WORKING Hausa fetch")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("All plumbing checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
