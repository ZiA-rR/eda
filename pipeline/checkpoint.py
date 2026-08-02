"""
checkpoint.py
-------------
Resumable loops, so a dropped Colab session costs one item instead of the
whole run.

The pattern everywhere else in this pipeline is a long loop that only saves
at the end. That is fine until the session drops at item 90 of 100, which
on the scoring stage means paying for those API calls twice.

Usage:

    from checkpoint import resumable_map

    topics = resumable_map(
        items    = events,
        key_fn   = lambda ev: f"{ev['asset']}_{ev['date']}",
        work_fn  = lambda ev: build_topic(ev["asset"], ev["date"]),
        path     = "topics_progress.json",
    )

Run the same cell again after a disconnect and it picks up where it left
off. Delete the file to start fresh.
"""

import json
import os
import time
import traceback
from typing import Callable, List, Any, Dict, Optional


def _load(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"done": {}, "failed": {}}
    try:
        with open(path) as f:
            state = json.load(f)
        state.setdefault("done", {})
        state.setdefault("failed", {})
        return state
    except (json.JSONDecodeError, OSError):
        # a half-written file from a hard kill. Keep the wreckage rather
        # than silently deleting someone's hours of work.
        backup = path + ".corrupt"
        os.replace(path, backup)
        print(f"  checkpoint was unreadable, moved to {backup}, starting fresh")
        return {"done": {}, "failed": {}}


def _save(path: str, state: Dict[str, Any]) -> None:
    """
    Write to a temp file then move it into place. A move is atomic, so the
    checkpoint is never left half-written if the session dies mid-save.
    """
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, default=str)
    os.replace(tmp, path)


def resumable_map(items: List[Any],
                  key_fn: Callable[[Any], str],
                  work_fn: Callable[[Any], Any],
                  path: str,
                  save_every: int = 1,
                  retry_failed: bool = False,
                  verbose: bool = True) -> List[Any]:
    """
    Apply work_fn to every item, saving progress as it goes.

    key_fn       must return a stable unique string per item. It is what
                 identifies an item as already done, so it has to be the
                 same across sessions.
    save_every   1 writes after every item, which is safest. Raise it if
                 writing to Drive feels slow.
    retry_failed items that raised last time are skipped by default. Set
                 this to retry them.
    """
    state = _load(path)
    done, failed = state["done"], state["failed"]

    if verbose and (done or failed):
        print(f"resuming from {path}: {len(done)} done, {len(failed)} failed")

    todo = []
    for it in items:
        k = key_fn(it)
        if k in done:
            continue
        if k in failed and not retry_failed:
            continue
        todo.append((k, it))

    if verbose:
        print(f"{len(todo)} to process, {len(items) - len(todo)} skipped\n")

    since_save = 0
    for i, (k, it) in enumerate(todo, 1):
        if verbose:
            print(f"[{i}/{len(todo)}] {k}")
        try:
            done[k] = work_fn(it)
            failed.pop(k, None)
        except KeyboardInterrupt:
            # save what we have before letting the interrupt through
            _save(path, {"done": done, "failed": failed})
            if verbose:
                print(f"\ninterrupted, {len(done)} results saved to {path}")
            raise
        except Exception as e:
            failed[k] = {"error": f"{type(e).__name__}: {e}",
                         "traceback": traceback.format_exc()[-800:]}
            if verbose:
                print(f"   failed: {type(e).__name__}: {e}")

        since_save += 1
        if since_save >= save_every:
            _save(path, {"done": done, "failed": failed})
            since_save = 0

    _save(path, {"done": done, "failed": failed})

    if verbose:
        print(f"\n{len(done)} done, {len(failed)} failed, saved to {path}")
        if failed:
            print("failed keys:", list(failed)[:10])

    return list(done.values())


def checkpoint_status(path: str) -> Dict:
    """What state is a checkpoint file in."""
    state = _load(path)
    return {
        "path": path,
        "exists": os.path.exists(path),
        "done": len(state["done"]),
        "failed": len(state["failed"]),
        "failed_keys": list(state["failed"])[:20],
    }


def show_failures(path: str, n: int = 5) -> None:
    """Print the errors, for working out what went wrong."""
    state = _load(path)
    for k, v in list(state["failed"].items())[:n]:
        print(f"--- {k} ---")
        print(v.get("error", ""))
        print()


def reset_checkpoint(path: str, confirm: bool = False) -> None:
    """Delete a checkpoint and start over. Needs confirm=True."""
    if not confirm:
        print(f"this deletes {path}. Call with confirm=True if you mean it.")
        return
    if os.path.exists(path):
        os.remove(path)
        print(f"deleted {path}")
