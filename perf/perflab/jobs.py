"""Long computations one after the other in a background thread, with progress and cancel."""

from __future__ import annotations

import itertools
import queue
import threading
import time
import traceback


class Cancelled(Exception):
    pass


class Job:
    def __init__(self, jid: str, kind: str, fn, args: dict):
        self.id, self.kind, self.fn, self.args = jid, kind, fn, args
        self.state = "queued"           # queued | running | done | failed | cancelled
        self.progress, self.text = 0.0, "waiting"
        self.result = None
        self.error = ""
        self.created, self.started, self.finished = time.time(), None, None
        self.cancelled = False

    def step(self, i, n, text=""):
        if self.cancelled:
            raise Cancelled()
        self.progress = float(i) / float(n) if n else 0.0
        self.text = str(text)

    def to_dict(self, with_result: bool = True) -> dict:
        d = {"id": self.id, "kind": self.kind, "state": self.state, "progress": self.progress, "text": self.text,
             "error": self.error, "elapsed": (self.finished or time.time()) - (self.started or time.time())}
        if with_result and self.state == "done":
            d["result"] = self.result
        return d


class Runner:
    def __init__(self, keep: int = 30):
        self.jobs: dict[str, Job] = {}
        self.keep = keep
        self.q: queue.Queue = queue.Queue()
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True, name="perflab-jobs").start()

    def submit(self, kind: str, fn, args: dict) -> Job:
        with self._lock:
            job = Job(f"j{next(self._ids)}", kind, fn, args)
            self.jobs[job.id] = job
            for old in sorted(self.jobs.values(), key=lambda j: j.created)[:-self.keep]:
                if old.state not in ("queued", "running"):
                    del self.jobs[old.id]
        self.q.put(job)
        return job

    def get(self, jid: str) -> Job | None:
        return self.jobs.get(jid)

    def cancel(self, jid: str) -> bool:
        j = self.jobs.get(jid)
        if j is None or j.state not in ("queued", "running"):
            return False
        j.cancelled = True
        return True

    def wait(self, jid: str, timeout: float = 600.0) -> Job:
        j = self.jobs[jid]
        t0 = time.time()
        while j.state in ("queued", "running") and time.time() - t0 < timeout:
            time.sleep(0.02)
        return j

    def _loop(self):
        while True:
            job = self.q.get()
            if job.cancelled:
                job.state = "cancelled"
                continue
            job.state, job.started = "running", time.time()
            try:
                job.result = job.fn(job=job, **job.args)
                job.state, job.progress = "done", 1.0
            except Cancelled:
                job.state = "cancelled"
            except Exception as e:  # noqa: BLE001
                job.state = "failed"
                job.error = f"{type(e).__name__}: {e}" if not isinstance(e, (ValueError, FileNotFoundError, FileExistsError, KeyError)) \
                    else str(e.args[0] if e.args else e)
                job.trace = traceback.format_exc()
            job.finished = time.time()
