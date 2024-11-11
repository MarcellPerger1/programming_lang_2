from __future__ import annotations

import asyncio
import multiprocessing as mp
import multiprocessing.pool
import os
import queue
import time
import weakref
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class _Task:
    inputs: tuple[int, Callable, tuple, dict[str, ...]]
    output: tuple[int, bool, object | Exception] = None


@dataclass
class _TimeoutMgr:
    timeout: float | None = None
    timeout_includes_waiting: bool = False
    start: float = field(default_factory=time.perf_counter)

    def is_timed_out(self, has_task_started: bool):
        return (self.timeout
                and (has_task_started or self.timeout_includes_waiting)
                and time.perf_counter() > self.start + self.timeout)

    def check_timeout(self, has_task_started: bool):
        if self.is_timed_out(has_task_started):
            raise TimeoutError(self.get_timeout_msg(has_task_started))

    def get_timeout_msg(self, has_task_started: bool):
        return ("While running task" if has_task_started
                else "While waiting for empty process in pool")

    def on_start_task(self):
        if self.timeout and not self.timeout_includes_waiting:
            self.start = time.perf_counter()


class SimpleProcessPool:
    def __init__(self, processes: int = None):
        self.n_processes = processes or os.cpu_count() or 4
        self.processes: list[_ProcessWrapper] = self._create_processes()
        self.key = 0
        self.tasks: dict[int, _Task] = {}

    def _create_processes(self):
        return [_ProcessWrapper(self, i) for i in range(self.n_processes)]

    def _append_new_process(self):
        self.processes.append(_ProcessWrapper(self, len(self.processes)))
        self.n_processes += 1

    def grow_processes(self, minsize: int):
        while len(self.processes) < minsize:
            self._append_new_process()

    def _kill_and_restart(self, i: int):
        self.processes[i].kill()
        self.processes[i] = _ProcessWrapper(self, i)

    async def _wait_for_empty_process(self, timeout_ctx: _TimeoutMgr, interval: float = 0):
        while True:
            for i, p in enumerate(self.processes):
                self._update_process(i)
                if p.is_waiting_for_task():
                    return i, p
            timeout_ctx.check_timeout(has_task_started=False)
            await asyncio.sleep(interval)

    def _update_process(self, i: int):
        p = self.processes[i]
        if p.is_dead():
            self._kill_and_restart(i)
        else:
            p.update_output_status()

    def _update_all_processes(self):
        for i, _ in enumerate(self.processes):
            self._update_process(i)

    # We only submit one task at a time, avoids being killed
    # just because something else timed out, etc.
    # so easier to support cancelling, on timeout or otherwise.
    # Also, a worker can only do one thing at a time so no loss of efficiency.
    async def apply(self, fn, args=None, kwargs=None, timeout: float = None,
                    interval: float = 0, timeout_includes_waiting=False):
        key, task = self._create_task(fn, args or (), kwargs or {})
        timeout_ctx = _TimeoutMgr(timeout, timeout_includes_waiting)
        proc_idx = await self._submit_to_empty_process(task, timeout_ctx, interval)
        await self._wait_for_task_result(key, proc_idx, timeout_ctx, interval)
        return self._handle_task_result(key)

    def _handle_task_result(self, key):
        # pop is so that we don't leak memory by keeping the result forever
        _, success, value_or_err = self.tasks.pop(key).output
        if success:
            return value_or_err
        else:
            raise value_or_err

    async def _wait_for_task_result(self, key, proc_idx, timeout_ctx, interval):
        while self.tasks[key].output is None:
            if timeout_ctx.is_timed_out(has_task_started=True):
                self._kill_and_restart(proc_idx)
                raise TimeoutError("While running task")
            await asyncio.sleep(interval)
            self._update_all_processes()

    async def _submit_to_empty_process(
            self, task: _Task, timeout_ctx: _TimeoutMgr, interval: float):
        idx, p = await self._wait_for_empty_process(timeout_ctx, interval)
        p.submit(task)
        timeout_ctx.on_start_task()
        return idx

    def _create_task(self, fn, args, kwargs):
        key = self.key
        self.key += 1
        task = self.tasks[key] = _Task((key, fn, args, kwargs))
        return key, task

    def put_task_result(self, result):
        """Internal!"""
        key, _success, _v = result
        self.tasks[key].output = result


class _ProcessWrapper:
    def __init__(self, parent: SimpleProcessPool, idx: int):
        self.idx = idx
        self.pool = parent
        self.tasks_in = mp.Queue()
        self.results_out = mp.Queue()
        self.p = mp.Process(name=f'_SimpleProcessPool:worker-{idx}',
                            target=self._worker,
                            args=(self.tasks_in, self.results_out),
                            daemon=True)
        self.waiting_for_task = True
        self.p.start()
        # weakref.finalize function is the civilized alternative to __del__
        # with actually sensible semantics, see
        # https://docs.python.org/3.11/library/weakref.html#weakref.finalize
        weakref.finalize(self, self._finalize)

    # (No tracing in finalizers so disable coverage for this)
    def _finalize(self, timeout=0.008):  # pragma: no cover
        # Give children 8ms to clean up, instead of rudely GC-ing them out of
        # existence (ruining coverage by not giving them a chance to run atexit)
        self.close(timeout, force=True)

    def submit(self, task: _Task):
        self.waiting_for_task = False
        self.tasks_in.put(task.inputs)

    def close(self, timeout=0.008, force=False) -> bool:
        # First, politely ask it to close
        if not self.p.is_alive():
            return True
        self.tasks_in.put(None)
        self.p.join(timeout)
        # If it's no longer alive, it did close
        if not self.p.is_alive():
            self.p.close()
            self._close_queues()
            return True
        if force:  # (Didn't close)
            self.kill()
        return False

    def kill(self):
        self.p.kill()
        self._close_queues()

    def _close_queues(self):
        self.tasks_in.close()
        self.results_out.close()

    def is_dead(self):
        return not self.p.is_alive()

    def is_waiting_for_task(self):
        self.update_output_status()
        return self.p.is_alive() and self.waiting_for_task

    def update_output_status(self):
        try:
            result = self.results_out.get(block=False)
        except queue.Empty:
            return
        self.waiting_for_task = True
        self.pool.put_task_result(result)

    @classmethod
    def _worker(cls, tasks_in: mp.Queue, results_out: mp.Queue):
        while True:
            try:
                task = tasks_in.get()
            except (EOFError, ValueError):
                return
            if task is None:
                return  # Sentinel value, stop
            key, fn, args, kwargs = task
            try:
                result = key, True, fn(*args, **kwargs)
            except Exception as e:
                # noinspection PyUnresolvedReferences
                result = key, False, mp.pool.ExceptionWithTraceback(e, e.__traceback__)
            try:
                results_out.put(result)
            except Exception as e:
                # noinspection PyUnresolvedReferences
                results_out.put(mp.pool.MaybeEncodingError(e, result[1]))
