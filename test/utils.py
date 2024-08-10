from __future__ import annotations

import asyncio
import concurrent.futures as cf
import ctypes
import functools
import multiprocessing as mp
import multiprocessing.pool
import multiprocessing.managers
import os
import queue
import signal
import threading
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import overload, TYPE_CHECKING, TypeVar, Protocol, Callable
from unittest.util import safe_repr

T = TypeVar('T')
T_contra = TypeVar('T_contra', contravariant=True)
U = TypeVar('U')
U_contra = TypeVar('U_contra', contravariant=True)

if TYPE_CHECKING:
    from _typeshed import SupportsDunderLE, SupportsDunderGE

    class SupportsLeAndGe(SupportsDunderLE[T_contra],
                          SupportsDunderGE[U_contra],
                          Protocol[T_contra, U_contra]):
        pass  # All from inheritance

    class ExtendsAndGE(T, SupportsDunderGE[U], Protocol[T, U]):
        ...

    class ExtendsAndLE(T, SupportsDunderLE[U], Protocol[T, U]):
        ...


class TestTimeout(Exception):
    ...


def _worker_process(fn, dest, args, kwargs, debug):
    if debug >= 1:
        print(f"Worker process running on pid={os.getpid()}")
    if debug >= 2:
        time.sleep(15)
        print('After sleep')
    # noinspection PyBroadException
    try:
        result = fn(*args, *kwargs)
    except Exception as e:
        print(e)
        dest.put(e)
        raise
    else:
        dest.put(result)


def run_with_timeout(timeout_sec: int, fn, args, kwargs, debug=0):
    """fn must be pickleable.

    debug:
      - 0: normal
      - 1: print pid on start
      - 2: also wait 15s in process to allow time to 'Attach to Process' in Pycharm """
    dest = mp.Queue()
    # The daemon's main purpose is to stop ourselves accidentally creating a fork-bomb
    #  by running run_with_timeout in the worker recursively
    #  (we do weird stuff to get pickle to work which might break?).
    # This is because daemonic processes cannot spawn more processes (even daemonic ones).
    # The worker exiting when we get a Ctrl+C is just an added (but necessary) bonus.
    p = mp.Process(target=_worker_process, args=(fn, dest, args, kwargs, debug), daemon=True)
    p.start()
    p.join(timeout_sec)
    if p.is_alive():
        p.kill()  # Die!
        raise TestTimeout(f"Function took too long (exceeded timeout of {timeout_sec}s)")
    if p.exitcode != 0:
        raise mp.ProcessError("Error in process") from dest.get()
    return dest.get()


async def join_async(p, timeout: float, interval: float = 0.005, join_timeout: float = 0.0):
    start = time.perf_counter()
    while p.is_alive() and time.perf_counter() < start + timeout:
        await asyncio.sleep(interval)
        p.join(join_timeout)


# TODO: make class for this async stuff
# async def _run_in_new_process(target, args, timeout: float, interval: float = 0.005):
#     # The daemon's main purpose is to stop ourselves accidentally creating a fork-bomb
#     #  by running run_with_timeout in the worker recursively
#     #  (we do weird stuff to get pickle to work which might break?).
#     # This is because daemonic processes cannot spawn more processes (even daemonic ones).
#     # The worker exiting when we get a Ctrl+C is just an added (but necessary) bonus.
#     p = mp.Process(target=target, args=args, daemon=True)
#     p.start()
#     await join_async(p, timeout, interval)
#     if p.is_alive():
#         p.kill()  # Die!
#         raise TestTimeout(f"Function took too long (exceeded timeout of {timeout}s)")
#     return p


# class _OutcomeType(StrEnum):
#     RUNNING = 'running'
#     SUCCESS = 'success'
#     ERROR = 'error'
#     TIMEOUT = 'timeout'
#
#     def is_finished(self):
#         return self != _OutcomeType.RUNNING


# @dataclass
# class _Outcome:
#     _outcome_type: _OutcomeType = d_field(init=False, default=_OutcomeType.RUNNING)
#     _exc: BaseException | None = d_field(init=False, default=None)
#     _value: object | None = d_field(init=False, default=None)
#
#     def is_finished(self):
#         return self._outcome_type.is_finished()
#
#     @property
#     def value(self):
#         assert self._outcome_type == _OutcomeType.SUCCESS
#         return self._value
#
#     @value.setter
#     def value(self, value: object):
#         assert self._outcome_type == _OutcomeType.RUNNING
#         self._value = value
#         self._outcome_type = _OutcomeType.SUCCESS
#
#     def set_value(self, value: object):
#         self.value = value
#
#     @property
#     def exc(self):
#         assert self._outcome_type == _OutcomeType.ERROR
#         return self._exc
#
#     @exc.setter
#     def exc(self, exc: BaseException):
#         assert self._outcome_type == _OutcomeType.RUNNING
#         self._exc = exc
#         self._outcome_type = _OutcomeType.SUCCESS
#
#     def set_exc(self, exc: BaseException):
#         self.exc = exc
#
#     @property
#     def state(self):
#         return self._outcome_type
#
#     def set_timed_out(self):
#         assert self._outcome_type == _OutcomeType.RUNNING
#         self._outcome_type = _OutcomeType.TIMEOUT
#
#     @classmethod
#     def from_exc(cls, exc: BaseException):
#         self = cls()
#         self.set_exc(exc)
#         return self
#
#     @classmethod
#     def from_value(cls, value: object):
#         self = cls()
#         self.set_value(value)
#         return self


def _pool_worker(send_my_pid: mp.managers.ValueProxy[int], fn, args, kwargs, debug):
    send_my_pid.set(os.getpid())  # Send it so main process knows what to kill
    return fn(*args, **kwargs)  # No exception handling - pool does it for us


# async def _run_with_timeout_async_pool(pool: mp.pool.Pool, timeout: float, target,
#                                        args=(), kwargs=None, debug=0):
#     m: mp.managers.SyncManager = mp.Manager()
#     send_your_pid = m.Value(ctypes.c_uint64, 0)
#     fut = cf.Future()  # We aren't meant to use this apparently but we NEED to be thread-safe
#     pool.apply_async(
#         _pool_worker, (send_your_pid, target, args, kwargs or {}, debug), {},
#         fut.set_result, fut.set_exception)
#     try:
#         return await asyncio.wait_for(asyncio.wrap_future(fut), timeout)
#     except asyncio.TimeoutError:
#         child_pid = send_your_pid.get()
#         os.kill(child_pid, signal.SIGTERM)
#         raise TestTimeout(f"Function took too long (exceeded timeout of {timeout}s)")
#     except Exception as e:
#         raise mp.ProcessError("Error in process") from e

def _pool_worker_2(fn, args, kwargs, debug=0):
    if debug >= 1:
        print(f"Worker process running on pid={os.getpid()}")
    if debug >= 2:
        time.sleep(15)
        print('After sleep')
    return fn(*args, **kwargs)  # No exception handling - pool does it for us


async def _run_with_timeout_async_pool(pool: SimpleProcessPool, timeout: float, target,
                                       args=(), kwargs=None, debug=0):
    try:
        return await pool.apply(_pool_worker_2, (target, args, kwargs, debug),
                                {}, timeout)
    except TimeoutError as e:
        raise TimeoutError(f"Function took too long "
                           f"(exceeded timeout of {timeout}s)") from e
    except Exception as e:
        raise mp.ProcessError("Error in process") from e


@dataclass
class _Task:
    inputs: tuple[int, Callable, tuple, dict[str, ...]]
    output: tuple[int, bool, object | Exception] = None


class SimpleProcessPool:
    def __init__(self, processes: int = None):
        self.n_processes = processes or os.cpu_count() or 4
        self.processes: list[_ProcessWrapper] = self._create_processes()
        self.key = 0
        self.tasks: dict[int, _Task] = {}

    def _create_processes(self):
        return [_ProcessWrapper(self, i)
                for i in range(self.n_processes)]

    def _kill_and_restart(self, i: int):
        print(f'Killing and restarting dead worker {i}')
        self.processes[i].p.kill()
        self.processes[i] = _ProcessWrapper(self, i)

    async def _wait_for_empty_process(self, interval: float = 0):
        while True:
            for i, p in enumerate(self.processes):
                self._update_process(i)
                if p.is_waiting():
                    return i, p
            print(f'Waiting for empty worker')
            await asyncio.sleep(interval)

    def _update_process(self, i: int):
        p = self.processes[i]
        if p.is_dead():
            self._kill_and_restart(i)
        p.update_output_status()

    def _update_all_processes(self):
        for i, _ in enumerate(self.processes):
            self._update_process(i)

    async def apply(self, fn, args, kwargs, timeout: float = None, interval: float = 0):
        key = self.key
        self.key += 1
        task = self.tasks[key] = _Task((key, fn, args, kwargs))
        # Only submit one task at a time, avoids being killed
        # because something else timed out, etc.
        idx, p = await self._wait_for_empty_process(interval)
        p.submit(task)
        start = time.perf_counter()
        while self.tasks[key].output is None:
            if timeout and time.perf_counter() > start + timeout:
                self._kill_and_restart(idx)
                del p  # I don't feel the need to keep that in tb_frame.f_locals forever
                raise TimeoutError
            await asyncio.sleep(interval)
            self._update_all_processes()
        output = self.tasks[key].output
        del self.tasks[key]  # Don't leak memory
        _, success, value_or_err = output
        if success:
            return value_or_err
        else:
            raise value_or_err


class _ProcessWrapper:
    def __init__(self, parent: SimpleProcessPool, i: int):
        self.i = i
        self.pool = parent
        self.tasks_in = mp.Queue()
        self.results_out = mp.Queue()
        self.p = mp.Process(name=f'_SimpleProcessPool:worker-{i}',
                            target=self._worker,
                            args=(self.tasks_in, self.results_out),
                            daemon=True)
        self.waiting = True
        self.p.start()

    def submit(self, task: _Task):
        print(f'Submitted {task} to pool {self.i}')
        self.waiting = False
        self.tasks_in.put(task.inputs)

    def is_dead(self):
        return not self.p.is_alive()

    def is_waiting(self):
        self.update_output_status()
        return self.p.is_alive() and self.waiting

    def update_output_status(self):
        try:
            result = self.results_out.get(False)
        except queue.Empty:
            return
        self.waiting = True
        key, *_ = result
        self.pool.tasks[key].output = result

    @classmethod
    def _worker(cls, tasks_in: mp.Queue, results_out: mp.Queue):
        while True:
            print(f'{mp.current_process().name}: Waiting for tasks')
            try:
                task = tasks_in.get()
            except (EOFError, ValueError):
                return
            print(f'{mp.current_process().name}: Received task {task}')
            if task is None:
                return  # Sentinel value, stop
            key, fn, args, kwargs = task
            try:
                result = key, True, fn(*args, **kwargs)
            except Exception as e:
                result = key, False, mp.pool.ExceptionWithTraceback(e)
            try:
                results_out.put(result)
            except Exception as e:
                results_out.put(mp.pool.MaybeEncodingError(e, result[1]))


async def _run_with_timeout_async_process(timeout: float, fn, args, kwargs,
                                          debug=0, interval=0.005):
    """fn must be pickleable and a regular function **not** coroutine!

    debug:
      - 0: normal
      - 1: print pid on start
      - 2: also wait 15s in process to allow time to 'Attach to Process' in Pycharm
    Note: don't use debug=2 with low timeouts unless you want to get terminate()d"""
    dest = mp.Queue()
    # The daemon's main purpose is to stop ourselves accidentally creating a fork-bomb
    #  by running run_with_timeout in the worker recursively
    #  (we do weird stuff to get pickle to work which might break?).
    # This is because daemonic processes cannot spawn more processes (even daemonic ones).
    # The worker exiting when we get a Ctrl+C is just an added (but necessary) bonus.
    p = mp.Process(target=_worker_process, args=(fn, dest, args, kwargs, debug), daemon=True)
    p.start()
    await join_async(p, timeout, interval)
    if p.is_alive():
        p.kill()  # Die!
        raise TestTimeout(f"Function took too long (exceeded timeout of {timeout}s)")
    if p.exitcode != 0:
        raise mp.ProcessError("Error in process") from dest.get()
    return dest.get()


async def run_with_timeout_async(timeout: float, fn, args, kwargs,
                                 debug=0, interval=0.005, pool: bool | mp.pool.Pool = None):
    """fn must be pickleable and a regular function **not** coroutine!

    debug:
      - 0: normal
      - 1: print pid on start
      - 2: also wait 15s in process to allow time to 'Attach to Process' in Pycharm
    Note: don't use debug=2 with low timeouts unless you want to get terminate()d"""
    if not pool:
        return await _run_with_timeout_async_process(timeout, fn, args, kwargs, debug, interval)
    if pool is True:
        if run_with_timeout_async.default_pool is None:
            run_with_timeout_async.default_pool = SimpleProcessPool(4)
        pool = run_with_timeout_async.default_pool
    return await _run_with_timeout_async_pool(pool, timeout, fn, args, kwargs, debug)


run_with_timeout_async.default_pool = None


def timeout_decor(timeout_sec: int, debug=0):
    def decor(fn):
        # This will be used as a decorator e.g.
        # @timeout_decor(10)
        # def foo(): ...
        # Let's call the actual inner function `foo:orig` and the wrapped one `foo:new`.
        # So if we try to pickle `foo:orig`, it will try to save_global
        @functools.wraps(fn)
        def new_fn(*args, **kwargs):
            return run_with_timeout(
                timeout_sec, _PickleWrapper(outer_func=new_fn), args, kwargs, debug)
        new_fn._timeout_sec_ = timeout_sec
        new_fn._orig_fn_ = fn
        return new_fn
    return decor


def timeout_decor_async(timeout_sec: int, debug=0, interval=0.005, pool=False):
    def decor(fn):
        # This will be used as a decorator e.g.
        # @timeout_decor(10)
        # def foo(): ...
        # Let's call the actual inner function `foo:orig` and the wrapped one `foo:new`.
        # So if we try to pickle `foo:orig`, it will try to save_global
        @functools.wraps(fn)
        async def new_fn(*args, **kwargs):
            return await run_with_timeout_async(
                timeout_sec, _PickleWrapper(outer_func=new_fn), args, kwargs,
                debug, interval, pool)
        new_fn._timeout_sec_ = timeout_sec
        new_fn._orig_fn_ = fn
        return new_fn
    return decor


class _PickleWrapper:
    def __init__(self, outer_func):
        """We store the outer func as that is the one that pickle
        will get when it does a load_global"""
        self._outer_func_ = outer_func

    @property
    def func(self):
        func = self._outer_func_
        while getattr(func, '_orig_fn_', None) is not None:
            # noinspection PyProtectedMember
            func = func._orig_fn_
        return func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class TestCaseUtils(unittest.TestCase):
    @overload  # b:GE & b:LE
    def assertBetweenIncl(self, lo: T, hi: U, value: SupportsLeAndGe[U, T],
                          msg: str | None = None): ...

    @overload  # a:LE & c:GE
    def assertBetweenIncl(self, lo: SupportsDunderLE[T], hi: SupportsDunderGE[T], value: T,
                          msg: str | None = None): ...

    @overload  # b:GE & c:GE  # Does this work? Who knows?
    def assertBetweenIncl(self, lo: T, hi: SupportsDunderGE[U], value: ExtendsAndGE[U, T],
                          msg: str | None = None): ...

    @overload  # a:LE & b:LE  # Does this work? Who knows?
    def assertBetweenIncl(self, lo: SupportsDunderLE[T], hi: U, value: ExtendsAndLE[T, U],
                          msg: str | None = None): ...

    def assertBetweenIncl(self, lo: SupportsDunderLE, hi: SupportsLeAndGe,
                          value: SupportsDunderGE, msg=None):
        """Just like self.assertTrue(lo <= value <= hi), but with a nicer default message."""
        if lo <= value <= hi:
            return
        standard_msg = (f'{safe_repr(value)} is not between'
                        f' {safe_repr(lo)} and {safe_repr(hi)}')
        self.fail(self._formatMessage(msg, standard_msg))

    @staticmethod
    def isProperCwdSet():
        return Path('./.github/workflows').exists()

    def setProperCwd(self):
        """Sets the working directory to the project root (if it isn't set already)"""
        if self.isProperCwdSet():
            return
        self._old_cwd = os.getcwd()
        dirname = Path(__file__).parent
        os.chdir(dirname.parent)
        assert self.isProperCwdSet()
        self.addCleanup(self.resetCwd)

    def resetCwd(self):
        os.chdir(self._old_cwd)
