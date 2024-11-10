from __future__ import annotations

import asyncio
import functools
import multiprocessing as mp
import os
import time

from util.simple_process_pool import SimpleProcessPool


class TestTimeout(Exception):
    ...


# coverage.py can't look inside other processes
def _worker_process(fn, dest, args, kwargs, debug):  # pragma: no cover
    if debug >= 1:
        print(f"Worker process running on pid={os.getpid()}")
    if debug >= 2:
        time.sleep(15)
        print('After sleep')
    if dest is None:
        # No error handling channel - let caller handle it (e.g. the pool can handle it)
        return fn(*args, **kwargs)
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


async def join_async(p, timeout: float, interval: float = 0, join_timeout: float = 0.0):
    start = time.perf_counter()
    while p.is_alive() and time.perf_counter() < start + timeout:
        await asyncio.sleep(interval)
        p.join(join_timeout)


async def _run_with_timeout_async_pool(pool: SimpleProcessPool, timeout: float, target,
                                       args=(), kwargs=None, debug=0):
    try:
        return await pool.apply(_worker_process, (target, None, args, kwargs, debug),
                                {}, timeout)
    except TimeoutError as e:
        raise TimeoutError(f"Function took too long "
                           f"(exceeded timeout of {timeout}s)") from e
    except Exception as e:
        raise mp.ProcessError("Error in process") from e


async def _run_with_timeout_async_process(timeout: float, fn, args, kwargs,
                                          debug=0, interval: float = 0):
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
                                 debug=0, interval=0,
                                 pool: bool | int | SimpleProcessPool = None):
    """fn must be pickleable and a regular function **not** coroutine!

    debug:
      - 0: normal
      - 1: print pid on start
      - 2: also wait 15s in process to allow time to 'Attach to Process' in Pycharm
    Note: don't use debug=2 with low timeouts unless you want to get terminate()d

    pool:
     Use this if you're going to call this in a loop (with pool=False,
      this function is rather expensive on Windows as it creates an entire new
      process).
     If you're only calling this once, pool is rather useless as the first time
      it is called, it creates all n processes.
     You can:
      - Pass your own SimpleProcessPool
      - Pass ``True`` to use the default pool (initial size is 4, see below)
      - Pass an integer to use the default pool
        after growing tit to a minimum size
    """
    if not pool:
        return await _run_with_timeout_async_process(
            timeout, fn, args, kwargs, debug, interval)
    if pool is True or isinstance(pool, int):
        minsize = 0 if pool is True else pool
        pool = run_with_timeout_async.default_pool
        if pool is None:
            pool = run_with_timeout_async.default_pool = SimpleProcessPool(4)
        pool.grow_processes(minsize=minsize)
    return await _run_with_timeout_async_pool(pool, timeout, fn, args, kwargs, debug)

run_with_timeout_async.default_pool = None


def timeout_decor(timeout_sec: int, debug=0):
    def decor(fn):
        @functools.wraps(fn)
        def new_fn(*args, **kwargs):
            return run_with_timeout(  # See explanation below for _PickleWrapper
                timeout_sec, _PickleWrapper(outer_func=new_fn), args, kwargs, debug)
        new_fn._timeout_sec_ = timeout_sec
        new_fn._orig_fn_ = fn
        return new_fn
    return decor


def timeout_decor_async(timeout_sec: int, debug=0, interval=0, pool=False):
    def decor(fn):
        @functools.wraps(fn)
        async def new_fn(*args, **kwargs):
            return await run_with_timeout_async(  # See explanation below for _PickleWrapper
                timeout_sec, _PickleWrapper(outer_func=new_fn), args, kwargs,
                debug, interval, pool)
        new_fn._timeout_sec_ = timeout_sec
        new_fn._orig_fn_ = fn
        return new_fn
    return decor


class _PickleWrapper:
    # timeout_decor will be used as a decorator e.g.
    # @timeout_decor(10)
    # def foo(): ...
    # Let's call the actual inner function `foo:orig` and the wrapped one `foo:new`.
    # So if we try to pickle `foo:orig`, it will try to save_global
    # but fail as it sees that the function doesn't match!
    # So we pass _PickleWrapper, and when it is pickled, it does
    # save_global on `foo:new` which does match the function at global scope.
    # However, when the _PickleWrapper object is called, it simply calls `foo:orig`.
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
