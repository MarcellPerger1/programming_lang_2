import functools
import multiprocessing as mp
import time


class TestTimeout(Exception):
    ...


def _worker(fn, dest, args, kwargs, debug):
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
    p = mp.Process(target=_worker, args=(fn, dest, args, kwargs, debug), daemon=True)
    p.start()
    if debug >= 1:
        print(f'Worker process running on pid={p.pid}.')
    p.join(timeout_sec)
    if p.is_alive():
        p.kill()  # Die!
        raise TestTimeout(f"Function took too long (exceeded timeout of {timeout_sec}s)")
    if p.exitcode != 0:
        raise mp.ProcessError("Error in process") from dest.get()
    return dest.get()


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


class _PickleWrapper:
    def __init__(self, outer_func):
        """We store the outer func as that is the one that pickle
        will get when it does a load_global"""
        self._outer_func_ = outer_func

    @property
    def func(self):
        func = self._outer_func_
        while getattr(func, '_orig_fn_', None) is not None:
            func = func._orig_fn_
        return func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
