import contextlib
import csv
import inspect
import os
import pprint
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import IO


# NOTE: This is copied from and should be kept up-to-date with
# https://github.com/MarcellPerger1/mini-snapshot/blob/main/mini_snapshot.py
class SnapshotsNotFound(RuntimeError):
    ...


class CantUpdateSnapshots(RuntimeError):
    ...


def _string_is_number(s: str):
    try:
        int(s)
        return True
    except ValueError:
        return False


def _safe_cls_name(o: type):
    try:
        return o.__qualname__
    except AttributeError:
        return o.__name__


def _open_or_create_rw(path: str):
    try:
        return open(path, 'r+')
    except FileNotFoundError:
        return open(path, 'w+')  # Try to create if not exists


_SNAPS_NOT_FOUND_MSG = (
    'Snapshots not found, execute this with PY_SNAPSHOTTEST_UPDATE=1 to write \n'
    'the following as the snapshot for {full_name}:\n'
    '{value}')
_SNAPS_DONT_MATCH_MSG = (
    'Snapshots dont match actual value, execute this with PY_SNAPSHOTTEST_UPDATE=1 \n'
    'to write the following as the snapshot for {full_name}:\n'
    '{value}'
)


class SnapshotTestCase(unittest.TestCase):
    snap_filename: str | None = None
    snaps_dir: Path | None = None
    snap_file: str | None = None
    cls_name: str | None = None

    update_snapshots: bool | None = None

    _files_cache: dict[str, dict[str, str]]
    _queued_changes: dict[str, dict[str, str]]
    _subtest = None

    format_dispatch = {}

    @classmethod
    def _lookup_in_dispatch(cls, t: type):
        if m := cls.format_dispatch.get(t):
            return m
        for sup in t.mro():
            if m := cls.format_dispatch.get(sup):
                cls.format_dispatch[t] = m  # Cache it so we don't need to walk MRO again.
                return m
        return None

    @classmethod
    def format_obj(cls, obj: object) -> str:
        if method := cls._lookup_in_dispatch(type(obj)):
            return method(obj)
        if method := getattr(obj, '_format_snapshot_', None):
            return method()
        if isinstance(obj, str):
            # already string, don't repr it to help with snapshot readability
            return obj
        try:
            return pprint.pformat(obj, width=120)
        except TypeError:
            # TODO: this will never match for classes without __repr__
            #  as the fallback is to <object at 0x1234...> and the address
            #  will (almost) never be the same
            return repr(obj)

    @classmethod
    def setUpClass(cls) -> None:
        cls._find_snapshot_filepath()
        cls.cls_name = cls.cls_name or _safe_cls_name(cls)
        cls._files_cache = {}
        if cls.update_snapshots is None:
            cls.update_snapshots = os.environ.get(
                'PY_SNAPSHOTTEST_UPDATE', '0').lower() in ('1', 'true', 'yes')
        # Store changes, not entire file. This reduces (but doesn't eliminate)
        # chance of race conditions when using threading.
        # Additionally, reduces memory load during the main bit of the tests.
        cls._queued_changes = {}
        super().setUpClass()

    @classmethod
    def _find_snapshot_filepath(cls):
        src_file = Path(inspect.getfile(cls))
        cls.snap_filename = cls.snap_filename or src_file.with_suffix('.txt').name
        cls.snaps_dir = cls.snaps_dir or src_file.parent / '.snapshots'
        cls.snap_file = cls.snap_file or str(cls.snaps_dir / cls.snap_filename)

    def setUp(self) -> None:
        self.method_name = self._testMethodName
        self.next_idx = 0

    @classmethod
    def _read_snapshot_text(cls):
        try:
            with open(cls.snap_file) as f:
                return f.read()
        except FileNotFoundError as orig_err:
            raise SnapshotsNotFound("Snapshot file not found") from orig_err

    @classmethod
    def _read_snapshot_file(cls):
        file = cls.snap_file
        if file in cls._files_cache:
            return cls._files_cache[file]
        try:
            snap_data = parse_snap(cls._read_snapshot_text())
        except SnapshotsNotFound:
            if not cls.update_snapshots:
                raise
            snap_data = {}  # need to have something
        cls._files_cache[file] = snap_data
        return snap_data

    def assertMatchesSnapshot(self, obj: object, name: str | None = None,
                              msg: str | None = None):
        SingleSnapshot(
            self, self._allocate_sub_name(name), self.get_opts()
        ).assert_matches(self.format_obj(obj), msg)

    def get_opts(self) -> 'SnapOptions':
        return SnapOptions(self.update_snapshots, self.longMessage)

    def get_self_name(self):
        parts = [f'{self.cls_name}::{self.method_name}']
        subt = self._subtest
        while subt is not None:
            # This formatting could be done better...
            parts.append('{%s}' % subt._subDescription())
            subt = getattr(subt, '_parent_', None)  # We override self.subTest
        return '+'.join(parts)

    def _allocate_sub_name(self, name: str):
        if name is None:
            v = self.next_idx
            self.next_idx += 1
            return str(v)
        if _string_is_number(name):
            raise ValueError("Custom name can't be a number")
        return name

    def queue_write_snapshot(self, full_name: str, new_value: str):
        self._queued_changes.setdefault(self.snap_file, {})[full_name] = new_value

    def lookup_snapshot(self, full_name: str):
        try:
            return self._read_snapshot_file()[full_name]
        except KeyError:
            raise SnapshotsNotFound(f"Snapshot for {full_name} not found") from None

    @contextlib.contextmanager
    def subTest(self, msg: str = unittest.case._subtest_msg_sentinel, **params):
        parent_st = self._subtest
        old_idx = self.next_idx
        self.next_idx = 0
        try:
            with super().subTest(msg, **params) as st:
                self._subtest._parent_ = parent_st
                yield st
        finally:
            self.next_idx = old_idx

    @classmethod
    def tearDownClass(cls) -> None:
        cls._files_cache.clear()  # Free that huge data structure ASAP (not used in write)
        if cls.update_snapshots:
            cls.write_queued_snapshots()

    @classmethod
    def _make_snaps_dir(cls):
        try:
            cls.snaps_dir.mkdir(exist_ok=True)
        except FileExistsError as e:
            raise CantUpdateSnapshots(
                f"Can't write snapshots ({cls.snaps_dir} is not a directory"
                f" so can't write snapshots to it)") from e

    @classmethod
    def write_queued_snapshots(cls):
        if not cls._queued_changes:
            return
        cls._make_snaps_dir()
        for path, changes in cls._queued_changes.items():
            if not changes:
                continue
            with _open_or_create_rw(path) as f:  # do in one go to reduce chance of racing
                orig = parse_snap(f.read())  # Use most up-to-date value (no cache)
                f.seek(0, os.SEEK_SET)  # go to start
                format_snap(f, cls._apply_file_changes(orig, changes))
                # Remove extra garbage that may be left over and not fully overwritten
                f.truncate()

    @classmethod
    def _apply_file_changes(cls, orig: dict[str, str], new: dict[str, str]):
        return orig | new


@dataclass
class SnapOptions:
    update: bool = False
    long_message: bool = False


class SingleSnapshot:
    def __init__(self, parent: SnapshotTestCase, name: str, opts: SnapOptions):
        self.p = parent
        self.sub_name = name
        self.opts = opts
        self.full_name = f'{self.p.get_self_name()}:{self.sub_name}'

    def assert_matches(self, actual: str, msg: str | None = None):
        try:
            expected = self.p.lookup_snapshot(self.full_name)
        except SnapshotsNotFound:
            if self.opts.update:
                return self.queue_write(actual)
            self._maybe_message_with_value(_SNAPS_NOT_FOUND_MSG, actual)
            raise
        if expected == actual:
            return
        if self.opts.update:
            return self.queue_write(actual)
        self._maybe_message_with_value(_SNAPS_DONT_MATCH_MSG, actual)
        self.p.assertEqual(expected, actual, msg)  # will fail and error

    def queue_write(self, value: str):
        self.p.queue_write_snapshot(self.full_name, value)

    def _maybe_message_with_value(self, msg: str, value: str):
        if self.opts.long_message:
            self._output_msg_with_value(msg, value)

    def _output_msg_with_value(self, msg: str, value: str):
        print(msg.format(full_name=self.full_name, value=value), file=sys.stderr)


def parse_snap(text: str):  # This is a bit overcomplicated - need a better format
    all_lines = text.splitlines()
    index_lines = all_lines[0:2]
    names: list[str]
    csv_output = tuple(csv.reader(index_lines, 'unix'))
    if len(csv_output) != 2:
        raise SnapshotsNotFound("Can't read snapshot file (invalid format)")
    lines_idx_str, names = csv_output
    lines_idx = [int(idx_str) for idx_str in lines_idx_str]
    assert len(set(names)) == len(names)
    try:
        line_name_tup: tuple[tuple[int, str], ...] = tuple(
            zip(lines_idx, names, strict=True))
    except ValueError:
        raise SnapshotsNotFound("Can't read snapshot file (invalid format)")
    name_to_src = {}
    for i, (line_idx, name) in enumerate(line_name_tup):
        start = line_idx
        if i == len(line_name_tup) - 1:
            # last item so go until end
            end = len(all_lines)  # exclusive, including first 2 lines
        else:
            end = line_name_tup[i + 1][0]
        src = '\n'.join(all_lines[start:end])
        name_to_src[name] = src
    return name_to_src


def format_snap(file: IO[str], snap_data: dict[str, str]):
    body_list = []
    line_idx_strs: list[str] = []
    names: list[str] = []
    start_line = 2
    for name, src in snap_data.items():
        n_lines = len(src.splitlines())
        line_idx_strs.append(str(start_line))
        names.append(name)
        start_line += n_lines
        body_list.append(src)
    body = '\n'.join(body_list)
    csv.writer(file, 'unix').writerows([line_idx_strs, names])
    file.write(body)
