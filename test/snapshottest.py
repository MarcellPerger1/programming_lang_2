import csv
import inspect
import os
import pprint
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import IO


class SnapshotsNotFound(RuntimeError):
    ...


class CantUpdateSnapshots(RuntimeError):
    ...


def format_obj(obj: object) -> str:
    if method := getattr(obj, '_format_snapshot_', None):
        return method()
    try:
        return pprint.pformat(obj)
    except TypeError:
        return repr(obj)


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
    _src_file: str
    _snap_file: str
    _snaps_dir: Path
    cls_name: str

    _files_cache: dict[str, dict[str, str]]

    update_snapshots: bool | None = None
    _queued_snapshot_writes: dict[str, dict[str, str]]

    @classmethod
    def setUpClass(cls) -> None:
        cls._find_snapshot_filepath()
        cls.cls_name = _safe_cls_name(cls)
        cls._files_cache = {}
        if cls.update_snapshots is None:
            cls.update_snapshots = os.environ.get(
                'PY_SNAPSHOTTEST_UPDATE', '0').lower() in ('1', 'true', 'yes')
        cls._queued_snapshot_writes = {}
        super().setUpClass()

    @classmethod
    def _find_snapshot_filepath(cls):
        cls._src_file = inspect.getfile(cls)
        src_file = Path(cls._src_file)
        if not cls.snap_filename:
            cls.snap_filename = src_file.with_suffix('.txt').name
        cls._snaps_dir = src_file.parent / '.snapshots'
        cls._snap_file = str(cls._snaps_dir / cls.snap_filename)

    def setUp(self) -> None:
        self.method_name = self._testMethodName
        self.next_idx = 0

    @classmethod
    def _read_snapshot_file_text(cls):
        try:
            with open(cls._snap_file) as f:
                return f.read()
        except FileNotFoundError as orig_err:
            raise SnapshotsNotFound("Snapshot file not found") from orig_err

    @classmethod
    def _read_snapshot_file(cls):
        file = cls._snap_file
        if file in cls._files_cache:
            return cls._files_cache[file]
        try:
            name_to_text = parse_snap(cls._read_snapshot_file_text())
        except SnapshotsNotFound:
            if not cls.update_snapshots:
                raise
            cls.create_snapshot_file()
            name_to_text = {}  # need to have something
        cls._files_cache[file] = name_to_text
        return name_to_text

    @classmethod
    def create_snapshot_file(cls):
        cls._make_snaps_dir()
        Path(cls._snap_file).touch()

    def assertMatchesSnapshot(self, obj: object, name: str | None = None,
                              msg: str | None = None):
        self.curr_idx = None
        v = SingleSnapshot(self, self._allocate_sub_name(name), self.get_opts())
        v.assert_matches(format_obj(obj), msg)
        if v.value_to_write is not None:
            self.queue_write_snapshot(v.full_name, v.value_to_write)

    def get_opts(self) -> 'SnapOptions':
        return SnapOptions(self.update_snapshots, self.longMessage)

    def _allocate_sub_name(self, name: str):
        if name is None:
            v = self.next_idx
            self.next_idx += 1
            return str(v)
        if _string_is_number(name):
            raise ValueError("Custom name can't be a number")
        return name

    def queue_write_snapshot(self, full_name: str, new_value: str):
        print(f'Queueing write for snapshot {full_name}', file=sys.stderr)
        if self._snap_file not in self._queued_snapshot_writes:
            # create the new entry, copied from existing
            self._queued_snapshot_writes[self._snap_file] = self._read_snapshot_file()
        self._queued_snapshot_writes[self._snap_file][full_name] = new_value

    def lookup_snapshot(self, full_name: str):
        try:
            return self._read_snapshot_file()[full_name]
        except KeyError:
            raise SnapshotsNotFound(f"Snapshot for {full_name} not found") from None

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.update_snapshots:
            cls.write_queued_snapshots()

    @classmethod
    def _make_snaps_dir(cls):
        if not cls._snaps_dir.exists():
            cls._snaps_dir.mkdir()

    @classmethod
    def write_queued_snapshots(cls):
        cls._make_snaps_dir()
        for filepath, snapshots in cls._queued_snapshot_writes.items():
            try:
                with open(filepath, 'w') as f:
                    format_snap(f, snapshots)
            except FileNotFoundError:
                if not cls._snaps_dir.is_dir():
                    raise CantUpdateSnapshots(
                        f"Can't write snapshots - {cls._snaps_dir} "
                        f"is not is directory so can't write snapshots in it")
                raise


@dataclass
class SnapOptions:
    update: bool = False
    long_message: bool = False


class SingleSnapshot:
    def __init__(self, parent: SnapshotTestCase, name: str, opts: SnapOptions):
        self.p = parent
        self.sub_name = name
        self.opts = opts
        self.full_name = f'{self.p.cls_name}::{self.p.method_name}:{self.sub_name}'
        self.value_to_write = None

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
        self.value_to_write = value

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
