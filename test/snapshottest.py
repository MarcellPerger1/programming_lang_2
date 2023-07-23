import csv
import inspect
import os
import pprint
import sys
import unittest
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


def _safe_cls_name(o: type):
    try:
        return o.__qualname__
    except AttributeError:
        return o.__name__


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
        except FileNotFoundError:
            raise SnapshotsNotFound(
                "Snapshot file not found; execute with "
                "PY_SNAPSHOTTEST_UPDATE=1 to update snapshots")

    @classmethod
    def _read_snapshot_file(cls):
        file = cls._snap_file
        if file in cls._files_cache:
            return cls._files_cache[file]
        try:
            contents = cls._read_snapshot_file_text()
            name_to_text = SnapParser(contents).parse_snap()
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

    @classmethod
    def _read_snapshot(cls, full_name: str):
        try:
            return cls._read_snapshot_file()[full_name]
        except KeyError:
            raise SnapshotsNotFound(f"Snapshot for {full_name} not found")

    def assertMatchesSnapshot(self, obj: object, name: str | None = None,
                              msg: str | None = None):
        full_name = self._get_full_name(name)
        actual = format_obj(obj)
        try:
            expected = self._read_snapshot(full_name)
        except SnapshotsNotFound:
            if self.update_snapshots:
                return self.queue_write_snapshot(full_name, actual)
            raise

        if expected == actual:
            return
        if self.update_snapshots:
            return self.queue_write_snapshot(full_name, actual)
        if self.longMessage:
            standard_msg = "Expected (from snapshot file) != Actual"
            self.assertEqual(expected, actual,
                             self._formatMessage(msg, standard_msg))
        else:
            self.assertEqual(expected, actual, msg)

    def _get_full_name(self, name: str | None):
        if name is None:
            name = str(self.next_idx)
            self.next_idx += 1
        else:
            try:
                int(name)
            except ValueError:
                pass
            else:
                raise ValueError("Custom name can't be a number")
        full_name = f'{self.cls_name}::{self.method_name}:{name}'
        return full_name

    def queue_write_snapshot(self, full_name: str, new_value: str):
        print(f'Queueing write for snapshot {full_name}', file=sys.stderr)
        if self._snap_file not in self._queued_snapshot_writes:
            # create the new entry, copied from existing
            self._queued_snapshot_writes[self._snap_file] = self._read_snapshot_file()
        self._queued_snapshot_writes[self._snap_file][full_name] = new_value

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
                    SnapFormatter(snapshots).format_snap(f)
            except FileNotFoundError:
                if not cls._snaps_dir.is_dir():
                    raise CantUpdateSnapshots(
                        f"Can't write snapshots - {cls._snaps_dir} "
                        f"is not is directory so can't write snapshots in it")
                raise


class SnapParser:
    def __init__(self, text: str):
        self.text = text

    def parse_snap(self):
        all_lines = self.text.splitlines()
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


class SnapFormatter:
    def __init__(self, name_to_src: dict[str, str]):
        self.name_to_str = name_to_src

    def format_snap(self, file: IO[str]):
        body_list = []
        line_idx_strs: list[str] = []
        names: list[str] = []
        start_line = 2
        for name, src in self.name_to_str.items():
            n_lines = len(src.splitlines())
            line_idx_strs.append(str(start_line))
            names.append(name)
            start_line += n_lines
            body_list.append(src)
        body = ''.join(body_list)
        csv.writer(file, 'unix').writerows([line_idx_strs, names])
        file.write(body)
