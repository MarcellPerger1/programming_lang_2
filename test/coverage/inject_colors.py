import argparse
import re
from pathlib import Path


def find_ext(root: Path, ext: str):
    """ext is including the dot i.e. '``.html``'"""
    res: Path | None = None
    for sub in root.iterdir():
        if sub.is_file() and sub.suffix == ext:
            assert res is None, f"Multiple candidates for {ext!r} file"
            res = sub
    assert res is not None, f"No candidates for {ext!r} file"
    return res


def append_to_file(filepath: str | Path, content: str):
    with open(filepath, 'a') as f:
        f.write(content)


def readfile(filepath: str | Path):
    with open(filepath, 'r') as f:
        return f.read()


_DETECT_INJECTION_RE = re.compile(r'Copyright \(c\) 202\d Marcell Perger', re.IGNORECASE)


def copy_append_unless_present(src: Path, dest: Path):
    code = readfile(src)
    assert _DETECT_INJECTION_RE.search(code) is not None, \
        "Expected magic detection string to be present in source"
    if _DETECT_INJECTION_RE.search(readfile(dest)) is None:
        append_to_file(dest, '\n' + code)
        return True
    return False


def run(cov_dir: Path):
    assert cov_dir.exists() and cov_dir.is_dir()
    css_dest = find_ext(cov_dir, '.css')
    js_dest = find_ext(cov_dir, '.js')
    dirname = Path(__file__).parent  # a la Node.js __dirname
    css_src = dirname/'inject_me.css'
    js_src = dirname/'inject_me.js'
    did_inject_1 = copy_append_unless_present(css_src, css_dest)
    did_inject_2 = copy_append_unless_present(js_src, js_dest)
    if did_inject_1 and did_inject_2:
        print('Colors/fixes injected successfully')
    elif did_inject_1 or did_inject_2:
        print('Colors/fixes partially injected (was already present in some files)')
    else:
        print('Colors/fixes already present')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description="Add colors to HTML coverage that was generated")
    ap.add_argument('cov_dir', type=Path, default='./htmlcov',
                    help="Path to HTML coverage dir", nargs='?')
    args = ap.parse_args()
    run(args.cov_dir)
