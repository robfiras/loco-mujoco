"""
Check that the working copy of Python files differs from a git revision only in docstrings and comments.

Used when improving docstrings, to guarantee that no behaviour changed:

    python docs/_scripts/check_docstring_only.py [--rev HEAD] [paths ...]

Without paths, every modified ``.py`` file under ``loco_mujoco/`` is checked. Exits non-zero if the abstract
syntax tree of a file differs from the revision after all docstrings are removed.
"""

import argparse
import ast
import subprocess
import sys


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree


def _normalized(source: str) -> str:
    return ast.dump(_strip_docstrings(ast.parse(source)), include_attributes=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rev", default="HEAD")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    paths = args.paths
    if not paths:
        out = subprocess.run(["git", "diff", "--name-only", args.rev, "--", "loco_mujoco"],
                             capture_output=True, text=True, check=True).stdout
        paths = [p for p in out.splitlines() if p.endswith(".py")]

    failed = []
    for path in paths:
        old = subprocess.run(["git", "show", f"{args.rev}:{path}"], capture_output=True, text=True)
        if old.returncode != 0:
            print(f"new file (not checked): {path}")
            continue
        with open(path) as f:
            new_source = f.read()
        try:
            same = _normalized(old.stdout) == _normalized(new_source)
        except SyntaxError as e:
            print(f"SYNTAX ERROR  {path}: {e}")
            failed.append(path)
            continue
        print(f"{'ok           ' if same else 'CODE CHANGED '} {path}")
        if not same:
            failed.append(path)

    if failed:
        print(f"\n{len(failed)} file(s) differ in more than docstrings.")
        return 1
    print(f"\n{len(paths)} file(s) checked: only docstrings and comments changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
