#!/usr/bin/env python
"""Check the Python examples in the documentation against the installed SMF.

Two levels of checking, both operating on every fenced ```python block of the Markdown files you pass in:

* static (default): parse each block and verify that
    - imports from ``Modeling_Tool`` / ``ExcelMaster`` / ``Report`` resolve,
    - calls to SMF classes and functions use parameter names that exist and supply all required arguments,
    - methods called on instances created in the same block exist (best effort),
    - attributes read from pipeline ``*Result`` objects are real dataclass fields.
* ``--run``: additionally execute the blocks of each page, in order, in one shared namespace, each block in a fresh
  temporary working directory, and report exceptions and timeouts.

Usage::

    python tools/check_examples.py README.md docs/*.md docs/*/*.md          # static check
    python tools/check_examples.py --run docs/quickstart.md                  # self-contained page: also execute it
    python tools/check_examples.py --run --prelude prelude.py docs/guides/model.md

A page that continues earlier pages can be executed with ``--prelude FILE``: a Python file that is executed first and
defines the objects the page assumes. A block whose first line is ``# check: skip`` is never executed (use it for
snippets that need credentials, a GUI, or a network), but it is still checked statically. A call that passes `...` as an
argument is treated as elided pseudo-code: it is checked for unknown keywords but not for missing arguments.

Exit status is 1 if any problem is found, 0 otherwise.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import importlib
import inspect
import io
import os
import re
import sys
import tempfile
import traceback
import warnings

warnings.filterwarnings("ignore")

try:  # pragma: no cover - environment dependent
    import Modeling_Tool as M
except ImportError as exc:  # pragma: no cover
    sys.exit(f"Cannot import Modeling_Tool ({exc}). Install SuperModelingFactory first.")

SMF_PACKAGES = ("Modeling_Tool", "ExcelMaster", "Report")
SUBPACKAGES = ("Pipeline", "Feature", "WOE", "UAT", "Eval", "Core", "Model", "Sample", "Explainability")
FENCE = re.compile(r"^(\s*)```(\w+)?\s*$")

# Methods that return the instance itself, so chained calls keep the same class.
RETURNS_SELF = {
    ("PerformanceEvaluator", "add_dataset"),
    ("EvaluationPipeline", "group_by"),
    ("EvaluationPipeline", "subset_by"),
}
# Pipeline / engine -> class of the object its ``run()`` returns.
RUN_RESULT = {
    "RejectInferencePipeline": "RejectInferencePipelineResult",
    "CreditModelPipeline": "CreditModelPipelineResult",
    "ScoreComparisonPipeline": "ScoreComparisonPipelineResult",
    "ScoreConsistencyUATPipeline": "ScoreConsistencyUATPipelineResult",
    "SampleAnalysisPipeline": "SampleAnalysisPipelineResult",
    "MockSamplePipeline": "MockSamplePipelineResult",
    "FeatureValidationPipeline": "FeatureValidationPipelineResult",
    "ProcCompareEngine": "ProcCompareResult",
    "ParallelApplyEngine": "ParallelApplyResult",
}


def python_blocks(path):
    """Yield ``(first_line_number, code)`` for every ```python block, even inside admonitions or tabs."""
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().split("\n")
    i = 0
    while i < len(lines):
        match = FENCE.match(lines[i])
        if match and match.group(2) == "python":
            indent = len(match.group(1))
            start = i + 1
            end = start
            while end < len(lines) and not FENCE.match(lines[end]):
                end += 1
            code = "\n".join(
                line[indent:] if line.startswith(" " * indent) else line.lstrip() for line in lines[start:end]
            )
            yield start + 1, code
            i = end + 1
        else:
            i += 1


def _signature(obj):
    try:
        return inspect.signature(obj)
    except (TypeError, ValueError):
        return None


def _check_call(sig, call, label, problems, base_line, skip_first=False):
    if sig is None:
        return
    params = list(sig.parameters.values())
    if skip_first and params and params[0].name in ("self", "cls"):
        params = params[1:]
    has_var_kw = any(p.kind == p.VAR_KEYWORD for p in params)
    has_var_pos = any(p.kind == p.VAR_POSITIONAL for p in params)
    names = {p.name for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    n_args = len([a for a in call.args if not isinstance(a, ast.Starred)])
    has_star_args = any(isinstance(a, ast.Starred) for a in call.args)
    has_star_kwargs = any(k.arg is None for k in call.keywords)
    line = base_line + call.lineno - 1

    if not has_var_pos and not has_star_args and n_args > len(positional):
        problems.append((line, f"{label}: {n_args} positional arguments given, at most {len(positional)} accepted"))
    positional_names = [p.name for p in positional][:n_args]
    for keyword in call.keywords:
        if keyword.arg is None:
            continue
        if keyword.arg not in names and not has_var_kw:
            problems.append((line, f"{label}: unknown keyword `{keyword.arg}` (valid: {sorted(names)})"))
        elif keyword.arg in positional_names:
            problems.append((line, f"{label}: `{keyword.arg}` given both positionally and by keyword"))
    elided = any(isinstance(a, ast.Constant) and a.value is Ellipsis for a in call.args) or any(
        isinstance(k.value, ast.Constant) and k.value.value is Ellipsis for k in call.keywords
    )
    if not has_star_args and not has_star_kwargs and not elided:
        given = set(positional_names) | {k.arg for k in call.keywords if k.arg}
        for p in params:
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and p.default is p.empty and p.name not in given:
                problems.append((line, f"{label}: missing required argument `{p.name}`"))


def _smf_namespace():
    """Classes and functions importable from Modeling_Tool and its subpackages (for un-imported names)."""
    env = {}

    def collect(module):
        for name in dir(module):
            if name.startswith("_") or name in env:
                continue
            obj = getattr(module, name)
            if inspect.isclass(obj) or inspect.isfunction(obj):
                env[name] = obj

    collect(M)
    for sub in SUBPACKAGES:
        try:
            collect(importlib.import_module(f"Modeling_Tool.{sub}"))
        except Exception:  # optional extras may be missing
            pass
    return env


_BASE_ENV = None


def static_check_block(code, base_line, problems):
    """Return ``'syntax'`` if the block is a fragment that does not parse, else ``'ok'``."""
    global _BASE_ENV
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "syntax"
    if _BASE_ENV is None:
        _BASE_ENV = _smf_namespace()
    env = dict(_BASE_ENV)
    instances = {}  # variable -> class name
    results = {}    # variable -> result class name

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if node.module.split(".")[0] not in SMF_PACKAGES:
                continue
            try:
                module = importlib.import_module(node.module)
            except Exception as exc:
                problems.append((base_line + node.lineno - 1, f"cannot import module {node.module}: {type(exc).__name__}"))
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                if hasattr(module, alias.name):
                    env[alias.asname or alias.name] = getattr(module, alias.name)
                else:
                    problems.append((base_line + node.lineno - 1, f"`{alias.name}` not found in {node.module}"))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in SMF_PACKAGES:
                    try:
                        target = alias.name if alias.asname else alias.name.split(".")[0]
                        env[alias.asname or alias.name.split(".")[0]] = importlib.import_module(target)
                    except Exception:
                        problems.append((base_line + node.lineno - 1, f"cannot import {alias.name}"))

    def class_of(expr):
        """Name of the SMF class that ``expr`` evaluates to an instance of (best effort)."""
        if isinstance(expr, ast.Name):
            return instances.get(expr.id)
        if isinstance(expr, ast.Call):
            func = expr.func
            if isinstance(func, ast.Name) and func.id in env and inspect.isclass(env[func.id]):
                return func.id
            if isinstance(func, ast.Attribute):
                owner = class_of(func.value)
                if owner and (owner, func.attr) in RETURNS_SELF:
                    return owner
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            cls_name = class_of(node.value)
            if cls_name:
                instances[name] = cls_name
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "run":
                owner = class_of(value.func.value)
                if owner in RUN_RESULT:
                    results[name] = RUN_RESULT[owner]

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in env:
                target = env[func.id]
                if callable(target):
                    _check_call(_signature(target), node, func.id, problems, base_line)
            elif isinstance(func, ast.Attribute):
                owner = class_of(func.value)
                if owner:
                    cls = env[owner]
                    if not hasattr(cls, func.attr):
                        problems.append((base_line + node.lineno - 1, f"{owner} has no attribute `{func.attr}`"))
                    else:
                        method = getattr(cls, func.attr)
                        if callable(method):
                            _check_call(_signature(method), node, f"{owner}.{func.attr}", problems, base_line, skip_first=True)
                elif isinstance(func.value, ast.Name) and func.value.id in env and inspect.isclass(env[func.value.id]):
                    cls = env[func.value.id]
                    if not hasattr(cls, func.attr):
                        problems.append((base_line + node.lineno - 1, f"{func.value.id} has no attribute `{func.attr}`"))
                    else:
                        _check_call(_signature(getattr(cls, func.attr)), node, f"{func.value.id}.{func.attr}", problems, base_line)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            result_cls = results.get(node.value.id)
            if result_cls:
                cls = getattr(M, result_cls, None) or env.get(result_cls)
                if cls is None:
                    continue
                fields = getattr(cls, "__dataclass_fields__", {})
                if node.attr not in fields and not hasattr(cls, node.attr):
                    problems.append(
                        (base_line + node.lineno - 1, f"{result_cls} has no field `{node.attr}` (fields: {sorted(fields)})")
                    )
    return "ok"


class _Timeout(Exception):
    pass


def run_page(path, prelude=None, timeout=240):
    """Execute the blocks of one page in order in a shared namespace. Returns a list of ``(line, status, message)``."""
    import signal

    use_alarm = hasattr(signal, "SIGALRM")
    if use_alarm:
        def _on_alarm(signum, frame):
            raise _Timeout()

        signal.signal(signal.SIGALRM, _on_alarm)

    import matplotlib

    matplotlib.use("Agg")

    path = os.path.abspath(path)
    prelude_path = os.path.abspath(prelude) if prelude else None   # resolve before changing directory
    namespace = {"__name__": "__doc_example__"}
    outcomes = []
    original_cwd = os.getcwd()
    workdir = tempfile.mkdtemp(prefix="smf_doc_")
    os.chdir(workdir)
    try:
        if prelude_path:
            with open(prelude_path, encoding="utf-8") as handle:
                exec(compile(handle.read(), prelude_path, "exec"), namespace)
        for line, code in python_blocks(path):
            if code.lstrip().startswith("# check: skip"):
                outcomes.append((line, "SKIP", "marked `# check: skip`"))
                continue
            try:
                compiled = compile(code, f"{os.path.basename(path)}:{line}", "exec")
            except SyntaxError:
                outcomes.append((line, "SKIP", "fragment (does not parse)"))
                continue
            if use_alarm:
                signal.alarm(timeout)
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    exec(compiled, namespace)
                outcomes.append((line, "OK", ""))
            except _Timeout:
                outcomes.append((line, "TIMEOUT", f"more than {timeout}s"))
            except BaseException as exc:  # noqa: BLE001 - report everything a snippet can raise
                frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename.startswith(os.path.basename(path))]
                where = f" (block line {frames[-1].lineno})" if frames else ""
                outcomes.append((line, "EXC", f"{type(exc).__name__}: {str(exc)[:300]}{where}"))
            finally:
                if use_alarm:
                    signal.alarm(0)
    finally:
        os.chdir(original_cwd)
    return outcomes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("files", nargs="+", help="Markdown files to check")
    parser.add_argument("--run", action="store_true", help="also execute the blocks of each page")
    parser.add_argument("--prelude", help="Python file executed before the blocks of each page (with --run)")
    parser.add_argument("--timeout", type=int, default=240, help="seconds allowed per executed block (default 240)")
    args = parser.parse_args(argv)

    problems_total = 0
    blocks_total = 0
    fragments = 0
    for path in args.files:
        for base_line, code in python_blocks(path):
            blocks_total += 1
            problems = []
            if static_check_block(code, base_line, problems) == "syntax":
                fragments += 1
            for line, message in problems:
                problems_total += 1
                print(f"{path}:{line}: {message}")
        if args.run:
            for line, status, message in run_page(path, args.prelude, args.timeout):
                if status in ("EXC", "TIMEOUT"):
                    problems_total += 1
                    print(f"{path}:{line}: {status} {message}")
    print(
        f"{blocks_total} Python blocks checked, {fragments} unparsable fragments, {problems_total} problems",
        file=sys.stderr,
    )
    return 1 if problems_total else 0


if __name__ == "__main__":
    sys.exit(main())
