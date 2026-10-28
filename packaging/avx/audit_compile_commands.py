#!/usr/bin/env python3
"""Check the compiler baseline of an AVX CPU build; no native imports."""
import argparse
import json
from pathlib import Path
import shlex

REQUIRED = frozenset(("-march=x86-64", "-mtune=generic", "-mavx", "-mno-avx2",
                      "-mno-fma", "-mno-fma4", "-mno-avx512f"))


def violations(record):
    args = record.get("arguments") or shlex.split(record.get("command", ""))
    issues = ["missing " + flag for flag in sorted(REQUIRED.difference(args))]
    for flag in args:
        if (flag in ("-mavx2", "-mfma", "-mfma4") or flag.startswith("-mavx512")
                or flag.startswith("-march=") and flag != "-march=x86-64"):
            issues.append("unsupported baseline flag " + flag)
    return issues


def audit(records):
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty compile_commands.json list")
    return [dict(file=record.get("file", "unknown"), issues=issues)
            for record in records if (issues := violations(record))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commands", type=Path)
    args = parser.parse_args()
    records = json.loads(args.commands.read_text())
    failures = audit(records)
    print(json.dumps(dict(commands=len(records), failures=failures), indent=2))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
