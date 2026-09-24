"""Recompute the per-section word counts shown in the report's table of contents.

The counts are embedded in each ``\\section[...]`` optional argument so they appear in
the ToC but not in the section heading itself. They go stale on every edit, so this
rewrites them in place from the current source.

Body prose only: tables, figures, TikZ pictures, equations and comments are excluded, so
the number reflects what a reader actually reads.

Run:  python scripts/update_word_counts.py [--check]
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

TEX = Path(__file__).resolve().parents[1] / "final_report.tex"
SECTION_RE = re.compile(
    r"\\section(?:\[(?P<opt>[^\]]*)\])?\{(?P<title>[^}]*)\}"
)


def strip_non_prose(body: str) -> str:
    """Remove environments and markup that a reader does not read as prose."""
    # drop whole float/graphic environments
    for env in ("table", "table*", "figure", "figure*", "tikzpicture",
                "tabular", "tabularx", "align", "align*", "equation",
                "equation*", "thebibliography", "minipage"):
        body = re.sub(rf"\\begin\{{{re.escape(env)}\}}.*?\\end\{{{re.escape(env)}\}}",
                      " ", body, flags=re.S)
    body = re.sub(r"(?<!\\)%.*", " ", body)          # comments
    body = re.sub(r"\$[^$]*\$", " x ", body)          # inline maths -> one token
    body = re.sub(r"\\[a-zA-Z@]+\*?", " ", body)      # control sequences
    body = re.sub(r"[{}\[\]&~^_\\]", " ", body)       # leftover markup
    return body


def count_words(body: str) -> int:
    text = strip_non_prose(body)
    return len([w for w in re.split(r"\s+", text) if re.search(r"[A-Za-z]", w)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report drift and exit non-zero instead of rewriting")
    args = ap.parse_args()

    src = TEX.read_text(encoding="utf8")
    # only the body before the appendix carries counts
    body_end = src.find("\\appendix")
    head, tail = (src[:body_end], src[body_end:]) if body_end > 0 else (src, "")

    matches = list(SECTION_RE.finditer(head))
    bounds = [m.start() for m in matches] + [len(head)]

    edits, drift = [], []
    for i, m in enumerate(matches):
        body = head[m.end():bounds[i + 1]]
        n = count_words(body)
        title = m.group("title")
        old = m.group("opt") or ""
        old_n = re.search(r"\((\d[\d,]*) words\)", old)
        old_n = int(old_n.group(1).replace(",", "")) if old_n else None
        new = f"\\section[{title} \\textnormal{{\\small({n:,} words)}}]{{{title}}}"
        if old_n != n:
            drift.append((title, old_n, n))
        edits.append((m.start(), m.end(), new))

    if args.check:
        for t, o, n in drift:
            print(f"  {t}: {o} -> {n}")
        print(f"{len(drift)} section(s) with stale counts")
        return 1 if drift else 0

    for start, end, new in reversed(edits):
        head = head[:start] + new + head[end:]
    TEX.write_text(head + tail, encoding="utf8")

    total = 0
    for m in SECTION_RE.finditer(head):
        got = re.search(r"\((\d[\d,]*) words\)", m.group("opt") or "")
        if got:
            total += int(got.group(1).replace(",", ""))
        print(f"  {m.group('title'):50s} {got.group(1) if got else '?':>7s}")
    print(f"  {'TOTAL (body sections)':50s} {total:>7,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
