"""V2 linkcheck with code-span exclusion + historical dir skip."""
import re
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[1]
INCLUDE = ["README.md", "CHANGELOG.md", "docs/", "scripts/launch/README.md"]
EXCLUDE_PARTS = {"_internal", ".history", "_archive", "_retired",
                 "v7_10_validation_20260424",  # historical bench archive
                 "upstream"}  # historical roadmap snapshots
LINK_RE = re.compile(r'\[([^\]]+)\]\(([^)#]+?)(#[^)]+)?\)')

def strip_code_spans(text):
    # Remove ```...``` blocks and `...` inline code
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`[^`]+`', '', text)
    return text

def is_local(t): return not t.startswith(("http://", "https://", "mailto:", "ftp://", "#"))

def scan():
    results = defaultdict(list)
    files = []
    for prefix in INCLUDE:
        if prefix.endswith("/"):
            files.extend(REPO.glob(f"{prefix}**/*.md"))
        else:
            p = REPO / prefix
            if p.is_file():
                files.append(p)
    for f in files:
        if any(part in f.parts for part in EXCLUDE_PARTS):
            continue
        text = strip_code_spans(f.read_text(errors="replace"))
        for m in LINK_RE.finditer(text):
            label, target, _anchor = m.groups()
            if not is_local(target):
                continue
            resolved = REPO / target.lstrip("/") if target.startswith("/") else (f.parent / target).resolve()
            if not resolved.exists():
                results[str(f.relative_to(REPO))].append((target, label[:40]))
    return results

r = scan()
total = sum(len(v) for v in r.values())
print(f"Active public docs with broken links: {len(r)} files / {total} links")
for fp, broken in sorted(r.items()):
    print(f"  {fp} ({len(broken)} broken):")
    for t, label in broken[:5]:
        print(f"    → [{label[:30]}]({t})")
    if len(broken) > 5:
        print(f"    ... +{len(broken)-5} more")
