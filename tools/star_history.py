"""Render a star-history chart from this repository's own stargazer data.

GitHub restricted the stargazers timestamp API on 2026-06-30 to a repository's own admins and
collaborators, which broke star-history.com and starchart.cc for everyone reading a public
README. Their suggested fix is to embed an access token in the chart URL; this script exists so
that we do not have to put a token in a public file. A workflow runs it with the repository's own
GITHUB_TOKEN and commits the resulting SVG, so the README points at a static file we control.

    GITHUB_TOKEN=... python tools/star_history.py HDSim-AI/hdsim docs/assets/star-history.svg

No third-party dependencies: urllib for the API, hand-written SVG out.
"""
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request
from datetime import datetime

W, H = 800, 350
# PAD_B carries the date row and the legend below it; too small and the two collide
PAD_L, PAD_R, PAD_T, PAD_B = 58, 34, 26, 62
API = "https://api.github.com"

# Kept out of the f-strings below: CSS braces would all need doubling, and the colours are
# picked so the chart reads on GitHub's light and dark backgrounds alike.
STYLE = (
    "<style>"
    ".bg{fill:#ffffff}.gr{stroke:#e6ebe8}.ax{fill:#5b6b78}.ttl{fill:#17212b}"
    "@media (prefers-color-scheme:dark){"
    ".bg{fill:#0d1117}.gr{stroke:#21262d}.ax{fill:#8b949e}.ttl{fill:#e6edf3}}"
    "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}"
    "</style>"
)


def fetch_stars(repo, token):
    """Every stargazer's starred_at, oldest first. Needs collaborator access since 2026-06-30."""
    out, page = [], 1
    while True:
        req = urllib.request.Request(
            f"{API}/repos/{repo}/stargazers?per_page=100&page={page}",
            headers={"Accept": "application/vnd.github.star+json",
                     "Authorization": f"Bearer {token}",
                     "X-GitHub-Api-Version": "2022-11-28"})
        try:
            batch = json.load(urllib.request.urlopen(req, timeout=60))
        except urllib.error.HTTPError as e:
            body = e.read()[:300].decode(errors="replace")
            raise SystemExit(f"GitHub API {e.code} for {repo}: {body}")
        if not batch:
            break
        out += [b["starred_at"] for b in batch if "starred_at" in b]
        if len(batch) < 100:
            break
        page += 1
    return sorted(out)


def nice_ticks(hi, want=4):
    """Whole-number y ticks; star counts are integers so fractional gridlines read as wrong."""
    if hi <= want:
        return list(range(hi + 1))
    step = max(1, round(hi / want))
    ticks = list(range(0, hi + step, step))
    return ticks if ticks[-1] >= hi else ticks + [ticks[-1] + step]


def render(series, out_path):
    """series: {label: [iso timestamps]}. Cumulative count against real dates."""
    stamps = [datetime.fromisoformat(t.replace("Z", "+00:00"))
              for ts in series.values() for t in ts]
    if not stamps:
        raise SystemExit("no stargazers to plot")
    t0, t1 = min(stamps), max(stamps)
    # a single day of data would divide by zero and draw a vertical wall
    span = max((t1 - t0).total_seconds(), 86400.0)
    hi = max(len(ts) for ts in series.values())
    ticks = nice_ticks(hi)
    top = ticks[-1] or 1

    def x(dt):
        return PAD_L + (dt - t0).total_seconds() / span * (W - PAD_L - PAD_R)

    def y(n):
        return H - PAD_B - (n / top) * (H - PAD_T - PAD_B)

    # brand green, then two tints that stay distinguishable in both themes
    colors = ["#2f7d5f", "#c2703d", "#4a6fa5"]
    p = [(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
          f'viewBox="0 0 {W} {H}" role="img" aria-label="Star history">'),
         STYLE,
         f'<rect class="bg" width="{W}" height="{H}"/>']

    for t in ticks:
        yy = y(t)
        p.append(f'<line class="gr" x1="{PAD_L}" y1="{yy:.1f}" x2="{W - PAD_R}" '
                 f'y2="{yy:.1f}" stroke-width="1"/>')
        p.append(f'<text class="ax" x="{PAD_L - 10}" y="{yy + 4:.1f}" font-size="12" '
                 f'text-anchor="end">{t}</text>')

    # Repos in one org tend to be starred by the same people on the same day, which puts the
    # step lines exactly on top of each other and hides every series but the last drawn. Nudge
    # each one a couple of pixels apart so all of them stay readable; it is a drawing offset
    # only, the plotted values are untouched.
    n_series = len(series)
    for i, (label, ts) in enumerate(series.items()):
        if not ts:
            continue
        dy = (i - (n_series - 1) / 2) * 2.4
        pts, n = [], 0
        for t in ts:
            dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
            pts.append((x(dt), y(n) + dy))  # step: hold, then rise, so each star is one step
            n += 1
            pts.append((x(dt), y(n) + dy))
        pts.append((x(t1), y(n) + dy))
        d = "M" + " L".join(f"{px:.1f} {py:.1f}" for px, py in pts)
        c = colors[i % len(colors)]
        ex, ey = pts[-1]
        legend_x = PAD_L + i * 210
        p.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2.5" '
                 f'stroke-linejoin="round"/>')
        p.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="3.5" fill="{c}"/>')
        p.append(f'<rect x="{legend_x}" y="{H - 16}" width="10" height="10" rx="2" '
                 f'fill="{c}"/>')
        p.append(f'<text class="ax" x="{legend_x + 16}" y="{H - 7}" '
                 f'font-size="12">{label}</text>')

    for lbl, anchor, xx in ((t0.strftime("%b %d, %Y"), "start", PAD_L),
                            (t1.strftime("%b %d, %Y"), "end", W - PAD_R)):
        p.append(f'<text class="ax" x="{xx}" y="{H - PAD_B + 18}" font-size="11" '
                 f'text-anchor="{anchor}">{lbl}</text>')
    p.append(f'<text class="ttl" x="{PAD_L}" y="16" font-size="13" '
             f'font-weight="600">Star history</text>')
    # No "generated on" stamp: it would change the file on every scheduled run and produce a
    # commit even when nobody starred anything. The x axis already ends at the last star.
    total = sum(len(t) for t in series.values())
    p.append(f'<text class="ax" x="{W - PAD_R}" y="16" font-size="11" '
             f'text-anchor="end">{total} total</text>')
    p.append("</svg>")

    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(p))
    return out


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    repos = sys.argv[1].split(",")
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        raise SystemExit("set GITHUB_TOKEN (needs collaborator access on the repos)")
    series = {r.split("/")[-1]: fetch_stars(r, token) for r in repos}
    for name, ts in series.items():
        print(f"  {name}: {len(ts)} stars")
    out = render(series, sys.argv[2])
    print(f"  wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
