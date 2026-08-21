"""Render the weekly brief as a standalone page for GitHub Pages.

Writes docs/index.html, which the workflow commits back to the repo. The
page is designed for one reader on a phone at 7am on a Friday, so it is
laid out as a teamsheet rather than a dashboard: position rails down the
left, monospaced figures, and the captain marked as a block rather than
a parenthetical.
"""
from __future__ import annotations

import html
from pathlib import Path

from . import config as C

DOCS = C.ROOT / "docs"

CSS = """
:root{
  --paper:#FBFBF9; --ink:#111318; --rule:#DCDBD4; --muted:#6E7480;
  --signal:#1B3BCC; --warn:#B3261E; --shade:#F1F0EC;
}
*{box-sizing:border-box;margin:0;padding:0}
body{
  background:var(--paper); color:var(--ink);
  font:16px/1.5 'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  -webkit-font-smoothing:antialiased; padding:0 0 4rem;
}
.wrap{max-width:46rem;margin:0 auto;padding:0 1.25rem}
.num{font-family:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
     font-variant-numeric:tabular-nums}

/* masthead */
header{border-bottom:3px solid var(--ink);padding:2.5rem 0 1rem;margin-bottom:1.75rem}
h1{font-family:'Archivo Narrow','Arial Narrow',sans-serif;font-weight:700;
   font-size:clamp(3rem,14vw,5.5rem);line-height:.86;letter-spacing:-.02em;
   text-transform:uppercase}
.sub{display:flex;flex-wrap:wrap;gap:.35rem 1.25rem;margin-top:.9rem;
     font-size:.75rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}

h2{font-family:'Archivo Narrow','Arial Narrow',sans-serif;font-weight:700;
   font-size:.8rem;letter-spacing:.16em;text-transform:uppercase;
   padding-bottom:.4rem;border-bottom:1px solid var(--ink);margin:2.5rem 0 0}

/* teamsheet */
.rail{display:grid;grid-template-columns:2.4rem 1fr;gap:0 .9rem;align-items:start}
.rail-label{font-family:'Archivo Narrow','Arial Narrow',sans-serif;font-weight:700;
  font-size:.7rem;letter-spacing:.12em;color:var(--muted);padding-top:.95rem;
  border-right:1px solid var(--rule);height:100%}
.player{display:flex;align-items:baseline;gap:.6rem;padding:.7rem 0;
        border-bottom:1px solid var(--rule)}
.player .nm{font-weight:600;flex:1;min-width:0}
.player .club{font-size:.7rem;letter-spacing:.08em;color:var(--muted);text-transform:uppercase}
.player .fig{font-size:.85rem;color:var(--muted)}
.player .ep{font-weight:600;color:var(--ink);min-width:3.1rem;text-align:right}
.cap .nm::after{content:"CAPTAIN";display:inline-block;margin-left:.55rem;
  background:var(--signal);color:#fff;font-family:'Archivo Narrow',sans-serif;
  font-size:.6rem;letter-spacing:.12em;padding:.12rem .38rem;vertical-align:.12em}
.news{color:var(--warn);font-size:.75rem;padding:0 0 .6rem}

.bench{background:var(--shade);padding:.35rem 1rem;margin-top:.4rem}
.bench .player{border-bottom:1px solid #E3E2DC}
.bench .player:last-child{border-bottom:none}

table{width:100%;border-collapse:collapse;font-size:.88rem;margin-top:.3rem}
th{font-family:'Archivo Narrow',sans-serif;font-weight:700;font-size:.68rem;
   letter-spacing:.11em;text-transform:uppercase;color:var(--muted);
   text-align:left;padding:.6rem .5rem .5rem 0;border-bottom:1px solid var(--rule)}
td{padding:.6rem .5rem .6rem 0;border-bottom:1px solid var(--rule)}
th:last-child,td:last-child{text-align:right;padding-right:0}
.yes{color:var(--signal);font-weight:600}
.no{color:var(--muted)}

.note{font-size:.85rem;color:var(--muted);margin-top:.7rem;max-width:34rem}
footer{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--rule);
       font-size:.75rem;color:var(--muted)}
@media(prefers-reduced-motion:no-preference){
  .player{animation:in .4s ease both}
  @keyframes in{from{opacity:0;transform:translateY(3px)}to{opacity:1;transform:none}}
}
"""


def render(payload: dict, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or DOCS
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "index.html"
    path.write_text(_page(payload), encoding="utf-8")
    return path


def _page(p: dict) -> str:
    gw, last = p["gw"], p["gw"] + p["horizon"] - 1
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GW{gw} — FPL model</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo+Narrow:wght@700&family=Inter:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head><body><div class="wrap">

<header>
  <h1>Gameweek<br>{gw}</h1>
  <div class="sub">
    <span>Horizon GW{gw}&ndash;{last}</span>
    <span class="num">&pound;{p['cost']:.1f}m squad</span>
    <span class="num">{p['projected']} pts projected</span>
    <span>Data {html.escape(str(p['pulled']))}</span>
  </div>
</header>

<h2>Starting XI</h2>
{_teamsheet(p['xi'])}

<h2>Bench</h2>
<div class="bench">{"".join(_player(x) for x in p['bench'])}</div>

{_transfers(p['transfers'])}
{_near(p['near'])}

<footer>
  Generated by the expected-points model. Probabilities, not predictions &mdash;
  roughly 40% of FPL variance is irreducible. Check Friday press conferences
  before the deadline.
</footer>
</div></body></html>
"""


def _teamsheet(players: list[dict]) -> str:
    order = ["GK", "DEF", "MID", "FWD"]
    out = []
    for pos in order:
        group = [x for x in players if x["pos"] == pos]
        if not group:
            continue
        body = "".join(_player(x) for x in group)
        out.append(
            f'<div class="rail"><div class="rail-label">{pos}</div>'
            f'<div>{body}</div></div>'
        )
    return "".join(out)


def _player(x: dict) -> str:
    cls = "player cap" if x.get("captain") else "player"
    news = (f'<div class="news">{html.escape(x["news"])}</div>'
            if x.get("news") else "")
    return (
        f'<div class="{cls}">'
        f'<span class="nm">{html.escape(str(x["name"]))}</span>'
        f'<span class="club">{html.escape(str(x["club"]))}</span>'
        f'<span class="fig num">&pound;{x["price"]:.1f}</span>'
        f'<span class="fig num">{x["own"]}%</span>'
        f'<span class="ep num">{x["ep"]}</span>'
        f'</div>{news}'
    )


def _transfers(rows: list[dict]) -> str:
    if not rows:
        return ""
    body = "".join(
        f'<tr><td>{html.escape(str(r["out"]))}</td>'
        f'<td>{html.escape(str(r["in"]))}</td>'
        f'<td class="num">{r["price_delta"]:+.1f}</td>'
        f'<td class="{"yes" if r["worth_hit"] else "no"}">'
        f'{"worth &minus;4" if r["worth_hit"] else "free only" if r["worth_free_transfer"] else "hold"}</td>'
        f'<td class="num">{r["gain"]:+.2f}</td></tr>'
        for r in rows
    )
    return f"""
<h2>Transfers</h2>
<table><thead><tr><th>Out</th><th>In</th><th>&pound;</th><th>Verdict</th><th>Gain</th></tr></thead>
<tbody>{body}</tbody></table>
<p class="note">Gain is expected points across the horizon. A hit needs to clear
4 points to pay for itself.</p>"""


def _near(rows: list[dict]) -> str:
    if not rows:
        return ""
    body = "".join(
        f'<tr><td>{html.escape(str(r["web_name"]))}</td>'
        f'<td>{html.escape(str(r["pos"]))}</td>'
        f'<td class="num">&pound;{r["price"]:.1f}</td>'
        f'<td class="num">{r["selected_by_percent"]}%</td>'
        f'<td class="num">{r["reduced_cost"]:.2f}</td></tr>'
        for r in rows
    )
    return f"""
<h2>Cost of overriding</h2>
<table><thead><tr><th>Player</th><th>Pos</th><th>&pound;</th><th>Own</th><th>Cost</th></tr></thead>
<tbody>{body}</tbody></table>
<p class="note">What you give up by forcing each player into the squad, after
re-solving. Anything under ~1.0 means the model cannot tell them apart from its
own picks &mdash; that is where your read of the team news is worth more.</p>"""
