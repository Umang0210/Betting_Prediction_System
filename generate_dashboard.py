import json, os

# ── helpers ────────────────────────────────────────────────────────
def tier_cls(rt):
    if "HIGH" in rt:   return "high"
    if "MEDIUM" in rt: return "med"
    return "low"

def conf_cls(c):
    if "HIGH"   in c: return "high"
    if "MEDIUM" in c: return "med"
    return "low"

def generate_html(bets, d, live_odds=None):
    t1, t2 = d["match"].split(" vs ")
    budget = d["budget"]
    staked = d["staked"]
    exp_p  = d["exp_pft"]
    exp_r  = d["exp_roi"]
    max_p  = d["max_pft"]
    max_roi = d["max_roi"]
    pev    = d["positive_ev_count"]
    stats  = d.get("stats", {})
    
    odds_json = json.dumps(live_odds or {})
    
    # Probabilities
    ens_d  = d.get("model_details", {}).get("ensemble", {})
    t1_key = f"{t1.lower()}_prob"
    p1 = ens_d.get(t1_key, d.get("ai_win_prob", {}).get(t1, 50))
    p2 = round(100 - p1, 1)

    h2h_t1 = stats.get("h2h_t1_wins", 0)
    h2h_t2 = stats.get("h2h_t2_wins", 0)
    h2h_tot = stats.get("h2h_total", h2h_t1 + h2h_t2)
    h2h_t1p = round(h2h_t1 / h2h_tot * 100) if h2h_tot else 50
    h2h_t2p = 100 - h2h_t1p

    ai_probs = {f"{b['market']}|{b['label']}": b['our_prob'] for b in bets}
    probs_json = json.dumps(ai_probs)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DeepStake AI v2 | Dashboard</title>
<style>
/* ── Reset & base ── */
*{{margin:0;padding:0;box-sizing:border-box}}
:root{{
  --bg:#0b1120;--card:#131f35;--card2:#1a2a42;--border:rgba(148,163,184,.1);
  --txt:#f1f5f9;--muted:#64748b;--muted2:#94a3b8;
  --blue:#38bdf8;--indigo:#818cf8;--green:#22d3a0;
  --amber:#fbbf24;--red:#f87171;--pink:#f472b6;
  --teal:#2dd4bf;--purple:#c084fc;
  --ev-green:#10b981;
}}
body{{font-family:'Inter',system-ui,-apple-system,sans-serif;
     background:var(--bg);color:var(--txt);min-height:100vh;padding:1.5rem}}
.container{{max-width:1140px;margin:0 auto}}

.hdr{{display:flex;align-items:center;justify-content:space-between;
     padding-bottom:1.25rem;margin-bottom:1.5rem;
     border-bottom:1px solid var(--border)}}
.logo{{display:flex;align-items:center;gap:.6rem}}
.logo-icon{{font-size:1.6rem}}
.logo-text{{font-size:1.3rem;font-weight:800;
           background:linear-gradient(135deg,var(--blue),var(--purple));
           -webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.logo-sub{{font-size:.72rem;color:var(--muted2);margin-top:.1rem}}
.hdr-badge{{background:rgba(16,185,129,.12);border:1px solid rgba(16,185,129,.25);
           color:var(--ev-green);border-radius:20px;padding:.3rem .85rem;font-size:.75rem;font-weight:600}}

.tabs{{display:flex;gap:.6rem;overflow-x:auto;padding-bottom:.75rem;margin-bottom:1.75rem;
       scrollbar-width:none}}
.tab{{background:var(--card);border:1px solid var(--border);color:var(--muted2);
     padding:.45rem 1.1rem;border-radius:20px;cursor:pointer;white-space:nowrap;
     font-size:.82rem;font-weight:500;transition:all .18s;flex-shrink:0}}
.tab.active{{background:var(--blue);color:#051520;font-weight:700;border-color:var(--blue)}}

.view{{display:block}}

.match-bar{{display:flex;align-items:center;justify-content:center;
           flex-wrap:wrap;gap:.6rem;margin-bottom:1.5rem;font-size:.88rem;color:var(--muted2)}}
.match-bar strong{{color:var(--txt);font-weight:700}}
.conf-pill{{padding:.22rem .75rem;border-radius:20px;font-size:.75rem;font-weight:700}}
.conf-pill.low   {{background:rgba(248,113,113,.12);color:var(--red);border:1px solid rgba(248,113,113,.25)}}
.conf-pill.med   {{background:rgba(251,191,36,.12);color:var(--amber);border:1px solid rgba(251,191,36,.25)}}
.conf-pill.high  {{background:rgba(34,211,160,.12);color:var(--green);border:1px solid rgba(34,211,160,.25)}}

.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem}}
@media(max-width:700px){{.two-col{{grid-template-columns:1fr}}}}

.prob-card{{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:1.25rem}}
.prob-card-title{{font-size:.72rem;text-transform:uppercase;letter-spacing:.8px;
                 color:var(--muted2);margin-bottom:.9rem;font-weight:600}}
.prob-teams{{display:flex;justify-content:space-between;margin-bottom:.5rem}}
.prob-team{{font-size:.9rem;font-weight:700}}
.prob-team .pct{{font-size:1.35rem;display:block;margin-top:.1rem}}
.prob-team.rr .pct{{color:var(--pink)}}
.prob-team.mi .pct{{color:var(--blue)}}
.prob-bar{{height:20px;border-radius:6px;overflow:hidden;display:flex;margin-bottom:1rem}}
.pb-rr{{background:linear-gradient(90deg,#e11d48,#f43f5e)}}
.pb-mi{{background:linear-gradient(90deg,#3b82f6,#2563eb)}}
.model-grid{{display:grid;grid-template-columns:1fr 1fr;gap:.5rem}}
.model-pill{{background:rgba(255,255,255,.04);border:1px solid var(--border);
            border-radius:8px;padding:.5rem .75rem}}
.model-pill .mn{{font-size:.7rem;color:var(--muted2);text-transform:uppercase;letter-spacing:.5px}}
.model-pill .mv{{font-size:1rem;font-weight:700;margin-top:.15rem}}
.model-pill .ma{{font-size:.68rem;color:var(--muted);margin-top:.1rem}}

.h2h-card{{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:1.25rem}}
.h2h-card-title{{font-size:.72rem;text-transform:uppercase;letter-spacing:.8px;
                color:var(--muted2);margin-bottom:.9rem;font-weight:600}}
.h2h-vs{{display:flex;align-items:center;justify-content:space-between;margin-bottom:.75rem}}
.h2h-wins{{text-align:center}}
.h2h-wins .wn{{font-size:2rem;font-weight:800}}
.h2h-wins .wl{{font-size:.75rem;color:var(--muted2)}}
.h2h-sep{{font-size:.85rem;color:var(--muted)}}
.h2h-bar-wrap{{margin:.75rem 0}}
.h2h-bar{{height:8px;border-radius:4px;overflow:hidden;display:flex;margin:.35rem 0}}
.hb-rr{{background:#e11d48}}.hb-mi{{background:#3b82f6}}
.h2h-pcts{{display:flex;justify-content:space-between;font-size:.75rem;color:var(--muted2)}}
.stats-strip{{display:grid;grid-template-columns:repeat(3,1fr);gap:.5rem;margin-top:.85rem}}
.sstrip{{background:rgba(255,255,255,.03);border-radius:8px;padding:.5rem .6rem;text-align:center}}
.sstrip .sv{{font-size:.95rem;font-weight:700;color:var(--txt)}}
.sstrip .sl{{font-size:.67rem;color:var(--muted);margin-top:.1rem}}

.kpi-row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.85rem;margin-bottom:1rem}}
.kpi{{background:var(--card);border:1px solid var(--border);border-radius:12px;
     padding:1rem 1rem .85rem;text-align:center}}
.kpi .kl{{font-size:.7rem;text-transform:uppercase;letter-spacing:.6px;color:var(--muted2)}}
.kpi .kv{{font-size:1.65rem;font-weight:800;margin-top:.3rem;line-height:1}}
.kpi .ks{{font-size:.72rem;color:var(--muted);margin-top:.3rem}}

.ev-notice{{background:rgba(16,185,129,.06);border:1px solid rgba(16,185,129,.18);
           border-radius:10px;padding:.8rem 1rem;margin-bottom:1.25rem;
           display:flex;align-items:center;gap:.6rem;font-size:.83rem;color:var(--green)}}
.ev-notice .icon{{font-size:1rem;flex-shrink:0}}

.sec-head{{display:flex;align-items:center;gap:.6rem;margin-bottom:.9rem}}
.sec-head h2{{font-size:1rem;font-weight:700}}
.sec-badge{{background:rgba(56,189,248,.12);color:var(--blue);
           border:1px solid rgba(56,189,248,.25);border-radius:20px;
           padding:.18rem .65rem;font-size:.72rem;font-weight:700}}

.bet-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:.9rem;margin-bottom:1.5rem}}
.bet-card{{background:var(--card);border:1px solid var(--border);
          border-radius:12px;padding:1.2rem;position:relative;overflow:hidden;
          transition:transform .15s,border-color .15s}}
.bet-card:hover{{transform:translateY(-2px);border-color:rgba(56,189,248,.25)}}
.bet-card::before{{content:'';position:absolute;left:0;top:0;bottom:0;
                  width:3px;border-radius:3px 0 0 3px}}
.bet-card.high::before{{background:var(--ev-green)}}
.bet-card.med::before{{background:var(--amber)}}
.bet-card.low::before{{background:var(--muted)}}

.bet-top{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.55rem}}
.bet-mkt{{font-size:.7rem;color:var(--muted2);text-transform:uppercase;
         letter-spacing:.5px;line-height:1.3;max-width:70%}}
.tier-pill{{font-size:.68rem;font-weight:700;padding:.18rem .6rem;border-radius:20px;flex-shrink:0}}
.tier-pill.high{{background:rgba(16,185,129,.15);color:var(--ev-green)}}
.tier-pill.med {{background:rgba(251,191,36,.15);color:var(--amber)}}
.tier-pill.low {{background:rgba(100,116,139,.15);color:var(--muted2)}}

.bet-label{{font-size:1.05rem;font-weight:800;margin-bottom:.85rem;line-height:1.3}}
.bet-metrics{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:.4rem;margin-bottom:.75rem}}
.bm{{background:rgba(255,255,255,.03);border-radius:7px;padding:.45rem .5rem;text-align:center}}
.bm .bml{{font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:.3px}}
.bm .bmv{{font-size:.92rem;font-weight:700;margin-top:.2rem}}
.bmv.ev   {{color:var(--ev-green)}}
.bmv.edge {{color:var(--teal)}}
.bmv.prob {{color:var(--indigo)}}
.bmv.odds {{color:var(--txt)}}

.bprob-bar{{height:5px;border-radius:3px;background:rgba(255,255,255,.08);margin:.45rem 0 .65rem;overflow:hidden}}
.bprob-fill{{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--blue),var(--purple))}}
.bprob-labels{{display:flex;justify-content:space-between;font-size:.68rem;color:var(--muted)}}

.reason{{background:rgba(255,255,255,.025);border-left:2px solid rgba(56,189,248,.35);
        border-radius:0 6px 6px 0;padding:.55rem .7rem;font-size:.78rem;
        color:var(--muted2);line-height:1.45;font-style:italic;margin-bottom:.9rem}}

.stake-row{{display:flex;justify-content:space-between;align-items:center;
           background:rgba(0,0,0,.25);border-radius:8px;padding:.7rem .9rem}}
.stake-left .sl-label{{font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:.4px}}
.stake-left .sl-val{{font-size:1.25rem;font-weight:800;margin-top:.15rem}}
.stake-right{{text-align:right}}
.stake-right .sr-label{{font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:.4px}}
.stake-right .sr-win{{font-size:1.1rem;font-weight:700;color:var(--ev-green);margin-top:.15rem}}
.stake-right .sr-exp{{font-size:.7rem;color:var(--teal);margin-top:.2rem}}

.context-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:.75rem;margin-bottom:1.5rem}}
.ctx-card{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:.9rem 1rem}}
.ctx-card .cc-label{{font-size:.7rem;text-transform:uppercase;letter-spacing:.5px;color:var(--muted2);margin-bottom:.5rem}}
.ctx-bar-wrap{{margin:.4rem 0}}
.ctx-bar{{height:6px;border-radius:3px;background:rgba(255,255,255,.06);overflow:hidden}}
.ctx-bar-fill{{height:100%;border-radius:3px}}
.ctx-val{{font-size:1rem;font-weight:700}}
.ctx-sub{{font-size:.7rem;color:var(--muted);margin-top:.25rem}}

/* ── Tracker Enhancements ── */
.tracker-section{{background:var(--card2);border:1px solid var(--border);border-radius:16px;padding:1.5rem;margin-bottom:2rem}}
.tracker-head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:1.25rem}}
.tracker-stats{{display:flex;gap:1rem}}
.t-stat{{background:rgba(255,255,255,.05);padding:.5rem 1rem;border-radius:10px;text-align:center;min-width:100px}}
.t-stat .sl{{font-size:.65rem;color:var(--muted2);text-transform:uppercase;letter-spacing:.5px}}
.t-stat .sv{{font-size:1.1rem;font-weight:800;margin-top:.15rem}}

.tracker-table{{width:100%;border-collapse:separate;border-spacing:0 .5rem}}
.tracker-table th{{text-align:left;padding:.75rem 1rem;font-size:.7rem;color:var(--muted2);text-transform:uppercase;letter-spacing:.5px}}
.tracker-table td{{padding:1rem;background:var(--card);border:1px solid var(--border);border-left:none;border-right:none}}
.tracker-table td:first-child{{border-left:1px solid var(--border);border-radius:10px 0 0 10px}}
.tracker-table td:last-child{{border-right:1px solid var(--border);border-radius:0 10px 10px 0}}

.inp{{background:rgba(255,255,255,.04);border:1px solid var(--border);color:var(--txt);padding:.5rem .75rem;border-radius:6px;width:100%;font-size:.85rem;outline:none}}
.inp:focus{{border-color:var(--blue)}}
.btn-add{{background:var(--blue);color:#051520;border:none;padding:.6rem 1.25rem;border-radius:8px;font-weight:700;cursor:pointer;font-size:.85rem;display:flex;align-items:center;gap:.4rem}}
.btn-del{{background:rgba(248,113,113,.15);color:var(--red);border:none;padding:.4rem .6rem;border-radius:6px;cursor:pointer}}

/* Rolling Slider / Search UI */
.search-container{{position:relative;width:100%;flex:2}}
.search-results{{
  position:absolute;top:100%;left:0;right:0;
  background:rgba(19,31,53,0.95);
  backdrop-filter:blur(8px);
  border:1px solid var(--border);
  border-radius:10px;max-height:280px;
  overflow-y:auto;z-index:100;display:none;
  margin-top:8px;
  box-shadow:0 12px 24px rgba(0,0,0,0.5), 0 0 0 1px rgba(56,189,248,0.1);
}}
.search-results::-webkit-scrollbar {{ width:6px; }}
.search-results::-webkit-scrollbar-track {{ background:transparent; }}
.search-results::-webkit-scrollbar-thumb {{ background:rgba(56,189,248,0.3); border-radius:10px; }}
.search-results::-webkit-scrollbar-thumb:hover {{ background:var(--blue); }}

.search-item{{padding:.8rem 1.1rem;cursor:pointer;font-size:.85rem;transition:all .15s;border-bottom:1px solid rgba(255,255,255,0.03)}}
.search-item:last-child{{border-bottom:none}}
.search-item:hover{{background:rgba(56,189,248,0.12);color:var(--blue);padding-left:1.3rem}}
.search-item strong{{color:var(--blue);text-shadow:0 0 8px rgba(56,189,248,0.4)}}

.inp:focus{{border-color:var(--blue);box-shadow:0 0 0 2px rgba(56,189,248,0.2);background:rgba(56,189,248,0.03)}}

.footer{{text-align:center;color:var(--muted);font-size:.72rem;
        margin-top:2rem;padding-top:1rem;border-top:1px solid var(--border)}}
</style>
</head>
<body>
<div class="container">
<div class="hdr">
  <div class="logo">
    <span class="logo-icon">🏏</span>
    <div>
      <div class="logo-text">DeepStake AI v2</div>
      <div class="logo-sub">Multi-Model Cricket Prediction Engine</div>
    </div>
  </div>
  <div class="hdr-badge">✅ Positive EV Only</div>
</div>
<div class="tabs" id="tabs">
  <div class="tab active">▶ {t1} vs {t2} · {d.get("date","Today")}</div>
</div>
<div class="match-bar">
  <strong>{t1} vs {t2}</strong> &bull; {d.get("date","Today")} &bull; {d.get("venue","Unknown")}
  <span class="conf-pill {conf_cls(d.get("confidence","LOW"))}">🔴 {d.get("confidence","LOW")} CONFIDENCE</span>
</div>

<div class="two-col">
<div class="prob-card">
  <div class="prob-card-title">🤖 Ensemble Win Probability</div>
  <div class="prob-teams">
    <div class="prob-team rr"><span>{t1}</span><span class="pct">{p1}%</span></div>
    <div class="prob-team mi" style="text-align:right"><span>{t2}</span><span class="pct">{p2}%</span></div>
  </div>
  <div class="prob-bar">
    <div class="pb-rr" style="width:{p1}%"></div>
    <div class="pb-mi" style="width:{p2}%"></div>
  </div>
  <div class="model-grid">
    <div class="model-pill"><div class="mn">XGBoost</div><div class="mv" style="color:var(--blue)">{d.get('model_details',{}).get('xgboost',{}).get(t1_key, "?")}%</div></div>
    <div class="model-pill"><div class="mn">Logistic</div><div class="mv" style="color:var(--teal)">{d.get('model_details',{}).get('logistic',{}).get(t1_key, "?")}%</div></div>
    <div class="model-pill"><div class="mn">Bayesian</div><div class="mv" style="color:var(--purple)">{d.get('model_details',{}).get('bayesian',{}).get(t1_key, "?")}%</div></div>
    <div class="model-pill"><div class="mn">Final</div><div class="mv" style="color:var(--amber)">{p1}%</div></div>
  </div>
</div>
<div class="h2h-card">
  <div class="h2h-card-title">📊 Head-to-Head &amp; Form</div>
  <div class="h2h-vs">
    <div class="h2h-wins" style="color:var(--pink)"><div class="wn">{h2h_t1}</div><div class="wl">{t1} Wins</div></div>
    <div class="h2h-sep">vs<br><span style="font-size:.72rem">{h2h_tot} matches</span></div>
    <div class="h2h-wins" style="color:var(--blue)"><div class="wn">{h2h_t2}</div><div class="wl">{t2} Wins</div></div>
  </div>
  <div class="h2h-bar-wrap">
    <div class="h2h-bar"><div class="hb-rr" style="width:{h2h_t1p}%"></div><div class="hb-mi" style="width:{h2h_t2p}%"></div></div>
  </div>
</div>
</div>

<div class="kpi-row">
  <div class="kpi"><div class="kl">Budget</div><div class="kv">₹{budget:,.0f}</div></div>
  <div class="kpi"><div class="kl">Staked</div><div class="kv">₹{staked:,.0f}</div></div>
  <div class="kpi"><div class="kl">Exp Pft</div><div class="kv" style="color:var(--ev-green)">₹{exp_p:,.0f}</div></div>
  <div class="kpi"><div class="kl">Max Pft</div><div class="kv" style="color:var(--green)">₹{max_p:,.0f}</div></div>
  <div class="kpi"><div class="kl">Win Chance</div><div class="kv" style="color:var(--indigo)">{d.get('win_chance', 0)}%</div></div>
</div>

<div class="sec-head"><h2>🎯 Personal Betting Tracker</h2><span class="sec-badge">Live Performance</span></div>
<div class="tracker-section">
  <div class="tracker-head">
    <div style="font-size:1.1rem;font-weight:700">My Actual Placed Bets</div>
    <div class="tracker-stats">
      <div class="t-stat"><div class="sl">Staked</div><div class="sv" id="t-staked" style="color:var(--amber)">₹0</div></div>
      <div class="t-stat"><div class="sl">Max Pft</div><div class="sv" id="t-max" style="color:var(--green)">₹0</div></div>
      <div class="t-stat"><div class="sl">Exp Pft</div><div class="sv" id="t-exp" style="color:var(--ev-green)">₹0</div></div>
      <div class="t-stat"><div class="sl">Win %</div><div class="sv" id="t-win" style="color:var(--indigo)">0%</div></div>
    </div>
  </div>
  <table class="tracker-table">
    <thead><tr><th>Market</th><th>Label</th><th>Stake</th><th>Odds</th><th>Action</th></tr></thead>
    <tbody id="tracker-body"></tbody>
  </table>
  <div style="display:flex;gap:.5rem;margin-top:1rem">
    <div class="search-container">
        <input type="text" id="in-mkt" class="inp" placeholder="Search Market (e.g. Head to Head)" oninput="filterMarkets()">
        <div id="mkt-results" class="search-results"></div>
    </div>
    <div class="search-container">
        <input type="text" id="in-lbl" class="inp" placeholder="Search Selection" oninput="filterLabels()" onclick="filterLabels()">
        <div id="lbl-results" class="search-results"></div>
    </div>
    <input type="number" id="in-stk" class="inp" style="width:100px" placeholder="Stake">
    <input type="number" step="0.01" id="in-ods" class="inp" style="width:80px" placeholder="Odds">
    <button class="btn-add" onclick="addBet()">Add</button>
  </div>
</div>

<div class="sec-head"><h2>🎯 AI Recommended Bets</h2><span class="sec-badge">{pev} Markets</span></div>
<div class="bet-grid">"""

    for bet in bets:
        tc = tier_cls(bet.get('risk_tier',''))
        reason = bet.get('reason', 'Based on current match dynamics and model projections.')
        html += f"""
<div class="bet-card {tc}">
  <div class="bet-top"><div class="bet-mkt">{bet["market"]}</div><div class="tier-pill {tc}">{bet.get("risk_tier","")}</div></div>
  <div class="bet-label">{bet["label"]}</div>
  <div class="reason">{reason}</div>
  <div class="bet-metrics">
    <div class="bm"><div class="bml">EV</div><div class="bmv ev">{bet.get('ev',0):+.3f}</div></div>
    <div class="bm"><div class="bml">Odds</div><div class="bmv odds">{bet["odds"]}</div></div>
    <div class="bm"><div class="bml">Prob</div><div class="bmv prob">{round(bet.get('our_prob',0)*100)}%</div></div>
  </div>
  <div class="stake-row">
    <div class="stake-left"><div class="sl-label">Stake</div><div class="sl-val">₹{bet["stake"]:.0f}</div></div>
    <div class="stake-right"><div class="sr-label">Return</div><div class="sr-win">₹{bet["ret"]:.0f}</div></div>
  </div>
</div>"""

    html += f"""</div>
<script>
const AI_PROBS = {probs_json};
const LIVE_ODDS = {odds_json};

function filterMarkets() {{
    const val = document.getElementById('in-mkt').value.toLowerCase();
    const res = document.getElementById('mkt-results');
    res.innerHTML = '';
    if(!val) {{ res.style.display = 'none'; return; }}
    
    const matches = Object.keys(LIVE_ODDS).filter(m => m.toLowerCase().includes(val));
    if(matches.length > 0) {{
        matches.forEach(m => {{
            const div = document.createElement('div');
            div.className = 'search-item';
            div.innerHTML = m.replace(new RegExp(val, 'gi'), s => `<strong>${{s}}</strong>`);
            div.onclick = () => selectMarket(m);
            res.appendChild(div);
        }});
        res.style.display = 'block';
    }} else {{
        res.style.display = 'none';
    }}
}}

function selectMarket(m) {{
    document.getElementById('in-mkt').value = m;
    document.getElementById('mkt-results').style.display = 'none';
    document.getElementById('in-lbl').value = '';
    document.getElementById('in-ods').value = '';
    filterLabels(); // Show all labels for this market
}}

function filterLabels() {{
    const mkt = document.getElementById('in-mkt').value;
    const val = document.getElementById('in-lbl').value.toLowerCase();
    const res = document.getElementById('lbl-results');
    res.innerHTML = '';
    
    if(!mkt || !LIVE_ODDS[mkt]) {{
        res.style.display = 'none';
        return;
    }}
    
    const labels = Object.keys(LIVE_ODDS[mkt]).filter(l => l.toLowerCase().includes(val));
    if(labels.length > 0) {{
        labels.forEach(l => {{
            const div = document.createElement('div');
            div.className = 'search-item';
            div.innerHTML = l.replace(new RegExp(val, 'gi'), s => `<strong>${{s}}</strong>`);
            div.onclick = () => selectLabel(l, mkt);
            res.appendChild(div);
        }});
        res.style.display = 'block';
    }} else {{
        res.style.display = 'none';
    }}
}}

function selectLabel(l, m) {{
    document.getElementById('in-lbl').value = l;
    document.getElementById('lbl-results').style.display = 'none';
    if(LIVE_ODDS[m] && LIVE_ODDS[m][l]) {{
        document.getElementById('in-ods').value = LIVE_ODDS[m][l];
    }}
}}

function addBet() {{
    const mkt = document.getElementById('in-mkt').value;
    const lbl = document.getElementById('in-lbl').value;
    const stk = parseFloat(document.getElementById('in-stk').value) || 0;
    const ods = parseFloat(document.getElementById('in-ods').value) || 0;
    if(!mkt || !lbl || !stk || !ods) return alert('Fill fields');
    
    const key = mkt + '|' + lbl;
    const prb = (AI_PROBS[key] ? (AI_PROBS[key]) : 0.5);
    
    const row = document.createElement('tr');
    row.innerHTML = `<td>${{mkt}}</td><td>${{lbl}}</td><td>₹${{stk}}</td><td>${{ods}}</td><td><button class="btn-del" onclick="this.closest('tr').remove(); updateStats()">✖</button></td>`;
    row.dataset.stk = stk; row.dataset.ods = ods; row.dataset.prb = prb;
    document.getElementById('tracker-body').appendChild(row);
    updateStats();
}}

function updateStats() {{
    let s=0, m=0, e=0, p=0, c=0;
    document.querySelectorAll('#tracker-body tr').forEach(r => {{
        const stk=parseFloat(r.dataset.stk), ods=parseFloat(r.dataset.ods), prb=parseFloat(r.dataset.prb);
        s+=stk; m+=(stk*ods-stk); e+=(prb*stk*ods-stk); p+=prb; c++;
    }});
    document.getElementById('t-staked').innerText='₹'+Math.round(s);
    document.getElementById('t-max').innerText='₹'+Math.round(m);
    document.getElementById('t-exp').innerText='₹'+Math.round(e);
    document.getElementById('t-win').innerText=(c>0?Math.round(p/c*100):0)+'%';
}}

// Close search results if clicked outside
document.addEventListener('click', (e) => {{
    if(!e.target.closest('.search-container')) {{
        document.getElementById('mkt-results').style.display = 'none';
        document.getElementById('lbl-results').style.display = 'none';
    }}
}});
</script>
<div class="footer">DeepStake AI v2 &bull; <strong>Not financial advice.</strong></div>
</div>
</body>
</html>"""

    with open("dashboard.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Dashboard updated → dashboard.html")
