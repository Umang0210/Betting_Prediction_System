"""
prediction_engine.py
Enhanced IPL Match Win Probability Engine
- Uses 2008-2025 historical data + IPL 2026 results
- 8 features: team strength, form, H2H, venue bias, toss effect, season form
- XGBoost + probability calibration
"""

import sys

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

DATA_PATH = "dataset/"
IPL2026_PATH = "dataset/IPL 2026/"


# ─────────────────────────────────────────
# 1. LOAD ALL DATA
# ─────────────────────────────────────────

def load_all_data():
    # Historical matches (2008-2025)
    hist = pd.read_csv(DATA_PATH + "matches.csv")

    # IPL 2026 results
    m26 = pd.read_csv(DATA_PATH + "ipl_2026_all_results.csv")

    # Also load the folder's own matches file
    m26_folder = pd.read_csv(IPL2026_PATH + "matches.csv")

    # IPL 2026 players (current season squads)
    players = pd.read_csv(DATA_PATH + "ipl_2026_players_enriched.csv")

    # IPL 2026 squads
    squads = pd.read_csv(IPL2026_PATH + "squads.csv")

    # Orange cap (top batsmen this season)
    orange = pd.read_csv(IPL2026_PATH + "orange_cap.csv")

    # Purple cap (top bowlers this season)
    purple = pd.read_csv(IPL2026_PATH + "purple_cap.csv")

    # Points table (current standings)
    points = pd.read_csv(IPL2026_PATH + "points_table.csv")

    # Historical player stats JSON
    with open(DATA_PATH + "All_Player_Stat_Season_wise_2016_to_2025.json", encoding='utf-8', errors='replace') as f:
        player_history = json.load(f)

    return hist, m26, players, squads, orange, purple, points, player_history


# ─────────────────────────────────────────
# 2. TEAM NAME NORMALIZER
# ─────────────────────────────────────────

TEAM_ALIASES = {
    # Full -> Short
    "Mumbai Indians": "MI",
    "Chennai Super Kings": "CSK",
    "Royal Challengers Bengaluru": "RCB",
    "Royal Challengers Bangalore": "RCB",
    "Kolkata Knight Riders": "KKR",
    "Delhi Capitals": "DC",
    "Delhi Daredevils": "DC",
    "Punjab Kings": "PBKS",
    "Kings XI Punjab": "PBKS",
    "Rajasthan Royals": "RR",
    "Sunrisers Hyderabad": "SRH",
    "Lucknow Super Giants": "LSG",
    "Gujarat Titans": "GT",
    "Deccan Chargers": "DC_OLD",
    "Rising Pune Supergiant": "RPS",
    "Rising Pune Supergiants": "RPS",
    "Pune Warriors": "PW",
    "Kochi Tuskers Kerala": "KTK",
    "Gujarat Lions": "GL",
}

def normalize_team(name):
    """Return short code for any team name variant."""
    if pd.isna(name):
        return None
    name = str(name).strip()
    return TEAM_ALIASES.get(name, name)


# ─────────────────────────────────────────
# 3. PLAYER SCORE ENGINE (CURRENT SEASON)
# ─────────────────────────────────────────

def compute_player_scores(players, orange, purple):
    """
    Combine enriched player dataset with actual 2026 season stats
    from orange cap (batting) and purple cap (bowling).
    """
    # Base score from enriched data
    def base_score(row):
        batting = row.get("Runs", 0) * 0.5 + row.get("Strike_Rate", 100) * 0.3
        bowling = row.get("Wickets", 0) * 20 - row.get("Bowling_Economy", 7) * 8
        return batting + bowling

    players = players.copy()
    players["base_score"] = players.apply(base_score, axis=1)

    # Build team -> total score dict
    team_col = "Team"
    team_strength = players.groupby(team_col)["base_score"].sum().to_dict()

    # Now boost using actual 2026 season stats from orange/purple cap
    orange_boost = {}
    for _, row in orange.iterrows():
        team = normalize_team(row["Team"])
        boost = row["Runs"] * 0.8 + row["Strike_rate"] * 0.5
        orange_boost[team] = orange_boost.get(team, 0) + boost

    purple_boost = {}
    for _, row in purple.iterrows():
        team = normalize_team(row["Team"])
        boost = row["Wickets"] * 25 - row["Economy_rate"] * 10
        purple_boost[team] = purple_boost.get(team, 0) + boost

    # Merge
    final_strength = {}
    for team_code in ["MI","CSK","RCB","KKR","DC","PBKS","RR","SRH","LSG","GT"]:
        base = team_strength.get(team_code, 3000)
        ob = orange_boost.get(team_code, 0)
        pb = purple_boost.get(team_code, 0)
        final_strength[team_code] = base + ob * 0.3 + pb * 0.3

    return final_strength


# ─────────────────────────────────────────
# 4. RECENT FORM (LAST 5 MATCHES - ROLLING)
# ─────────────────────────────────────────

def compute_rolling_form(all_matches_df, window=5):
    """
    Compute rolling win rate for each team's last N matches.
    Returns dict: team_code -> win_rate (0.0 to 1.0)
    """
    df = all_matches_df.copy()
    # Normalize team codes
    df["t1"] = df["team1"].apply(normalize_team)
    df["t2"] = df["team2"].apply(normalize_team)
    df["winner_code"] = df["winner"].apply(normalize_team)

    # Sort by date
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)
    df = df.dropna(subset=["date_parsed"]).sort_values("date_parsed")

    # Build match list per team
    team_results = {t: [] for t in TEAM_ALIASES.keys() if len(t) <= 4}

    for _, row in df.iterrows():
        t1, t2, w = row["t1"], row["t2"], row["winner_code"]
        if t1 in team_results:
            team_results[t1].append(1 if w == t1 else 0)
        if t2 in team_results:
            team_results[t2].append(1 if w == t2 else 0)

    # Last N matches win rate
    form = {}
    active_teams = ["MI","CSK","RCB","KKR","DC","PBKS","RR","SRH","LSG","GT"]
    for team in active_teams:
        results = team_results.get(team, [])
        recent = results[-window:] if len(results) >= window else results
        form[team] = round(np.mean(recent), 3) if recent else 0.5

    return form


# ─────────────────────────────────────────
# 5. SEASON 2026 WIN RATE
# ─────────────────────────────────────────

def compute_2026_season_form(m26):
    """Win rate in IPL 2026 so far."""
    df = m26.copy()
    df["t1"] = df["team1"].apply(normalize_team)
    df["t2"] = df["team2"].apply(normalize_team)
    df["winner_code"] = df["winner"].apply(normalize_team)

    played = {}
    won = {}
    active_teams = ["MI","CSK","RCB","KKR","DC","PBKS","RR","SRH","LSG","GT"]
    for t in active_teams:
        played[t] = 0
        won[t] = 0

    for _, row in df.iterrows():
        t1, t2, w = row["t1"], row["t2"], row["winner_code"]
        if t1 in played:
            played[t1] += 1
            if w == t1:
                won[t1] += 1
        if t2 in played:
            played[t2] += 1
            if w == t2:
                won[t2] += 1

    season_form = {}
    for t in active_teams:
        season_form[t] = won[t] / played[t] if played[t] > 0 else 0.5

    return season_form


# ─────────────────────────────────────────
# 6. HEAD-TO-HEAD AT VENUE
# ─────────────────────────────────────────

def compute_h2h(hist_df, m26_df):
    """
    Returns nested dict: h2h[team1][team2] = win_rate of team1 vs team2
    Also venue-specific h2h.
    """
    all_df = pd.concat([hist_df, m26_df], ignore_index=True)
    all_df["t1"] = all_df["team1"].apply(normalize_team)
    all_df["t2"] = all_df["team2"].apply(normalize_team)
    all_df["winner_code"] = all_df["winner"].apply(normalize_team)

    h2h_wins = {}
    h2h_total = {}

    for _, row in all_df.iterrows():
        t1, t2, w = row["t1"], row["t2"], row["winner_code"]
        if not t1 or not t2:
            continue
        key = tuple(sorted([t1, t2]))
        h2h_total[key] = h2h_total.get(key, 0) + 1
        if w == t1:
            pair = (t1, t2)
            h2h_wins[pair] = h2h_wins.get(pair, 0) + 1
        elif w == t2:
            pair = (t2, t1)
            h2h_wins[pair] = h2h_wins.get(pair, 0) + 1

    return h2h_wins, h2h_total


def get_h2h_winrate(team1, team2, h2h_wins, h2h_total):
    """Win rate of team1 against team2."""
    key = tuple(sorted([team1, team2]))
    total = h2h_total.get(key, 0)
    if total == 0:
        return 0.5
    wins = h2h_wins.get((team1, team2), 0)
    return wins / total


# ─────────────────────────────────────────
# 7. VENUE AVERAGE SCORE
# ─────────────────────────────────────────

def compute_venue_stats(hist_df, m26_df):
    """Average first innings score at each venue."""
    all_df = pd.concat([hist_df, m26_df], ignore_index=True)
    
    # Normalize venue column name
    venue_col = "venue" if "venue" in all_df.columns else None
    score_col = None
    for col in ["first_innings_score", "first_ings_score"]:
        if col in all_df.columns:
            score_col = col
            break
    
    if not venue_col or not score_col:
        return {}

    all_df[score_col] = pd.to_numeric(all_df[score_col], errors="coerce")
    venue_avg = all_df.groupby(venue_col)[score_col].mean().to_dict()
    return venue_avg


# ─────────────────────────────────────────
# 8. BUILD TRAINING DATASET
# ─────────────────────────────────────────

def build_training_data(hist_df, team_strength, rolling_form, h2h_wins, h2h_total, venue_stats):
    data = []

    for _, row in hist_df.iterrows():
        t1_raw = row.get("team1", "")
        t2_raw = row.get("team2", "")
        winner_raw = row.get("winner", "")
        venue_raw = row.get("venue", "Unknown")
        toss_raw = row.get("toss_winner", "")
        toss_decision = row.get("toss_decision", "bat")

        t1 = normalize_team(t1_raw)
        t2 = normalize_team(t2_raw)
        winner = normalize_team(winner_raw)

        if not t1 or not t2 or not winner:
            continue
        if t1 not in team_strength or t2 not in team_strength:
            continue

        s1 = team_strength[t1]
        s2 = team_strength[t2]
        f1 = rolling_form.get(t1, 0.5)
        f2 = rolling_form.get(t2, 0.5)
        h2h = get_h2h_winrate(t1, t2, h2h_wins, h2h_total)
        toss = 1 if normalize_team(toss_raw) == t1 else 0
        toss_bat = 1 if toss_decision == "bat" else 0
        venue_avg = venue_stats.get(str(venue_raw), 165.0)
        label = 1 if winner == t1 else 0

        data.append([
            s1, s2,
            s1 - s2,
            f1, f2,
            f1 - f2,
            h2h,
            toss,
            toss_bat,
            venue_avg,
            label
        ])

    cols = ["s1","s2","strength_diff","f1","f2","form_diff",
            "h2h","toss","toss_bat","venue_avg","winner"]
    return pd.DataFrame(data, columns=cols)


# ─────────────────────────────────────────
# 9. TRAIN MODEL
# ─────────────────────────────────────────

def train_model(df):
    X = df.drop("winner", axis=1)
    y = df["winner"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        eval_metric="logloss",
        verbosity=0
    )

    # Calibrate for better probability estimates
    model = CalibratedClassifierCV(xgb, method="isotonic", cv=5)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    return model, acc, X.columns.tolist()


# ─────────────────────────────────────────
# 10. PREDICT MATCH
# ─────────────────────────────────────────

def predict_match(team1, team2, venue, team_strength, rolling_form, 
                  season_form, h2h_wins, h2h_total, venue_stats, model):
    """
    Returns (p1, p2, confidence_tier)
    p1 = probability team1 wins
    p2 = probability team2 wins
    """
    t1 = normalize_team(team1) or team1
    t2 = normalize_team(team2) or team2

    s1 = team_strength.get(t1, 3000)
    s2 = team_strength.get(t2, 3000)

    f1 = rolling_form.get(t1, 0.5)
    f2 = rolling_form.get(t2, 0.5)
    
    # Blend rolling form with 2026 season form
    sf1 = season_form.get(t1, 0.5)
    sf2 = season_form.get(t2, 0.5)
    f1 = 0.5 * f1 + 0.5 * sf1
    f2 = 0.5 * f2 + 0.5 * sf2

    h2h = get_h2h_winrate(t1, t2, h2h_wins, h2h_total)

    # Use neutral toss for prediction (we don't know toss result yet)
    toss = 0.5
    toss_bat = 0.5

    venue_avg = venue_stats.get(str(venue), 165.0)

    X = np.array([[s1, s2, s1-s2, f1, f2, f1-f2, h2h, toss, toss_bat, venue_avg]])

    prob = model.predict_proba(X)[0][1]
    p1 = round(float(prob), 4)
    p2 = round(1 - p1, 4)

    # Confidence tier
    diff = abs(p1 - 0.5)
    if diff >= 0.15:
        confidence = "🟢 HIGH"
    elif diff >= 0.07:
        confidence = "🟡 MEDIUM"
    else:
        confidence = "🔴 LOW"

    return p1, p2, confidence


# ─────────────────────────────────────────
# 11. MAIN ENGINE BUILDER
# ─────────────────────────────────────────

def build_engine():
    """Build and return the full prediction engine."""
    print("  📂 Loading datasets...")
    hist, m26, players, squads, orange, purple, points, player_history = load_all_data()

    print("  ⚡ Computing team strengths...")
    team_strength = compute_player_scores(players, orange, purple)

    print("  📈 Computing rolling form (last 5 matches)...")
    # Combine historical + 2026 for form
    hist_mini = hist[["team1","team2","winner","date"]].copy()
    m26_mini = m26[["team1","team2","winner","date"]].copy()
    all_matches = pd.concat([hist_mini, m26_mini], ignore_index=True)
    rolling_form = compute_rolling_form(all_matches, window=5)

    print("  🏆 Computing 2026 season form...")
    season_form = compute_2026_season_form(m26)

    print("  🤝 Computing head-to-head records...")
    h2h_wins, h2h_total = compute_h2h(hist, m26)

    print("  🏟️  Computing venue statistics...")
    venue_stats = compute_venue_stats(hist, m26)

    print("  🧠 Building training dataset...")
    train_df = build_training_data(hist, team_strength, rolling_form, 
                                    h2h_wins, h2h_total, venue_stats)

    print(f"  🎯 Training XGBoost model on {len(train_df)} matches...")
    model, acc, feature_cols = train_model(train_df)
    print(f"  ✅ Model accuracy: {round(acc*100,1)}%")

    return {
        "model": model,
        "team_strength": team_strength,
        "rolling_form": rolling_form,
        "season_form": season_form,
        "h2h_wins": h2h_wins,
        "h2h_total": h2h_total,
        "venue_stats": venue_stats,
        "accuracy": acc,
        "points": points,
    }


if __name__ == "__main__":
    engine = build_engine()
    print("\nTeam strengths:")
    for k, v in sorted(engine["team_strength"].items(), key=lambda x: -x[1]):
        print(f"  {k}: {round(v)}")
    print("\nRolling form (last 5):")
    for k, v in sorted(engine["rolling_form"].items(), key=lambda x: -x[1]):
        print(f"  {k}: {round(v*100)}%")
    print("\n2026 Season form:")
    for k, v in sorted(engine["season_form"].items(), key=lambda x: -x[1]):
        print(f"  {k}: {round(v*100)}%")
