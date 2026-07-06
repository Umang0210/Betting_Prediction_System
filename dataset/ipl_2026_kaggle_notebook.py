# ============================================================
# IPL 2026 Complete Team & Player Dataset
# Role Classification and Performance Analysis
# ============================================================
# Kaggle Notebook | Beginner to Intermediate Level
# Author: [Your Name]
# Dataset: IPL 2026 Synthetic Player Stats (All 10 Teams)
# ============================================================

# ────────────────────────────────────────────────────────────
# SECTION 1: Import Libraries
# ────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Set a clean visual style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.family"] = "DejaVu Sans"

print("✅ Libraries loaded successfully!")


# ────────────────────────────────────────────────────────────
# SECTION 2: Load Dataset
# ────────────────────────────────────────────────────────────

df = pd.read_csv("ipl_2026_players.csv")

print(f"\n📊 Dataset Shape: {df.shape[0]} players × {df.shape[1]} columns")
print("\n🔍 First 10 Rows:")
print(df.head(10).to_string(index=False))


# ────────────────────────────────────────────────────────────
# SECTION 3: Basic Info & Exploration
# ────────────────────────────────────────────────────────────

print("\n📋 Column Info:")
print(df.dtypes)

print("\n📈 Summary Statistics:")
print(df.describe().round(2))

print("\n🏏 Teams in Dataset:", df["Team"].unique().tolist())
print("👥 Total Players per Team:")
print(df["Team"].value_counts().to_string())

print("\n🎭 Role Distribution:")
print(df["Role"].value_counts().to_string())

print("\n🌍 Nationality Distribution:")
print(df["Nationality"].value_counts().to_string())

print("\n🔎 Missing Values:")
print(df.isnull().sum())


# ────────────────────────────────────────────────────────────
# SECTION 4: Top 10 Batsmen by Runs
# ────────────────────────────────────────────────────────────

# Filter players who primarily bat (Batters + Wicketkeepers + All-Rounders)
top_batsmen = (
    df[df["Runs"] > 0]
    .sort_values("Runs", ascending=False)
    .head(10)[["Player_Name", "Team", "Role", "Runs", "Strike_Rate", "Batting_Average"]]
    .reset_index(drop=True)
)
top_batsmen.index += 1  # Start rank from 1

print("\n🏆 Top 10 Batsmen (by Runs):")
print(top_batsmen.to_string())


# ────────────────────────────────────────────────────────────
# SECTION 5: Top 10 Bowlers by Wickets
# ────────────────────────────────────────────────────────────

top_bowlers = (
    df[df["Wickets"] > 0]
    .sort_values("Wickets", ascending=False)
    .head(10)[["Player_Name", "Team", "Role", "Wickets", "Bowling_Economy"]]
    .reset_index(drop=True)
)
top_bowlers.index += 1

print("\n🎯 Top 10 Bowlers (by Wickets):")
print(top_bowlers.to_string())


# ────────────────────────────────────────────────────────────
# SECTION 6: Team-wise Total Runs Analysis
# ────────────────────────────────────────────────────────────

team_runs = df.groupby("Team")["Runs"].sum().sort_values(ascending=False).reset_index()
team_runs.columns = ["Team", "Total_Runs"]
team_runs["Rank"] = range(1, len(team_runs) + 1)

print("\n🏟️ Team-wise Total Runs:")
print(team_runs.to_string(index=False))


# ────────────────────────────────────────────────────────────
# SECTION 7: Best All-Rounders (Runs + Wickets Combined Score)
# ────────────────────────────────────────────────────────────

# Composite score: normalize runs (max=680) + wickets*10 bonus
df["AllRounder_Score"] = (df["Runs"] / 680 * 100).round(2) + (df["Wickets"] * 4)

best_ar = (
    df[df["Role"] == "All-Rounder"]
    .sort_values("AllRounder_Score", ascending=False)
    .head(10)[["Player_Name", "Team", "Runs", "Wickets", "AllRounder_Score"]]
    .reset_index(drop=True)
)
best_ar.index += 1

print("\n⭐ Best All-Rounders (Composite Score = Runs + Wickets bonus):")
print(best_ar.to_string())


# ────────────────────────────────────────────────────────────
# SECTION 8: VISUALIZATIONS
# ────────────────────────────────────────────────────────────

# ── 8.1 Role Distribution (Pie + Bar) ──────────────────────

role_counts = df["Role"].value_counts()
colors_pie = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("IPL 2026 — Player Role Distribution", fontsize=15, fontweight="bold", y=1.01)

# Pie chart
axes[0].pie(
    role_counts,
    labels=role_counts.index,
    autopct="%1.1f%%",
    colors=colors_pie,
    startangle=140,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
axes[0].set_title("Proportion by Role", fontsize=12)

# Bar chart
sns.barplot(
    x=role_counts.index, y=role_counts.values,
    palette=colors_pie, ax=axes[1], edgecolor="white"
)
axes[1].set_title("Count by Role", fontsize=12)
axes[1].set_xlabel("Role")
axes[1].set_ylabel("Number of Players")
for bar in axes[1].patches:
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        int(bar.get_height()),
        ha="center", va="bottom", fontsize=11, fontweight="bold"
    )

plt.tight_layout()
plt.savefig("role_distribution.png", bbox_inches="tight")
plt.show()
print("✅ Plot saved: role_distribution.png")


# ── 8.2 Top 10 Batsmen Chart ───────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))
colors_bat = sns.color_palette("Blues_r", n_colors=10)
bars = ax.barh(
    top_batsmen["Player_Name"] + " (" + top_batsmen["Team"] + ")",
    top_batsmen["Runs"],
    color=colors_bat, edgecolor="white"
)
ax.invert_yaxis()
ax.set_title("🏏 Top 10 Batsmen — IPL 2026", fontsize=14, fontweight="bold")
ax.set_xlabel("Total Runs", fontsize=12)
ax.set_ylabel("Player (Team)", fontsize=12)

for bar, val in zip(bars, top_batsmen["Runs"]):
    ax.text(bar.get_width() + 4, bar.get_y() + bar.get_height() / 2,
            f"{val}", va="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("top10_batsmen.png", bbox_inches="tight")
plt.show()
print("✅ Plot saved: top10_batsmen.png")


# ── 8.3 Top 10 Bowlers Chart ───────────────────────────────

top_bowlers_plot = (
    df[df["Wickets"] > 0]
    .sort_values("Wickets", ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(12, 6))
colors_bowl = sns.color_palette("Reds_r", n_colors=10)
bars = ax.barh(
    top_bowlers_plot["Player_Name"] + " (" + top_bowlers_plot["Team"] + ")",
    top_bowlers_plot["Wickets"],
    color=colors_bowl, edgecolor="white"
)
ax.invert_yaxis()
ax.set_title("🎯 Top 10 Bowlers — IPL 2026", fontsize=14, fontweight="bold")
ax.set_xlabel("Total Wickets", fontsize=12)
ax.set_ylabel("Player (Team)", fontsize=12)

for bar, val in zip(bars, top_bowlers_plot["Wickets"]):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            f"{val}", va="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("top10_bowlers.png", bbox_inches="tight")
plt.show()
print("✅ Plot saved: top10_bowlers.png")


# ── 8.4 Team Performance Comparison ────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("IPL 2026 — Team Performance Comparison", fontsize=15, fontweight="bold")

palette_team = sns.color_palette("tab10", n_colors=10)

# Total Runs per team
sns.barplot(
    data=team_runs, x="Team", y="Total_Runs",
    palette=palette_team, ax=axes[0], edgecolor="white"
)
axes[0].set_title("Total Runs by Team", fontsize=13)
axes[0].set_xlabel("Team")
axes[0].set_ylabel("Total Runs")
axes[0].tick_params(axis="x", rotation=45)
for bar in axes[0].patches:
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 10,
        int(bar.get_height()),
        ha="center", va="bottom", fontsize=8, fontweight="bold"
    )

# Avg Wickets per team
team_wkts = df.groupby("Team")["Wickets"].sum().sort_values(ascending=False).reset_index()
sns.barplot(
    data=team_wkts, x="Team", y="Wickets",
    palette=palette_team, ax=axes[1], edgecolor="white"
)
axes[1].set_title("Total Wickets by Team", fontsize=13)
axes[1].set_xlabel("Team")
axes[1].set_ylabel("Total Wickets")
axes[1].tick_params(axis="x", rotation=45)
for bar in axes[1].patches:
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.2,
        int(bar.get_height()),
        ha="center", va="bottom", fontsize=8, fontweight="bold"
    )

plt.tight_layout()
plt.savefig("team_performance.png", bbox_inches="tight")
plt.show()
print("✅ Plot saved: team_performance.png")


# ── 8.5 Scatter: Runs vs Wickets (All Players) ─────────────

fig, ax = plt.subplots(figsize=(12, 7))
role_colors = {"Batter": "#4C72B0", "Bowler": "#C44E52",
               "All-Rounder": "#55A868", "Wicketkeeper": "#DD8452"}

for role, group in df.groupby("Role"):
    ax.scatter(
        group["Runs"], group["Wickets"],
        label=role, color=role_colors[role],
        alpha=0.75, s=70, edgecolors="white", linewidths=0.5
    )

ax.set_title("Runs vs Wickets — All Players (IPL 2026)", fontsize=14, fontweight="bold")
ax.set_xlabel("Runs Scored", fontsize=12)
ax.set_ylabel("Wickets Taken", fontsize=12)
ax.legend(title="Role", fontsize=10)
plt.tight_layout()
plt.savefig("runs_vs_wickets.png", bbox_inches="tight")
plt.show()
print("✅ Plot saved: runs_vs_wickets.png")


# ── 8.6 Strike Rate Distribution by Role ───────────────────

fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(
    data=df, x="Role", y="Strike_Rate",
    palette=list(role_colors.values()), ax=ax,
    linewidth=1.2
)
ax.set_title("Strike Rate Distribution by Role — IPL 2026", fontsize=14, fontweight="bold")
ax.set_xlabel("Role", fontsize=12)
ax.set_ylabel("Strike Rate", fontsize=12)
plt.tight_layout()
plt.savefig("strike_rate_by_role.png", bbox_inches="tight")
plt.show()
print("✅ Plot saved: strike_rate_by_role.png")


# ── 8.7 Nationality Breakdown ──────────────────────────────

nat_counts = df["Nationality"].value_counts().head(10)
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(
    x=nat_counts.values, y=nat_counts.index,
    palette="viridis", ax=ax, edgecolor="white"
)
ax.set_title("Top Nationalities in IPL 2026", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Players")
ax.set_ylabel("Nationality")
for bar in ax.patches:
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            int(bar.get_width()), va="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("nationality_breakdown.png", bbox_inches="tight")
plt.show()
print("✅ Plot saved: nationality_breakdown.png")


# ────────────────────────────────────────────────────────────
# SECTION 9: DATA INSIGHTS
# ────────────────────────────────────────────────────────────

print("\n" + "═" * 60)
print("📌 KEY INSIGHTS — IPL 2026 DATASET")
print("═" * 60)

# Insight 1: Best batting team
best_bat_team = team_runs.iloc[0]
print(f"\n💡 Insight 1 — Strongest Batting Team:")
print(f"   {best_bat_team['Team']} leads all teams with {best_bat_team['Total_Runs']} total runs.")

# Insight 2: Most dominant bowler
top_bowler = df.sort_values("Wickets", ascending=False).iloc[0]
print(f"\n💡 Insight 2 — Most Dominant Bowler:")
print(f"   {top_bowler['Player_Name']} ({top_bowler['Team']}) tops the wickets chart "
      f"with {top_bowler['Wickets']} wickets (Economy: {top_bowler['Bowling_Economy']}).")

# Insight 3: Highest scorer
top_scorer = df.sort_values("Runs", ascending=False).iloc[0]
print(f"\n💡 Insight 3 — Highest Run-Scorer:")
print(f"   {top_scorer['Player_Name']} ({top_scorer['Team']}) scored "
      f"{top_scorer['Runs']} runs with a SR of {top_scorer['Strike_Rate']}.")

# Insight 4: Best economy bowler (min 8 wickets)
econ_best = df[df["Wickets"] >= 8].sort_values("Bowling_Economy").iloc[0]
print(f"\n💡 Insight 4 — Best Economy Bowler (min 8 wickets):")
print(f"   {econ_best['Player_Name']} ({econ_best['Team']}) — "
      f"Economy: {econ_best['Bowling_Economy']} with {econ_best['Wickets']} wickets.")

# Insight 5: Best all-rounder
top_ar = best_ar.iloc[0]
print(f"\n💡 Insight 5 — Best All-Rounder:")
print(f"   {top_ar['Player_Name']} ({top_ar['Team']}) — "
      f"{top_ar['Runs']} runs + {top_ar['Wickets']} wickets (Score: {top_ar['AllRounder_Score']}).")

# Insight 6: India dominance
indian_count = df[df["Nationality"] == "Indian"].shape[0]
total = df.shape[0]
print(f"\n💡 Insight 6 — Indian Player Dominance:")
print(f"   {indian_count} out of {total} players ({round(indian_count/total*100,1)}%) are Indian nationals.")

# Insight 7: Role averages
print(f"\n💡 Insight 7 — Average Stats by Role:")
role_avg = df.groupby("Role")[["Runs", "Wickets", "Strike_Rate", "Bowling_Economy"]].mean().round(2)
print(role_avg.to_string())

# Insight 8: Highest Strike Rate Batter
best_sr = df[df["Role"] == "Batter"].sort_values("Strike_Rate", ascending=False).iloc[0]
print(f"\n💡 Insight 8 — Highest Strike Rate (Batters):")
print(f"   {best_sr['Player_Name']} ({best_sr['Team']}) — SR: {best_sr['Strike_Rate']}")

print("\n" + "═" * 60)


# ────────────────────────────────────────────────────────────
# SECTION 10: PREDICT BEST PLAYING XI
# ────────────────────────────────────────────────────────────

print("\n🏆 BEST PLAYING XI — IPL 2026 ALL-STAR TEAM")
print("═" * 60)
print("Selection Logic:")
print("  • 1 Wicketkeeper  → highest Runs among WKs")
print("  • 4 Batters       → top 4 by Runs (non-WK)")
print("  • 3 All-Rounders  → top 3 by AllRounder_Score")
print("  • 3 Bowlers       → top 3 by Wickets (lowest Economy tiebreak)")
print("═" * 60)

xi = []

# 1 Wicketkeeper (best by runs)
wk = df[df["Role"] == "Wicketkeeper"].sort_values("Runs", ascending=False).head(1)
xi.append(("Wicketkeeper", wk.iloc[0]["Player_Name"], wk.iloc[0]["Team"],
           f"{wk.iloc[0]['Runs']} runs"))

# 4 Pure Batters
batters = df[df["Role"] == "Batter"].sort_values("Runs", ascending=False).head(4)
for _, row in batters.iterrows():
    xi.append(("Batter", row["Player_Name"], row["Team"], f"{row['Runs']} runs"))

# 3 All-Rounders
ars = df[df["Role"] == "All-Rounder"].sort_values("AllRounder_Score", ascending=False).head(3)
for _, row in ars.iterrows():
    xi.append(("All-Rounder", row["Player_Name"], row["Team"],
               f"{row['Runs']} runs, {row['Wickets']} wkts"))

# 3 Bowlers (best wickets, lowest economy as tiebreak)
bowlers = (df[df["Role"] == "Bowler"]
           .sort_values(["Wickets", "Bowling_Economy"], ascending=[False, True])
           .head(3))
for _, row in bowlers.iterrows():
    xi.append(("Bowler", row["Player_Name"], row["Team"],
               f"{row['Wickets']} wkts, Econ {row['Bowling_Economy']}"))

# Print XI
print(f"\n{'No.':<4} {'Role':<14} {'Player':<28} {'Team':<6} {'Stats'}")
print("-" * 75)
for i, (role, name, team, stats) in enumerate(xi, 1):
    print(f"{i:<4} {role:<14} {name:<28} {team:<6} {stats}")

print("-" * 75)
print(f"\n✅ Best Playing XI selected from {df.shape[0]} IPL 2026 players!")


# ────────────────────────────────────────────────────────────
# SECTION 11: BONUS — Team-wise Best XI Export
# ────────────────────────────────────────────────────────────

print("\n📦 BONUS — Each Team's Best 5 Players by Overall Score")
print("═" * 60)

df["Overall_Score"] = (df["Runs"] / df["Runs"].max() * 60 +
                       df["Wickets"] / df["Wickets"].max() * 40).round(2)

for team in sorted(df["Team"].unique()):
    team_best = (df[df["Team"] == team]
                 .sort_values("Overall_Score", ascending=False)
                 .head(5)[["Player_Name", "Role", "Runs", "Wickets", "Overall_Score"]])
    print(f"\n🏏 {team}")
    print(team_best.to_string(index=False))

# Save final enriched dataset
df.to_csv("ipl_2026_players_enriched.csv", index=False)
print("\n\n✅ Enriched dataset saved as: ipl_2026_players_enriched.csv")
print("✅ Notebook complete! All plots and insights generated successfully.")
