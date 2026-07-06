# Betting Prediction System

This project is an IPL cricket prediction and betting decision support system. It combines historical match data, IPL 2026 season data, player statistics, and live odds to estimate match probabilities, calculate expected value, and generate a betting dashboard.

The codebase is organized around three main ideas:

1. Build an internal win-probability engine using historical and current-season data.
2. Compare model probabilities against live odds to find positive expected value markets.
3. Publish the final recommendations and tracking data into a HTML dashboard.

## What The Project Does

At a high level, the system:

- Loads historical IPL match records from `dataset/matches.csv`.
- Loads current season and supporting IPL 2026 data from the `dataset/` folder.
- Normalizes team names so old and new naming conventions map to the same team code.
- Computes team strength, recent form, head-to-head record, and venue trends.
- Trains an XGBoost-based model and calibrates its probabilities.
- Combines multiple probability signals into a final ensemble prediction.
- Reads `live_odds.json` and filters only markets with positive expected value.
- Writes prediction output to `predictions_history.json`.
- Generates an interactive `dashboard.html` file for viewing results.

## Repository Layout

### Main Scripts

#### `prediction_engine.py`
The core model-building module.

This file is responsible for:

- Loading all datasets needed by the prediction engine.
- Standardizing team names through `normalize_team()`.
- Building player-based team strength scores with `compute_player_scores()`.
- Computing recent team form with `compute_rolling_form()`.
- Computing 2026 season form with `compute_2026_season_form()`.
- Computing head-to-head results with `compute_h2h()` and `get_h2h_winrate()`.
- Calculating venue averages with `compute_venue_stats()`.
- Building the training dataset with `build_training_data()`.
- Training the calibrated XGBoost model with `train_model()`.
- Producing match predictions with `predict_match()`.
- Exposing `build_engine()` as the main entry point.

In practice, this is the most important file in the project because it contains the reusable prediction logic used by the other scripts.

#### `run_live_match.py`
The live prediction and betting recommendation script.

This script:

- Sets up a specific match context.
- Loads historical data, 2026 results, player data, and live odds.
- Computes team strength, recent form, and ensemble probabilities.
- Defines a market probability map for different betting outcomes.
- Calculates expected value for each market.
- Filters out non-positive EV bets.
- Chooses the best bets within budget and staking limits.
- Saves the result into `predictions_history.json`.
- Calls `generate_dashboard.generate_html()` to build the HTML dashboard.

This is the script to run when you want a fresh prediction session.

#### `generate_dashboard.py`
The dashboard generator.

This file converts prediction data into a styled HTML page. It:

- Builds the visual dashboard layout.
- Shows model win probabilities.
- Shows head-to-head and form summaries.
- Lists recommended bets with their EV, odds, probability, and stake.
- Includes a small tracker where you can add your own placed bets.
- Writes the final output to `dashboard.html`.

The dashboard is self-contained, so opening the HTML file in a browser is enough to view the results.

#### `model_feedback.py`
The post-match feedback and retraining script.

This file is intended to be run after a match is finished. It:

- Captures the actual result of a completed match.
- Appends the new result to `dataset/ipl_2026_all_results.csv`.
- Calls back into `prediction_engine.build_engine()` to retrain the model.
- Prints the updated model accuracy.
- Attempts to read prior bet history and report simple profit/loss analysis.

This is the feedback loop for continuously improving the model using new match outcomes.

#### `predict_dc_gt.py`
Match-specific prediction script for Delhi Capitals vs Gujarat Titans.

This file is a preconfigured version of the live engine for one specific fixture. It includes:

- Fixed team names, venue, and date.
- A budget for that session.
- Feature engineering for DC and GT.
- XGBoost, logistic regression, and ensemble probability estimates.
- Positive-EV market filtering.
- Bet sizing and dashboard data output.

#### `predict_pbks_srh.py`
Match-specific prediction script for Punjab Kings vs Sunrisers Hyderabad.

This script follows the same overall structure as `predict_dc_gt.py`, but it is tuned for a different fixture and budget. It:

- Uses PBKS and SRH as the target teams.
- Computes team form and team strength.
- Estimates probabilities for a fixed set of betting markets.
- Selects bets with a positive expected value.
- Writes the final dashboard data.

#### `clear_history.py`
Utility script for resetting prediction history.

This script deletes `predictions_history.json` if it exists. Use it when you want to clear the stored dashboard history and start a new prediction run from a clean state.

### Dashboard Files

#### `dashboard.html`
The generated HTML dashboard.

This file is produced by `generate_dashboard.py`. It displays:

- Match title and venue.
- Ensemble win probability.
- Model breakdown.
- Head-to-head information.
- Betting recommendations.
- A small manual tracker for bets you enter yourself.

Because it is generated output, it can be regenerated at any time by rerunning the prediction workflow.

#### `live_odds.json`
Input file containing live betting odds.

The scripts compare model probabilities to these odds to compute expected value. The structure is market-based, so each market contains a set of labels and numeric odds.

#### `predictions_history.json`
Stored prediction sessions.

This file keeps the latest or previous generated prediction summaries. It is used by the dashboard logic and can be cleared with `clear_history.py`.

### Data Folder

#### `dataset/`
The data directory contains all inputs used by the model.

Important files include:

- `matches.csv`: historical IPL match results.
- `ipl_2026_all_results.csv`: IPL 2026 match results.
- `ipl_2026_players_enriched.csv`: enriched current player data.
- `All_Player_Stat_Season_wise_2016_to_2025.json`: season-wise player history.
- `ipl.csv`, `matches.csv`, `players.csv`, `seasons.csv`: supporting IPL reference datasets.
- `short_name_to_full_name.json`: team name mapping helper.
- `ipl_db.sql`: SQL representation of IPL data.
- `IPL 2026/`: season-specific files such as deliveries, squads, points table, orange cap, purple cap, and venue data.

#### `dataset/ipl_2026_kaggle_notebook.py`
An analysis notebook/script that explores the 2026 player dataset.

This file focuses on:

- Loading the 2026 player dataset.
- Printing summary statistics.
- Identifying top batsmen and bowlers.
- Producing several visualization images.

It is more of a data exploration and analysis asset than part of the live betting workflow.

## How The Prediction Flow Works

The core workflow is:

1. Load historical and current-season data.
2. Convert full team names to short codes such as `MI`, `RCB`, and `GT`.
3. Compute team strength from player batting and bowling performance.
4. Compute recent form from the last few matches.
5. Compute season form from 2026 results.
6. Compute head-to-head records between the two teams.
7. Train and calibrate an XGBoost model.
8. Blend model outputs into a final ensemble probability.
9. Compare probabilities to live odds.
10. Keep only positive expected value bets.
11. Write the final session data to `predictions_history.json`.
12. Generate the HTML dashboard.

## Key Concepts Used By The Model

### Team Strength
Team strength is built from player-level data. Batting and bowling performance are turned into a single score per player, then summed for each team. In the main engine, this score is further adjusted using current-season orange cap and purple cap statistics.

### Rolling Form
Rolling form measures how many of a team’s recent matches were won. The system typically looks at the last five matches, giving a quick snapshot of current momentum.

### Season Form
Season form measures win rate in IPL 2026 so far. This helps the engine account for how a team has performed in the current season, not just historically.

### Head To Head
Head-to-head records compare how two teams have performed against each other in the past. This is used as a separate signal because some matchups are stylistically favorable regardless of overall strength.

### Venue Stats
Venue averages are used to estimate how the pitch and ground conditions influence scoring. This helps the model adjust for high-scoring or low-scoring venues.

### Expected Value
Expected value is the difference between your estimated probability and the odds being offered. A market is considered attractive when the model probability implies a positive return after comparing with the bookmaker odds.

## Outputs Generated By The System

After a run, the project may create or update:

- `dashboard.html`: the rendered betting dashboard.
- `predictions_history.json`: structured prediction session data.
- `dataset/ipl_2026_all_results.csv`: updated results after feedback runs.
- Image files from the analysis notebook, such as charts for role distribution and top batsmen/bowlers.

## Requirements

The main scripts use the following Python packages:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`

The dataset notebook also uses:

- `matplotlib`
- `seaborn`

## Getting Started

### 1. Install Python Dependencies

If you are using a virtual environment, install the packages above with pip.

### 2. Run A Prediction Session

Use the live match script or one of the match-specific scripts.

Examples:

- `python run_live_match.py`
- `python predict_dc_gt.py`
- `python predict_pbks_srh.py`

### 3. Open The Dashboard

After the prediction script runs, open `dashboard.html` in a browser.

### 4. Record Match Feedback

Once a match is complete, run `model_feedback.py` to append the actual result and retrain the engine.

### 5. Clear Old Sessions If Needed

Run `clear_history.py` if you want to remove previous prediction history before generating a new session.

## Notes And Limitations

- The scripts rely on the CSV and JSON files already present in `dataset/`.
- Some scripts are hardcoded for a specific match, venue, and date.
- The betting logic is intentionally conservative and only keeps markets with positive expected value.
- The project is for prediction and analysis; it does not guarantee profitable betting outcomes.
- Some files are generated outputs and can be overwritten by the next run.

## Suggested Project Structure

If you want to extend this project later, the cleanest split is:

- Keep reusable model logic in `prediction_engine.py`.
- Keep match-specific entry scripts separate.
- Keep HTML generation isolated in `generate_dashboard.py`.
- Keep retraining and feedback logic in `model_feedback.py`.
- Keep raw datasets in `dataset/` and avoid editing them manually unless you are intentionally updating source data.

## License

No explicit license file is present in this workspace. Add one if you plan to share or publish the project.