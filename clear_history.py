import os

HISTORY_FILE = "predictions_history.json"

if os.path.exists(HISTORY_FILE):
    os.remove(HISTORY_FILE)
    print("✅ Prediction history successfully deleted!")
else:
    print("⚠️ History is already empty.")

print("\n(Note: Your dashboard will look empty until you generate a new prediction using 'python run_live_match.py')")
