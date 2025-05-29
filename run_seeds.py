import subprocess
import time
import os
import shutil
import glob

START_SEED = 596
NUM_RUNS = 500  # Number of runs
PACMAN_SCRIPT = "pacman.py"
AGENT = "ReflexAgent"
RUNS_DIR = "runs"
WINS_DIR = os.path.join(RUNS_DIR, "wins")
DATA_DIR = "pacman_data"
DATA_WINS_DIR = "pacman_wins"  # Changed: wins now outside pacman_data

os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(WINS_DIR, exist_ok=True)
os.makedirs(DATA_WINS_DIR, exist_ok=True)

# Count existing win files to avoid overwriting
existing_win_csvs = glob.glob(os.path.join(DATA_WINS_DIR, "game_*.csv"))
existing_win_logs = glob.glob(os.path.join(WINS_DIR, "game_*.txt"))
win_count = max(
    [int(os.path.splitext(os.path.basename(f))[0].split("_")[1]) for f in existing_win_csvs + existing_win_logs] + [-1]
) + 1

for i in range(NUM_RUNS):
    seed = START_SEED + i
    print(f"\nRun {i+1}: seed={seed}")
    # Overwrite seed.py
    with open("seed.py", "w") as f:
        f.write(f"PACMAN_SEED = {seed}\n")
    # Run pacman.py with the specified agent
    result = subprocess.run(
        ["python", PACMAN_SCRIPT, "-p", AGENT],
        capture_output=True, text=True
    )
    output = result.stdout
    # Save log file
    log_filename = f"game_{i}.txt"
    log_path = os.path.join(RUNS_DIR, log_filename)
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(output)
    # Find the latest CSV file in pacman_data (assuming new file is created per run)
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")), key=os.path.getmtime, reverse=True)
    latest_csv = csv_files[0] if csv_files else None
    # Check for win
    if "Pacman died" not in output:
        # Use next available win_count for naming
        win_log_filename = f"game_{win_count}.txt"
        win_log_path = os.path.join(WINS_DIR, win_log_filename)
        shutil.copy(log_path, win_log_path)
        if latest_csv:
            win_csv_filename = f"game_{win_count}.csv"
            win_csv_path = os.path.join(DATA_WINS_DIR, win_csv_filename)
            shutil.copy(latest_csv, win_csv_path)
            print(f"Game with seed={seed} WON! Log saved to {win_log_path}, CSV saved to {win_csv_path}")
        else:
            print(f"Game with seed={seed} WON! Log saved to {win_log_path}, but no CSV found.")
        win_count += 1
    else:
        print(f"Game with seed={seed} lost.")
    # time.sleep(1)
