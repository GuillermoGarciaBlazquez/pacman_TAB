import subprocess
import time
import os
import shutil
import glob
import argparse
import random

# Nombre del script de pacman y directorios de salida
PACMAN_SCRIPT  = "pacman.py"
RUNS_DIR       = "runs"
WINS_DIR       = os.path.join(RUNS_DIR, "wins")
DATA_DIR       = "pacman_data"
DATA_WINS_DIR  = "pacman_wins"

def main():
    parser = argparse.ArgumentParser(
        description="Lanza N partidas de Pacman con semillas aleatorias en un rango pequeño."
    )
    parser.add_argument(
        "-n", "--numRuns", type=int, default=500,
        help="Número de ejecuciones a realizar"
    )
    parser.add_argument(
        "-a", "--agent", type=str, default="ReflexAgent",
        help="Nombre del agente Pacman a usar (p.ej. ReflexAgent)"
    )
    parser.add_argument(
        "--minSeed", type=int, default=1,
        help="Valor mínimo de la semilla (inclusive)"
    )
    parser.add_argument(
        "--maxSeed", type=int, default=1000,
        help="Valor máximo de la semilla (inclusive)"
    )
    parser.add_argument(
        "-q", "--quietTextGraphics",
        action="store_true",
        help="Generate minimal output and no graphics"
    )
    args = parser.parse_args()

    # Directorios de logs y datos
    os.makedirs(RUNS_DIR,      exist_ok=True)
    os.makedirs(WINS_DIR,      exist_ok=True)
    os.makedirs(DATA_WINS_DIR, exist_ok=True)

    # Computar desde qué índice numerar las partidas ganadas
    existing = glob.glob(os.path.join(DATA_WINS_DIR, "game_*.csv")) + \
               glob.glob(os.path.join(WINS_DIR,      "game_*.txt"))
    win_count = max(
        [int(os.path.basename(f).split("_")[1].split(".")[0]) for f in existing] + [-1]
    ) + 1

    for i in range(args.numRuns):
        # Generar una semilla aleatoria en el rango [minSeed, maxSeed]
        seed = random.randint(args.minSeed, args.maxSeed)
        print(f"\nRun {i+1}/{args.numRuns}: seed={seed}")

        # Sobrescribir seed.py
        with open("seed.py", "w") as f:
            f.write(f"PACMAN_SEED = {seed}\n")

        # Ejecutar pacman.py con el agente indicado
        result = subprocess.run(
            ["python", PACMAN_SCRIPT, "-p", args.agent],
            capture_output=True, text=True
        )
        output = result.stdout

        # Guardar log de la partida
        log_path = os.path.join(RUNS_DIR, f"game_{i}.txt")
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(output)

        # Buscar el CSV recién generado
        csvs = sorted(
            glob.glob(os.path.join(DATA_DIR, "*.csv")),
            key=os.path.getmtime, reverse=True
        )
        latest_csv = csvs[0] if csvs else None

        # Si Pacman ganó, copiar a carpeta de wins
        if "Pacman died" not in output:
            win_log = os.path.join(WINS_DIR, f"game_{win_count}.txt")
            shutil.copy(log_path, win_log)

            if latest_csv:
                win_csv = os.path.join(DATA_WINS_DIR, f"game_{win_count}.csv")
                shutil.copy(latest_csv, win_csv)
                print(f"  ¡GANÓ! Log → {win_log}, CSV → {win_csv}")
            else:
                print(f"  ¡GANÓ! Log → {win_log}, pero no encontré CSV.")

            win_count += 1
        else:
            print("  Perdió.")

        # Pequeña pausa para cambiar la semilla de random
        time.sleep(0.01)

if __name__ == "__main__":
    main()
