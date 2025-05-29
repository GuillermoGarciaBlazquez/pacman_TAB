import csv
import ast
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

folder = "pacman_data"  # Папка с CSV файлами
csv_files = glob.glob(os.path.join(folder, "*.csv"))

# Определяем размер карты по первому файлу
for file in csv_files:
    with open(file, newline='') as f:
        reader = csv.DictReader(f)
        first_row = next(reader, None)
        if first_row is not None:
            map_matrix = ast.literal_eval(first_row["map_matrix"])
            width, height = len(map_matrix[0]), len(map_matrix)
            visit_counts = np.zeros((height, width), dtype=int)
            break
else:
    print("No CSV files found.")
    exit()

# Обрабатываем все файлы
for file in csv_files:
    with open(file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            map_matrix = ast.literal_eval(row["map_matrix"])
            for y, row_vals in enumerate(map_matrix):
                for x, val in enumerate(row_vals):
                    if val == 5:
                        visit_counts[y, x] += 1

plt.imshow(np.rot90(visit_counts), cmap="hot", interpolation="nearest")
plt.title("Pacman Most Visited Points (All Games)")
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar(label="Visit count")
plt.show()
