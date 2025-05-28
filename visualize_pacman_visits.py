import csv
import ast
import numpy as np
import matplotlib.pyplot as plt

csv_path = "pacman_data/game_636.csv"

# First, determine map size from the first row
with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    first_row = next(reader)
    map_matrix = ast.literal_eval(first_row["map_matrix"])
    width, height = len(map_matrix[0]), len(map_matrix)
    visit_counts = np.zeros((height, width), dtype=int)

# Now process all rows
with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        map_matrix = ast.literal_eval(row["map_matrix"])
        # Find Pacman's position (value 5)
        for y, row_vals in enumerate(map_matrix):
            for x, val in enumerate(row_vals):
                if val == 5:
                    visit_counts[y, x] += 1

plt.imshow(np.rot90(visit_counts), cmap="hot", interpolation="nearest")
plt.title("Pacman Most Visited Points (Rotated 90°)")
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar(label="Visit count")
plt.show()
