import random

COUNT = 7
KNAPSACK_SIZE = 30
MAX_WEIGHT = 35
MAX_PRISE = 70
FILENAME = "data_small"

with open("data/" + FILENAME, "w") as file:
    file.write(f"{KNAPSACK_SIZE}\n")
    for i in range(COUNT):
        file.write(f"{random.randint(0, MAX_WEIGHT)} {random.randint(0, MAX_PRISE)}\n")