"""
generate_data.py
-----------------
Creates a synthetic (but realistic) student placement dataset and saves it as
data/placement_data.csv

Run:
    python src/generate_data.py
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 1200

cgpa = np.round(np.random.normal(7.2, 1.1, N).clip(4.5, 10.0), 2)
backlogs = np.random.choice([0, 0, 0, 1, 1, 2, 3], size=N)
internships = np.random.choice([0, 1, 1, 2, 2, 3], size=N)
projects = np.random.choice([0, 1, 2, 2, 3, 4], size=N)
coding_score = np.round(np.random.normal(65, 18, N).clip(0, 100), 1)
communication_score = np.round(np.random.normal(70, 15, N).clip(0, 100), 1)
attendance_percent = np.round(np.random.normal(82, 10, N).clip(40, 100), 1)

# A "hidden" scoring function decides placement probability (this is what the
# tree will try to learn), with some noise so it's not a trivial rule.
score = (
    0.9 * (cgpa - 5)
    - 1.3 * backlogs
    + 0.8 * internships
    + 0.5 * projects
    + 0.03 * coding_score
    + 0.02 * communication_score
    + 0.02 * (attendance_percent - 75)
)
prob = 1 / (1 + np.exp(-(score - 6.5)))  # logistic squashing, tuned for ~55/45 split
noise = np.random.normal(0, 0.12, N)
placed = ((prob + noise) > 0.5).astype(int)

df = pd.DataFrame({
    "cgpa": cgpa,
    "backlogs": backlogs,
    "internships": internships,
    "projects": projects,
    "coding_score": coding_score,
    "communication_score": communication_score,
    "attendance_percent": attendance_percent,
    "placed": placed,
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/placement_data.csv", index=False)
print(f"Saved data/placement_data.csv with {len(df)} rows")
print(df["placed"].value_counts(normalize=True))
