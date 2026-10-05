"""Quick tracking-quality check.  Usage: python src/check_ids.py path/to/tracks.csv"""
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
life = df.groupby("id")["t"].agg(lambda s: s.max() - s.min())
print("unique IDs:", df["id"].nunique())
print("IDs lasting under 5 s:", int((life < 5).sum()))
print("mean detection confidence:", round(df["conf"].mean(), 3) if "conf" in df else "n/a")
print("\nShortest-lived IDs (likely broken tracks):")
print(life.sort_values().head(15))
