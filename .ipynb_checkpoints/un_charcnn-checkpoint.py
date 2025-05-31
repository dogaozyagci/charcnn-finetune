import pandas as pd
import re

df_fake = pd.read_csv("data/Fake.csv")
df_real = pd.read_csv("data/True.csv")
df_fake["label"] = 1
df_real["label"] = 0
df_main = pd.concat([df_fake, df_real], ignore_index=True)
df_main = df_main[["text", "label"]].dropna()

df_fake2 = pd.read_csv("data/fine_tune_data/Second_Fake.csv")
df_real2 = pd.read_csv("data/fine_tune_data/Second_True.csv")
df_fake2["label"] = 1
df_real2["label"] = 0
df_isot = pd.concat([df_fake2, df_real2], ignore_index=True)
df_isot = df_isot[["text", "label"]].dropna()

df_liar = pd.read_csv("data/liar/train.tsv", sep='\t', header=None)
df_liar.columns = [
    "id", "label", "statement", "subject", "speaker", "job_title", "state_info",
    "party_affiliation", "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts", "context"
]

true_labels = ["true", "half-true", "mostly-true"]
false_labels = ["false", "barely-true", "pants-fire"]
df_liar = df_liar[df_liar["label"].isin(true_labels + false_labels)].copy()
df_liar["label"] = df_liar["label"].apply(lambda x: 1 if x in false_labels else 0)
df_liar = df_liar[["statement", "label"]].dropna()
df_liar.rename(columns={"statement": "text"}, inplace=True)

print(f"Main dataset: {len(df_main)} entries")
print(f"ISOT fine-tune dataset: {len(df_isot)} entries")
print(f"LIAR dataset: {len(df_liar)} entries")
