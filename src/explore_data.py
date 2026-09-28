"""Quick look at BANKING77. Run: python src/explore_data.py"""
from load_data import load_banking77

train = load_banking77("train")
test = load_banking77("test")

print(f"train: {len(train)} rows, test: {len(test)} rows")
print(f"columns: {list(train.columns)}\n")

print("Five random examples:")
for _, row in train.sample(5, random_state=0).iterrows():
    print(f"  [{row['intent']}] {row['text']}")

lengths = train["text"].str.split().str.len()
print(f"\nMessage length in words: median {lengths.median():.0f}, min {lengths.min()}, max {lengths.max()}")

counts = train["intent"].value_counts()
print(f"\nMessages per intent (train): min {counts.min()}, median {counts.median():.0f}, max {counts.max()}")
print("\nAll 77 intents with one example each:")
for intent in sorted(counts.index):
    example = train.loc[train["intent"] == intent, "text"].iloc[0]
    print(f"  {intent:<50} {counts[intent]:>4}  e.g. \"{example[:70]}\"")
