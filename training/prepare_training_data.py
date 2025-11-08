import pandas as pd
import re

#Load annotated data
df = pd.read_csv("annotated_data.csv")

def extract_features(word): 
    w = str(word).lower()

    # Filipino prefixes
    prefixes = ("mag", "nag", "ma", "na", "ka", "pa", "pag", "mang", "pang", "pala", "i")
    x1 = "Yes" if w.startswith(prefixes) else "No"

    # Ends with vowel
    x2 = "Yes" if w.endswith(("a", "e", "i", "o", "u")) else "No"

    # English clusters or suffixes
    x3 = "Yes" if any(cl in w for cl in ["th", "sh", "ch", "ph", "tion", "ous", "ment"]) else "No"

    # Contains 'c'
    x4 = "Yes" if "c" in w else "No"

    # Word length
    x5 = len(w)

    # Starts with capital (safe)
    x6 = "Yes" if word and word[0].isupper() else "No"

    # Is purely alphabetic
    x7 = "Yes" if word.isalpha() else "No"

    return [x1, x2, x3, x4, x5, x6, x7]


# Apply feature extraction
features = df["word"].apply(extract_features)
df[["x1","x2","x3","x4","x5","x6","x7"]] = pd.DataFrame(features.tolist(), index=df.index)

# Clean missing or invalid labels
df["label"] = df["label"].fillna("OTH")
df["label"] = df["label"].replace(["UNK", "unknown", "", " "], "OTH")

# Enforce FIL/ENG/OTH only
df["label"] = df["label"].apply(lambda x: x if x in ["FIL", "ENG"] else "OTH")

# Keep only needed columns
processed_df = df[["word","x1","x2","x3","x4","x5","x6","x7","label"]]

# Save new training file
processed_df.to_csv("training_data.csv", index=False)

