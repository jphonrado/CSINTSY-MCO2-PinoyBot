import pandas as pd
import re

#Load annotated data
df = pd.read_csv("annotated_data.csv")

ENG_FUNCTION_LIST = {
    "the","a","an","of","to","in","on","at","by","for","with","as","from",
    "and","but","or","so","if","then","because","than",
    "this","that","these","those","which","who","whom",
    "is","are","was","were","be","been","being","do","does","did","not",
    "i","you","we","they","he","she","it","me","him","her","us","them",
    "my","your","our","their","its","there","here","when","where","why","how"
}


def extract_features(word): 
    w = str(word).lower()

    # Filipino prefixes
    prefixes = ("mag", "nag", "ma", "na", "ka", "pa")
    x1 = "Yes" if w.startswith(prefixes) else "No"

    # Vowel ratio
    letters = [ch for ch in w if ch.isalpha()]
    L = len(letters)
    vowels = {"a", "e", "i", "o", "u"}
    vcount = sum(ch in vowels for ch in letters)
    vratio = (vcount / L) if L else 0.0
    
    if vratio < 0.35:
        x2 = "Low"
    elif vratio <= 0.55:
        x2 = "Mid"
    else: 
        x2 = "High"

    # English clusters or suffixes
    x3 = "Yes" if any(cl in w for cl in ["th", "sh", "ch", "ph", "tion", "ous", "ment"]) else "No"

    # Contains 'c'
    x4 = "Yes" if "c" in w else "No"


    # Word length
    L = len(w)
    if L <= 3: x5 = "1-3"
    elif L <= 6: x5 = "4-6"
    elif L <= 9: x5 = "7-9"
    else: x5 = "10+"

    # Starts with capital (safe)
    x6 = "Yes" if word and word[0].isupper() else "No"

    # Is purely alphabetic
    x7 = "Yes" if word.isalpha() else "No"

    # Contains 'v', 'w', 'x', 'z', 'q' (rare in FIL)
    rare = sum(ch in "vwxzq" for ch in w)
    x8 = "0" if rare == 0 else ("1" if rare == 1 else "2+")

    # Double consonant
    x9 = "Yes" if any(dd in w for dd in ("tt","ll","pp","rr","ss","nn","mm")) else "No"

    # Ends with 's' only for plurals 
    x10 = "Yes" if (w.endswith("s") and not w.endswith(("us","is","as","ng"))) else "No"

    # 'r' following after a vowel
    x11 = "Yes" if any(rv in w for rv in ("ar","er","ir","or","ur")) else "No"
    
    ENG_SUFFIXES = (
    "tion","sion","ment","ness","ity","ous","able","ible","less","ful",
    "ist","ism","ship","ly","al","ary","ery","ed","ing"
    )
    x12 = "Yes" if any(w.endswith(s) for s in ENG_SUFFIXES) else "No"
    
    # function words
    x13 = "Yes" if w in ENG_FUNCTION_LIST else "No"


    return [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13]


# Apply feature extraction
features = df["word"].apply(extract_features)
df[["x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11","x12_eng_suffix", "x13_eng_function"]] = pd.DataFrame(features.tolist(), index=df.index)

# Clean missing or invalid labels
df["label"] = df["label"].fillna("OTH")
df["label"] = df["label"].replace(["UNK", "unknown", "", " "], "OTH")

# Enforce FIL/ENG/OTH only
df["label"] = df["label"].apply(lambda x: x if x in ["FIL", "ENG"] else "OTH")

# Keep only needed columns
processed_df = df[["word","x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11","x12_eng_suffix","x13_eng_function","label"]]

# Save new training file
processed_df.to_csv("training_data.csv", index=False)