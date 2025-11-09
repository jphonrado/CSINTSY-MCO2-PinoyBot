"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies 
the language of each word in a code-switched Filipino-English text. The function is designed 
to be called with a list of tokens and returns a list of tags ("ENG", "FIL", or "OTH").

Model training and feature extraction are implemented in separate scripts. 
The trained model and encoders are loaded here for prediction.
"""

import pickle
import pandas as pd
from typing import List

# ===============================
# Load trained model and encoders
# ===============================
with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("feature_encoder.pkl", "rb") as f:
    feature_encoder = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


# ===============================
# Feature Extraction Function
# ===============================
def extract_features(word):
    """Extracts linguistic and orthographic features from a given word."""
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
    
    if vratio < 0.35: x2 = "Low"
    elif vratio <= 0.55: x2 = "Mid"
    else: x2 = "High"

    # Contains English clusters
    x3 = "Yes" if any(cl in w for cl in ["th", "sh", "ch", "ph", "tion", "ous", "ment"]) else "No"

    # Contains the letter 'c'
    x4 = "Yes" if "c" in w else "No"

    # Word length
    L = len(w)
    if L <= 3: x5 = "1-3"
    elif L <= 6: x5 = "4-6"
    elif L <= 9: x5 = "7-9"
    else: x5 = "10+"

    # Starts with a capital letter
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

    return [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11]


# ===============================
# Main Tagging Function
# ===============================
def tag_language(tokens: List[str]) -> List[str]:
    """
    Tags each token in the input list with its predicted language.
    Args:
        tokens: List of word tokens (strings).
    Returns:
        tags: List of predicted tags ("ENG", "FIL", or "OTH"), one per token.
    """
    # 1. Extract features for each token
    features = [extract_features(token) for token in tokens]
    feature_df = pd.DataFrame(features, columns=["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11"])

    # 2. Encode the features using the saved encoder
    X_encoded = feature_encoder.transform(feature_df)

    # 3. Predict using the trained model
    predictions = model.predict(X_encoded)

    # 4. Decode numeric labels back to text form
    tags = label_encoder.inverse_transform(predictions)

    return tags.tolist()


# ===============================
# Example Usage
# ===============================
if __name__ == "__main__":
    example_tokens = ["Love", "kita", "the", "world", "Marcos", "nanalo", "hindi", "ako", "nakasdl;kf"]
    print("Tokens:", example_tokens)
    predicted_tags = tag_language(example_tokens)
    print("Predicted tags:", predicted_tags)

