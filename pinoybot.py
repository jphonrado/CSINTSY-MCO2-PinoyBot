"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""

import os
import pickle
from typing import List

<<<<<<< Updated upstream
# Main tagging function
=======
# ===============================
# Load trained model and encoders
# ===============================
with open("training/trained_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("training/feature_encoder.pkl", "rb") as f:
    feature_encoder = pickle.load(f)
with open("training/label_encoder.pkl", "rb") as f:
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

    return [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11]


# ===============================
# Main Tagging Function
# ===============================
>>>>>>> Stashed changes
def tag_language(tokens: List[str]) -> List[str]:
    """
    Tags each token in the input list with its predicted language.
    Args:
        tokens: List of word tokens (strings).
    Returns:
        tags: List of predicted tags ("ENG", "FIL", or "OTH"), one per token.
    """
    # 1. Load your trained model from disk (e.g., using pickle or joblib)
    #    Example: with open('trained_model.pkl', 'rb') as f: model = pickle.load(f)
    #    (Replace with your actual model loading code)

    # 2. Extract features from the input tokens to create the feature matrix
    #    Example: features = ... (your feature extraction logic here)

    # 3. Use the model to predict the tags for each token
    #    Example: predicted = model.predict(features)

    # 4. Convert the predictions to a list of strings ("ENG", "FIL", or "OTH")
    #    Example: tags = [str(tag) for tag in predicted]

    # 5. Return the list of tags
    #    return tags

    # You can define other functions, import new libraries, or add other Python files as needed, as long as
    # the tag_language function is retained and correctly accomplishes the expected task.

    # Currently, the bot just tags every token as FIL. Replace this with your more intelligent predictions.
    return ['FIL' for i in tokens]

if __name__ == "__main__":
    # Example usage
    example_tokens = ["Love", "kita", "."]
    print("Tokens:", example_tokens)
    tags = tag_language(example_tokens)