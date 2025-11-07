import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

data = pd.read_csv("//put path ng validated set")

X_words = data["word"].astype(str)
y = data["label"]

# (70-15-15)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_words, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# feature extraction using character n-grams
vectorizer = CountVectorizer(analyzer="char", ngram_range=(2, 4))
X_train_feats = vectorizer.fit_transform(X_train)
X_val_feats = vectorizer.transform(X_val)
X_test_feats = vectorizer.transform(X_test)

# training the model
nb = MultinomialNB()
nb.fit(X_train_feats, y_train)

# evaluation of model
print("Validation Performance:")
print(classification_report(y_val, nb.predict(X_val_feats)))

print("Test Performance:")
y_test_pred = nb.predict(X_test_feats)
print(classification_report(y_test, y_test_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))


joblib.dump(nb, "pinoybot_nb_model.pkl")
joblib.dump(vectorizer, "pinoybot_vectorizer.pkl")

print("\n Model and vectorizer saved successfully!")
