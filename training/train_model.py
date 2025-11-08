import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the training data
df = pd.read_csv("training_data.csv")

# 2. Separate features and labels
X = df[["x1", "x2", "x3", "x4", "x5", "x6", "x7"]]
y = df["label"]

# 3. Encode categorical feature columns (Yes/No → 0/1, etc.)
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)

# 4. Encode labels (FIL/ENG/OTH → 0/1/2)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 5. Split the data (70% train, 15% val, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(X_encoded, y_encoded, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 6. Train a Naive Bayes classifier
model = CategoricalNB()
model.fit(X_train, y_train)

# 7. Evaluate model performance
y_pred_val = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))
print("\nClassification Report (Validation Set):\n", classification_report(y_val, y_pred_val, target_names=label_encoder.classes_))

# Optional: test performance
y_pred_test = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))

# 8. Save the trained model and encoders
with open("trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("feature_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("\nModel and encoders saved successfully!")
