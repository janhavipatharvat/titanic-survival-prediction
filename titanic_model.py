"""
Titanic Survival Prediction
----------------------------
A beginner-friendly end-to-end machine learning project that predicts
whether a passenger survived the Titanic disaster, based on features
like age, sex, ticket class, and fare.

Workflow:
1. Load & explore the data
2. Clean missing values
3. Engineer new features
4. Encode categorical variables
5. Train multiple models
6. Compare accuracy and pick the best one
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
df = pd.read_csv("titanic.csv")
print("Dataset shape:", df.shape)
print("\nMissing values per column:\n", df.isnull().sum())

# ------------------------------------------------------------------
# 2. CLEAN DATA
# ------------------------------------------------------------------
# Age: fill missing values with median age
df["Age"] = df["Age"].fillna(df["Age"].median())

# Embarked: fill missing values with the most common port
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Cabin has too many missing values to be useful directly,
# so instead of dropping it, turn it into a simple "HasCabin" feature
df["HasCabin"] = df["Cabin"].notnull().astype(int)

# Fare: fill the rare missing value with median fare
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# ------------------------------------------------------------------
# 3. FEATURE ENGINEERING (this is what makes the project "yours",
# not just a copy-paste of a tutorial)
# ------------------------------------------------------------------

# FamilySize = siblings/spouses + parents/children + self
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# IsAlone = traveling without any family
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# Extract title from name (Mr, Mrs, Miss, Master, etc.) - a classic
# trick that gives the model social/age/gender info packed into one field
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
# Group rare titles together
rare_titles = df["Title"].value_counts()[df["Title"].value_counts() < 10].index
df["Title"] = df["Title"].replace(rare_titles, "Rare")

# ------------------------------------------------------------------
# 4. ENCODE CATEGORICAL VARIABLES
# ------------------------------------------------------------------
le_sex = LabelEncoder()
df["Sex"] = le_sex.fit_transform(df["Sex"])  # male=1, female=0

le_embarked = LabelEncoder()
df["Embarked"] = le_embarked.fit_transform(df["Embarked"])

le_title = LabelEncoder()
df["Title"] = le_title.fit_transform(df["Title"])

# ------------------------------------------------------------------
# 5. SELECT FEATURES & SPLIT DATA
# ------------------------------------------------------------------
features = [
    "Pclass", "Sex", "Age", "Fare", "Embarked",
    "FamilySize", "IsAlone", "HasCabin", "Title"
]
X = df[features]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------------
# 6. TRAIN & COMPARE MODELS
# ------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200, max_depth=6),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=["Did not survive", "Survived"]))

# ------------------------------------------------------------------
# 7. IDENTIFY BEST MODEL
# ------------------------------------------------------------------
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name} with accuracy {results[best_model_name]:.4f}")

# ------------------------------------------------------------------
# 8. FEATURE IMPORTANCE (Random Forest gives us this for free)
# ------------------------------------------------------------------
rf_model = models["Random Forest"]
importance = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature importance (Random Forest):\n", importance)

# ------------------------------------------------------------------
# 9. SAVE VISUALIZATIONS
# ------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.barplot(x=importance.values, y=importance.index, palette="viridis")
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120)
plt.close()

plt.figure(figsize=(6, 4))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette="mako")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=120)
plt.close()

cm = confusion_matrix(y_test, best_model.predict(X_test))
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Did not survive", "Survived"],
            yticklabels=["Did not survive", "Survived"])
plt.title(f"Confusion Matrix - {best_model_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
plt.close()

print("\nSaved plots: feature_importance.png, model_comparison.png, confusion_matrix.png")
print("\nDone! See README.md for a full writeup of this project.")
