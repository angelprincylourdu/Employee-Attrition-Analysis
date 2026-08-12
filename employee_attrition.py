import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries imported successfully!")

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

print("Dataset loaded successfully!")
print(df.head())
print("Dataset Shape:", df.shape)
# Check dataset information
print("\nDataset Information:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Check attrition count
print("\nAttrition Count:")
print(df["Attrition"].value_counts())
# Employee Attrition Distribution

plt.figure(figsize=(6, 4))

sns.countplot(data=df, x="Attrition")

plt.title("Employee Attrition Distribution")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")

plt.show()

# Age vs Attrition

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Age",
    hue="Attrition",
    bins=15,
    multiple="stack"
)
# Salary vs Attrition

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Attrition",
    y="MonthlyIncome"
)

plt.title("Monthly Income vs Employee Attrition")
plt.xlabel("Attrition")
plt.ylabel("Monthly Income")

plt.show()
plt.title("Age vs Employee Attrition")
plt.xlabel("Age")
plt.ylabel("Number of Employees")

# MACHINE LEARNING

# Convert Attrition into numbers
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

# Convert text columns into numbers
df = pd.get_dummies(df, drop_first=True)

# Separate input and output
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

print("\nData prepared for Machine Learning!")
print("Input columns:", X.shape[1])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model trained successfully!")
y_pred = model.predict(X_test)

print("Predictions completed!")

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\n==========================")
print("MODEL RESULTS")
print("==========================")
print("Accuracy:", round(accuracy * 100, 2), "%")

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Employee Attrition - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# Find important factors affecting attrition

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Factors Affecting Employee Attrition:")
print(importance.head(10))
plt.show()

# ==============================
# HR DASHBOARD
# ==============================

# Load original data again for dashboard
dashboard_df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Chart 1: Attrition
sns.countplot(
    data=dashboard_df,
    x="Attrition",
    ax=axes[0]
)

axes[0].set_title("Employee Attrition")
axes[0].set_xlabel("Attrition")
axes[0].set_ylabel("Number of Employees")


# Chart 2: Department
sns.countplot(
    data=dashboard_df,
    x="Department",
    hue="Attrition",
    ax=axes[1]
)

axes[1].set_title("Attrition by Department")
axes[1].set_xlabel("Department")
axes[1].set_ylabel("Number of Employees")

plt.tight_layout()
plt.show()
 # ==============================
# FINAL MODEL EVALUATION
# ==============================

from sklearn.metrics import classification_report

print("\nFINAL CLASSIFICATION REPORT")
print("============================")
print(classification_report(y_test, y_pred))