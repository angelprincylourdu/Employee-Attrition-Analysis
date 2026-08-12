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

# ==============================
# HR DASHBOARD WITH KPIs
# ==============================

import matplotlib.pyplot as plt
import seaborn as sns

# KPI calculations
total_employees = len(df)
employees_left = (df["Attrition"] == "Yes").sum()
attrition_rate = (employees_left / total_employees) * 100

# ==============================
# HR DASHBOARD WITH KPIs
# ==============================

import matplotlib.pyplot as plt
import seaborn as sns

# KPI calculations
total_employees = len(df)
employees_left = (df["Attrition"] == "Yes").sum()
attrition_rate = (employees_left / total_employees) * 100

# ============================================================
# EMPLOYEE ATTRITION ANALYSIS
# Python, Data Visualization and Machine Learning
# ============================================================

# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# -----------------------------
# 2. LOAD DATASET
# -----------------------------

file_path = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

df = pd.read_csv(file_path)

print("\n===================================")
print("EMPLOYEE ATTRITION ANALYSIS")
print("===================================")

print("\nDataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())


# -----------------------------
# 3. DATA INFORMATION
# -----------------------------

print("\n===================================")
print("DATA INFORMATION")
print("===================================")

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# -----------------------------
# 4. BASIC STATISTICS
# -----------------------------

print("\n===================================")
print("BASIC STATISTICS")
print("===================================")

print(df.describe())


# -----------------------------
# 5. ATTRITION ANALYSIS
# -----------------------------

print("\n===================================")
print("ATTRITION ANALYSIS")
print("===================================")

print("\nAttrition Count:")
print(df["Attrition"].value_counts())

print("\nAttrition Percentage:")
print(df["Attrition"].value_counts(normalize=True) * 100)


# -----------------------------
# 6. GRAPH 1 - EMPLOYEE ATTRITION
# -----------------------------

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Attrition")

plt.title("Employee Attrition")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")

plt.tight_layout()
plt.show()


# -----------------------------
# 7. GRAPH 2 - ATTRITION BY DEPARTMENT
# -----------------------------

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="Department",
    hue="Attrition"
)

plt.title("Employee Attrition by Department")
plt.xlabel("Department")
plt.ylabel("Number of Employees")

plt.xticks(rotation=15)

plt.tight_layout()
plt.show()


# -----------------------------
# 8. GRAPH 3 - ATTRITION BY JOB ROLE
# -----------------------------

plt.figure(figsize=(12, 6))

sns.countplot(
    data=df,
    x="JobRole",
    hue="Attrition"
)

plt.title("Employee Attrition by Job Role")
plt.xlabel("Job Role")
plt.ylabel("Number of Employees")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()


# -----------------------------
# 9. GRAPH 4 - AGE DISTRIBUTION
# -----------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Age",
    hue="Attrition",
    kde=True,
    bins=20
)

plt.title("Age Distribution and Employee Attrition")
plt.xlabel("Age")
plt.ylabel("Number of Employees")

plt.tight_layout()
plt.show()


# -----------------------------
# 10. GRAPH 5 - MONTHLY INCOME
# -----------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Attrition",
    y="MonthlyIncome"
)

plt.title("Monthly Income vs Employee Attrition")
plt.xlabel("Attrition")
plt.ylabel("Monthly Income")

plt.tight_layout()
plt.show()


# -----------------------------
# 11. GRAPH 6 - JOB SATISFACTION
# -----------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="JobSatisfaction",
    hue="Attrition"
)

plt.title("Job Satisfaction vs Employee Attrition")
plt.xlabel("Job Satisfaction")
plt.ylabel("Number of Employees")

plt.tight_layout()
plt.show()


# -----------------------------
# 12. GRAPH 7 - OVERTIME
# -----------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="OverTime",
    hue="Attrition"
)

plt.title("Overtime vs Employee Attrition")
plt.xlabel("Overtime")
plt.ylabel("Number of Employees")

plt.tight_layout()
plt.show()


# -----------------------------
# 13. GRAPH 8 - YEARS AT COMPANY
# -----------------------------

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="Attrition",
    y="YearsAtCompany"
)

plt.title("Years at Company vs Employee Attrition")
plt.xlabel("Attrition")
plt.ylabel("Years at Company")

plt.tight_layout()
plt.show()


# ============================================================
# MACHINE LEARNING
# ============================================================


# -----------------------------
# 14. CREATE COPY FOR ML
# -----------------------------

ml_df = df.copy()


# -----------------------------
# 15. REMOVE UNNECESSARY COLUMNS
# -----------------------------

columns_to_remove = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

ml_df.drop(
    columns=columns_to_remove,
    inplace=True,
    errors="ignore"
)


# -----------------------------
# 16. ENCODE CATEGORICAL DATA
# -----------------------------

label_encoder = LabelEncoder()

for column in ml_df.select_dtypes(include="object").columns:

    ml_df[column] = label_encoder.fit_transform(
        ml_df[column]
    )


# -----------------------------
# 17. SEPARATE FEATURES AND TARGET
# -----------------------------

X = ml_df.drop("Attrition", axis=1)

y = ml_df["Attrition"]


print("\n===================================")
print("MACHINE LEARNING DATA")
print("===================================")

print("Input columns:", X.shape[1])
print("Total rows:", X.shape[0])


# -----------------------------
# 18. TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# -----------------------------
# 19. RANDOM FOREST MODEL
# -----------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# -----------------------------
# 20. TRAIN MODEL
# -----------------------------

model.fit(
    X_train,
    y_train
)

print("\nModel trained successfully!")


# -----------------------------
# 21. MAKE PREDICTIONS
# -----------------------------

y_pred = model.predict(X_test)

print("Predictions completed!")


# -----------------------------
# 22. MODEL ACCURACY
# -----------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n===================================")
print("MODEL RESULTS")
print("===================================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# -----------------------------
# 23. CLASSIFICATION REPORT
# -----------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# -----------------------------
# 24. CONFUSION MATRIX
# -----------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)


plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()


# -----------------------------
# 25. FEATURE IMPORTANCE
# -----------------------------

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


print("\n===================================")
print("TOP IMPORTANT FEATURES")
print("===================================")

print(
    feature_importance.head(10)
)


# -----------------------------
# 26. FEATURE IMPORTANCE GRAPH
# -----------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_importance.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Factors Influencing Employee Attrition")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()
plt.show()


# -----------------------------
# 27. FINAL MESSAGE
# -----------------------------

print("\n===================================")
print("ANALYSIS COMPLETED SUCCESSFULLY!")
print("===================================")

print(
    "Final Model Accuracy:",
    round(accuracy * 100, 2),
    "%"
)