# Exploratory Data Analysis (EDA) - Heart Disease Dataset
# File: heart.csv
# Install once if needed:
# pip install pandas numpy matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("heart.csv")

print("\n========== DATASET OVERVIEW ==========")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

# -----------------------------
# 2. Basic information
# -----------------------------
print("\n========== DATASET INFO ==========")
df.info()

print("\n========== DESCRIPTIVE STATISTICS ==========")
print(df.describe())

# -----------------------------
# 3. Missing values
# -----------------------------
print("\n========== MISSING VALUES ==========")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100

missing_table = pd.DataFrame({
    "Missing Values": missing,
    "Missing Percentage": missing_pct
})

print(missing_table)

# -----------------------------
# 4. Duplicate rows
# -----------------------------
print("\n========== DUPLICATES ==========")
print("Number of duplicate rows:", df.duplicated().sum())

# -----------------------------
# 5. Unique values
# -----------------------------
print("\n========== UNIQUE VALUES ==========")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values -> {df[col].unique()}")

# -----------------------------
# 6. Target distribution
# target: 0 = no heart disease
#         1 = heart disease
# -----------------------------
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="target")
plt.title("Heart Disease Target Distribution")
plt.xlabel("Target (0 = No Disease, 1 = Disease)")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.show()

# -----------------------------
# 7. Histograms for numerical features
# -----------------------------
df.hist(figsize=(16, 12), bins=20, edgecolor="black")
plt.suptitle("Distribution of Dataset Features", fontsize=16)
plt.tight_layout()
plt.show()

# -----------------------------
# 8. Age distribution
# -----------------------------
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="age", bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# -----------------------------
# 9. Heart disease by age
# -----------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="target", y="age")
plt.title("Age Distribution by Heart Disease Status")
plt.xlabel("Target (0 = No Disease, 1 = Disease)")
plt.ylabel("Age")
plt.tight_layout()
plt.show()

# -----------------------------
# 10. Heart disease by sex
# -----------------------------
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="sex", hue="target")
plt.title("Heart Disease Distribution by Sex")
plt.xlabel("Sex (0 = Female, 1 = Male)")
plt.ylabel("Number of Patients")
plt.legend(title="Target", labels=["No Disease", "Disease"])
plt.tight_layout()
plt.show()

# -----------------------------
# 11. Chest pain type vs target
# -----------------------------
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="cp", hue="target")
plt.title("Chest Pain Type vs Heart Disease")
plt.xlabel("Chest Pain Type")
plt.ylabel("Number of Patients")
plt.legend(title="Target", labels=["No Disease", "Disease"])
plt.tight_layout()
plt.show()

# -----------------------------
# 12. Maximum heart rate vs target
# -----------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="target", y="thalach")
plt.title("Maximum Heart Rate vs Heart Disease")
plt.xlabel("Target")
plt.ylabel("Maximum Heart Rate (thalach)")
plt.tight_layout()
plt.show()

# -----------------------------
# 13. Cholesterol distribution by target
# -----------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="target", y="chol")
plt.title("Cholesterol Level by Heart Disease Status")
plt.xlabel("Target")
plt.ylabel("Cholesterol")
plt.tight_layout()
plt.show()

# -----------------------------
# 14. Resting blood pressure by target
# -----------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="target", y="trestbps")
plt.title("Resting Blood Pressure by Heart Disease Status")
plt.xlabel("Target")
plt.ylabel("Resting Blood Pressure")
plt.tight_layout()
plt.show()

# -----------------------------
# 15. Exercise-induced angina vs target
# -----------------------------
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="exang", hue="target")
plt.title("Exercise-Induced Angina vs Heart Disease")
plt.xlabel("Exercise-Induced Angina (0 = No, 1 = Yes)")
plt.ylabel("Number of Patients")
plt.legend(title="Target", labels=["No Disease", "Disease"])
plt.tight_layout()
plt.show()

# -----------------------------
# 16. Correlation heatmap
# -----------------------------
plt.figure(figsize=(12, 9))
correlation = df.corr(numeric_only=True)
sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# -----------------------------
# 17. Pairplot of important variables
# -----------------------------
selected_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "target"]

sns.pairplot(
    df[selected_cols],
    hue="target",
    diag_kind="hist"
)
plt.suptitle("Pairwise Relationships of Important Features", y=1.02)
plt.show()

# -----------------------------
# 18. Outlier detection using IQR
# -----------------------------
print("\n========== OUTLIER DETECTION ==========")

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print(f"{col}: {len(outliers)} potential outliers")

# -----------------------------
# 19. Boxplots for numerical features
# -----------------------------
plt.figure(figsize=(16, 10))
df[numeric_cols].plot(
    kind="box",
    subplots=True,
    layout=(4, 4),
    figsize=(16, 12),
    sharex=False,
    sharey=False
)
plt.suptitle("Boxplots for Numerical Features", fontsize=16)
plt.tight_layout()
plt.show()

# -----------------------------
# 20. Target-wise summary
# -----------------------------
print("\n========== TARGET-WISE MEANS ==========")
print(df.groupby("target").mean(numeric_only=True).round(2))

print("\n========== FINAL EDA SUMMARY ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print("Target distribution:")
print(df["target"].value_counts())
