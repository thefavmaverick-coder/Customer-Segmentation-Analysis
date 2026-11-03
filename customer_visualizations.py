# CUSTOMER SEGMENTATION - DATA VISUALIZATION
# Creates the 7 charts shown in the reference screenshots.
#
# Expected input file:
#   customer_segmentation.csv
#
# Run:
#   python customer_visualizations.py

from pathlib import Path
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# -----------------------------
# 1. SETTINGS
# -----------------------------
DATA_FILE = "customer_segmentation.csv"
OUTPUT_DIR = Path("visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="white")  # Similar clean style to the reference images

# -----------------------------
# 2. LOAD AND PREPARE DATA
# -----------------------------
df = pd.read_csv(DATA_FILE)

# Remove rows with missing Income, as in the project workflow
df = df.dropna(subset=["Income"]).copy()

# Age is calculated using the same fixed reference year as the notebook
df["Age"] = 2025 - df["Year_Birth"]

# Add total customer spending across the six product categories
spending_columns = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]
df["Total_Spend"] = df[spending_columns].sum(axis=1)

print(f"Rows used for visualizations: {len(df):,}")
print(f"Charts will be saved in: {OUTPUT_DIR.resolve()}")


def save_chart(filename):
    """Save the current Matplotlib figure as a high-resolution PNG."""
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


# -----------------------------
# 3. CUSTOMER AGE DISTRIBUTION
# -----------------------------
plt.figure(figsize=(9, 6))
sns.histplot(
    data=df,
    x="Age",
    bins=25,
    kde=True,
    color="steelblue"
)
plt.title("Distribution of Customer Ages", fontsize=14)
plt.xlabel("Age")
plt.ylabel("Count")
save_chart("01_age_distribution.png")


# -----------------------------
# 4. CUSTOMER INCOME DISTRIBUTION
# -----------------------------
plt.figure(figsize=(9, 6))
sns.histplot(
    data=df,
    x="Income",
    bins=25,
    kde=True,
    color="steelblue"
)
plt.title("Distribution of Customer Income", fontsize=14)
plt.xlabel("Income")
plt.ylabel("Count")
save_chart("02_income_distribution.png")


# -----------------------------
# 5. CUSTOMER SPENDING DISTRIBUTION
# -----------------------------
plt.figure(figsize=(9, 6))
sns.histplot(
    data=df,
    x="Total_Spend",
    bins=30,
    kde=True,
    color="steelblue"
)
plt.title("Distribution of Customer Spending", fontsize=14)
plt.xlabel("Spending")
plt.ylabel("Count")
save_chart("03_customer_spending.png")


# -----------------------------
# 6. INCOME DISTRIBUTION BY EDUCATION
# -----------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df,
    x="Education",
    y="Income",
    color="steelblue"
)
plt.title("Income Distribution by Education Level", fontsize=14)
plt.xlabel("Education")
plt.ylabel("Income")
plt.xticks(rotation=45)
save_chart("04_income_education_level.png")


# -----------------------------
# 7. SPENDING BY MARITAL STATUS
# -----------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df,
    x="Marital_Status",
    y="Total_Spend",
    color="steelblue"
)
plt.title("Spending by Marital Status", fontsize=14)
plt.xlabel("Marital_Status")
plt.ylabel("Total_Spend")
plt.xticks(rotation=45)
save_chart("05_spending_by_marital_status.png")


# -----------------------------
# 8. CORRELATION MATRIX
# -----------------------------
correlation_columns = [
    "Income",
    "Age",
    "Recency",
    "Total_Spend",
    "NumWebPurchases",
    "NumStorePurchases",
]

correlation_matrix = df[correlation_columns].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2g",
    linewidths=0.5
)
plt.title("Correlation Matrix", fontsize=14)
save_chart("06_correlation_matrix.png")


# -----------------------------
# 9. ELBOW METHOD - K-MEANS
# -----------------------------
cluster_features = [
    "Age",
    "Income",
    "Total_Spend",
    "NumWebPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "Recency",
]

# Use the same rows for all clustering features
cluster_data = df[cluster_features].dropna().copy()

# Standardize features before K-Means
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)

wcss = []
k_values = range(2, 10)

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    model.fit(scaled_data)
    wcss.append(model.inertia_)

plt.figure(figsize=(9, 6))
plt.plot(
    list(k_values),
    wcss,
    marker="o",
    linestyle="-"
)
plt.title("The Elbow Method", fontsize=14)
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.xticks(list(k_values))
save_chart("07_elbow_method.png")


print("\nAll 7 visualizations have been created successfully.")
print(f"Check this folder: {OUTPUT_DIR.resolve()}")
