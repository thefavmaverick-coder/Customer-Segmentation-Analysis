# Customer Segmentation Analysis

A customer analytics and machine-learning project using Python, exploratory data analysis (EDA), feature engineering, and K-Means clustering to identify groups of customers with similar demographic and purchasing characteristics.

## Project Overview

This project analyzes a customer marketing dataset containing demographic information, income, product spending, purchase-channel activity, recency, and campaign-response fields. The workflow cleans the data, creates customer-level features, explores relationships and spending patterns, and applies K-Means clustering to group customers.

The analysis was developed in Google Colab and exported as a PDF report. The exported report is included in [`docs/`](doc/Customer_Segmentation_Analysis - Colab.pdf).

## Objectives

- Inspect and prepare the customer dataset for analysis.
- Explore customer demographics, income, spending, and purchase behavior.
- Engineer useful customer-level features.
- Examine relationships between income, age, recency, spending, and purchase channels.
- Scale selected numerical features before clustering.
- Use the Elbow Method to examine candidate cluster counts.
- Apply K-Means and summarize the resulting customer groups.
- Use Principal Component Analysis (PCA) to prepare a lower-dimensional representation.

## Dataset

The report analyzes a dataset with **2,240 rows and 29 original columns**. After dropping records with missing income values, **2,216 customer records** remain.

The original fields include:

- Customer attributes: `ID`, `Year_Birth`, `Education`, `Marital_Status`, `Income`
- Household information: `Kidhome`, `Teenhome`
- Customer activity: `Dt_Customer`, `Recency`
- Product spending: `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`
- Purchase channels: `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth`
- Campaign and response fields: `AcceptedCmp1`–`AcceptedCmp5`, `Complain`, `Response`
- Additional source fields: `Z_CostContact`, `Z_Revenue`

The PDF report is provided for documentation. The raw dataset is not included in this repository; add it only if its source terms permit redistribution.

## Tools and Libraries

- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Workflow and Methods

### 1. Data loading and inspection

Loaded the CSV with Pandas and reviewed the dataset dimensions, sample records, column names, descriptive statistics, and missing-value counts.

### 2. Data cleaning

- Identified 24 missing values in `Income`.
- Removed rows containing missing values using `dropna()`.
- The resulting analysis dataset contains 2,216 rows.
- Converted `Dt_Customer` to a datetime field using day-first parsing.

The PDF also shows an unusually high income value of `666666` in the descriptive statistics. The report does not show a documented outlier treatment for this value, so it should be reviewed before using the results for business decisions.

### 3. Feature engineering

Created the following derived fields:

- `Age`: calculated as `2025 - Year_Birth`.
- `Total_Children`: sum of `Kidhome` and `Teenhome`.
- `Total_Spend`: sum of the six product spending columns.
- `Customer_Since`: elapsed days between the customer enrollment date and the date the notebook was run.
- `AcceptedAny`: binary indicator showing whether at least one of the campaign acceptance/response fields was positive.
- `AgeGroup`: grouped age bands used for demographic comparisons.

**Reproducibility note:** `Age` uses 2025 as a fixed baseline in the notebook. `Customer_Since` uses `pd.Timestamp("today")`, so that feature changes depending on when the notebook is executed.

### 4. Exploratory Data Analysis

The report includes:

- Customer age distribution.
- Customer income distribution.
- Total customer spending distribution.
- Income distribution by education level.
- Spending by marital status.
- Correlation matrix for income, age, recency, total spending, and purchase-channel counts.
- Average income by education and marital status.
- Average spending by education level.
- Campaign acceptance rate by marital status.
- Average income by age group.

### 5. Customer segmentation with K-Means

The clustering model uses these seven numerical features:

1. `Age`
2. `Income`
3. `Total_Spend`
4. `NumWebPurchases`
5. `NumStorePurchases`
6. `NumWebVisitsMonth`
7. `Recency`

The selected features are standardized with Scikit-learn's `StandardScaler`. The notebook calculates within-cluster sum of squares (WCSS/inertia) for cluster counts from 2 through 9 and plots the Elbow Method. It then fits a K-Means model with **6 clusters**.

### 6. Cluster profile

The report presents the following mean values for the seven clustering features:

| Cluster | Age | Income | Total Spend | Web Purchases | Store Purchases | Web Visits / Month | Recency |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 55.22 | 5,165.78 | 88.56 | 0.33 | 0.22 | 17.33 | 43.33 |
| 1 | 52.37 | 34,773.54 | 120.39 | 2.24 | 3.39 | 6.41 | 24.10 |
| 2 | 59.75 | 59,512.23 | 879.62 | 7.96 | 7.79 | 6.44 | 46.75 |
| 3 | 54.96 | 36,548.55 | 135.28 | 2.42 | 3.49 | 6.38 | 74.96 |
| 4 | 46.60 | 78,013.16 | 1,301.58 | 4.68 | 8.68 | 2.72 | 48.94 |
| 5 | 70.13 | 72,362.11 | 1,137.83 | 4.48 | 8.25 | 2.73 | 51.61 |

Cluster sizes reported in the notebook:

| Cluster | Customers |
|---:|---:|
| 0 | 9 |
| 1 | 560 |
| 2 | 421 |
| 3 | 544 |
| 4 | 349 |
| 5 | 333 |

These are descriptive cluster profiles. The notebook does not assign validated business labels to the clusters, so labels such as “loyal,” “premium,” or “at-risk” should not be treated as established findings without further validation.

### 7. PCA

The notebook imports `PCA` and computes a two-component transformation from the standardized features. The supplied PDF ends while accessing `df["PCA1"]`; it does not show the completed PCA visualization or further PCA interpretation.

## Key Findings from the Report

- The cleaned dataset contains 2,216 customers after 24 records with missing income were removed.
- Income and total spending have a positive correlation in the reported correlation matrix (approximately 0.668).
- Total spending and store purchases have a positive correlation (approximately 0.675).
- Average spending differs across education groups in the report; the PhD group has the highest displayed average, while the Basic group has the lowest.
- The six-cluster K-Means result contains one very small cluster of 9 customers, which merits additional investigation.
- Cluster 4 has the highest mean total spending in the reported cluster summary, while Cluster 0 has the lowest mean income and spending.

## How to Reproduce

1. Obtain the source customer CSV dataset from its original provider, subject to its terms of use.
2. Open Google Colab or a local Python environment.
3. Install the libraries listed in `requirements.txt` (if you add that file) or install the packages listed above.
4. Run the notebook cells in order, updating the CSV path or upload step as needed.
5. Review the generated EDA, Elbow Method, and cluster summary outputs.

## Limitations and Recommended Improvements

- Investigate the unusually high income value before finalizing segmentation.
- Consider median imputation as an alternative to dropping the 24 records with missing income, and compare the results.
- Set a fixed `random_state` and an explicit `n_init` for reproducible K-Means results.
- Evaluate cluster quality with metrics such as silhouette score; the PDF imports `silhouette_score` but does not show a computed score.
- Investigate the very small cluster (9 customers) and test whether the six-cluster solution is stable.
- Make the `Customer_Since` reference date fixed and document it for reproducibility.
- Complete the PCA visualization and interpretation.
- Save the fitted scaler and K-Means model if building a prediction application; the PDF does not show model-export steps.

## Conclusion

This project demonstrates an end-to-end customer analytics workflow, from data inspection and cleaning through exploratory analysis, feature engineering, standardization, and K-Means clustering. The resulting six-cluster solution provides a starting point for understanding differences in customer income, spending, purchase-channel activity, and recency. The cluster summaries highlight meaningful variation between groups, while the small size of one cluster and the unusually high income value indicate areas that deserve further validation. Overall, the analysis illustrates how Python-based analytics and unsupervised learning can help organize customer data into interpretable groups and support more focused customer strategy.

---

**Project:** Customer Segmentation Analysis  
**Category:** Data Analytics | Machine Learning | Customer Analytics  
**Language:** Python  
**Environment:** Google Colab
