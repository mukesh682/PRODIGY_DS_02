import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load the dataset
df = pd.read_csv('synthetic_dataset.csv')

# Data Cleaning
# Handle missing values
df['income'].fillna(df['income'].mean(), inplace=True)
df['purchase_amount'].fillna(df['purchase_amount'].median(), inplace=True)

# Remove duplicates
df = df.drop_duplicates()

# Fix inconsistent data
df['gender'] = df['gender'].str.lower().replace({'male': 'Male', 'female': 'Female'})

# Handle outliers using Z-score method
z_scores = np.abs(stats.zscore(df['income']))
df = df[z_scores < 3]

# Exploratory Data Analysis (EDA)
# Histograms for numerical features
df.hist(bins=50, figsize=(20, 15))
plt.show()

# Scatter plot for income vs. purchase_amount
df.plot(kind="scatter", x="income", y="purchase_amount", alpha=0.5)
plt.title('Income vs Purchase Amount')
plt.show()

# Pair plot for age, income, and purchase_amount
sns.pairplot(df[['age', 'income', 'purchase_amount']])
plt.show()

# Bar plot for gender distribution
df['gender'].value_counts().plot(kind='bar')
plt.title('Gender Distribution')
plt.show()

# Correlation matrix and heatmap (excluding non-numeric columns)
numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Feature Engineering: Create a new feature 'age_group'
df['age_group'] = pd.cut(df['age'], bins=[18, 30, 50, 70], labels=['18-30', '31-50', '51-70'])

# Bar plot to visualize the new feature 'age_group'
df['age_group'].value_counts().plot(kind='bar')
plt.title('Age Group Distribution')
plt.show()
