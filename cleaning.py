import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("data.csv")

print("Original Dataset")
print(df)

# Dataset Info
print("\nDataset Information")
print(df.info())

# Missing Values
print("\nMissing Values")
print(df.isnull().sum())

# Fill Missing Values
df["Age"].fillna(df["Age"].mean(), inplace=True)
df["Salary"].fillna(df["Salary"].mean(), inplace=True)

# Remove Duplicate Rows
df.drop_duplicates(inplace=True)

# Outlier Detection using IQR

Q1 = df["Salary"].quantile(0.25)
Q3 = df["Salary"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df["Salary"] >= lower) & (df["Salary"] <= upper)]

print("\nCleaned Dataset")
print(df)

# Save Cleaned Data
df.to_csv("cleaned_data.csv", index=False)

# -----------------------------
# Visualization
# -----------------------------

sns.set(style="whitegrid")

# Salary Distribution

plt.figure(figsize=(8,5))
sns.histplot(df["Salary"], bins=10, color="blue")
plt.title("Salary Distribution")
plt.show()

# Department Count

plt.figure(figsize=(6,4))
sns.countplot(x="Department", data=df)
plt.title("Department Count")
plt.show()

# Gender Count

plt.figure(figsize=(5,4))
sns.countplot(x="Gender", data=df)
plt.title("Gender Distribution")
plt.show()

# Age vs Salary

plt.figure(figsize=(7,5))
sns.scatterplot(x="Age", y="Salary", hue="Department", data=df)
plt.title("Age vs Salary")
plt.show()

# Average Salary

plt.figure(figsize=(6,4))
df.groupby("Department")["Salary"].mean().plot(kind="bar", color="green")
plt.title("Average Salary by Department")
plt.ylabel("Salary")
plt.show()

print("\nProject Completed Successfully")