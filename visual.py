import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (replace with your own)
df = sns.load_dataset("titanic")

# Basic info
print(df.head())
print(df.info())

# Pie Chart: Survival count
survived_counts = df['survived'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(survived_counts, labels=['Not Survived', 'Survived'], autopct='%1.1f%%', colors=['red', 'green'])
plt.title('Survival Rate')
plt.show()

# Barplot: Average fare paid by class
plt.figure(figsize=(8,6))
sns.barplot(x='class', y='fare', data=df, palette='pastel')
plt.title('Average Fare by Passenger Class')
plt.show()

# Countplot: Survival by Sex
plt.figure(figsize=(8,6))
sns.countplot(x='sex', hue='survived', data=df, palette='Set2')
plt.title('Survival Count by Gender')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

# Heatmap: Correlation Matrix
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()
