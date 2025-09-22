# Heart Attack Likelihood ML Analysis & Visualization
# Author: YOUR NAME
# License: MIT

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

### 1. Load Data
df = pd.read_csv('heart_attack_prediction_dataset.csv')

### 2. Data Overview
print("First 5 rows:\n", df.head())
print("Missing values:\n", df.isnull().sum())

### 3. Data Visualization

# Cholesterol distribution
plt.figure(figsize=(7,4))
sns.histplot(df['Cholesterol'], bins=30, kde=True)
plt.title('Cholesterol Level Distribution')
plt.xlabel('Cholesterol')
plt.show()

# Heart attack incidence by cholesterol
plt.figure(figsize=(7,4))
sns.boxplot(x='Heart Attack Risk', y='Cholesterol', data=df)
plt.title('Cholesterol Levels by Heart Attack Occurrence')
plt.show()

# Correlation heatmap (numeric columns only)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(10,8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

### 4. Feature Engineering

# Extract systolic blood pressure (first number before the slash)
df['Systolic BP'] = df['Blood Pressure'].apply(lambda x: float(x.split('/')[0]))

features = ['Cholesterol', 'Age', 'Systolic BP', 'Smoking']  # Modify as needed!
X = df[features]
y = df['Heart Attack Risk']

# Scaling (good for most models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

### 5. Model Training and Evaluation

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print("Logistic Regression Report\n", classification_report(y_test, lr_pred))
print("Logistic Regression AUC:", roc_auc_score(y_test, lr.predict_proba(X_test)[:,1]))

# Random Forest (feature importance)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("\nRandom Forest Report\n", classification_report(y_test, rf_pred))
print("Random Forest AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))

# Confusion matrix
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d')
plt.title('Random Forest Confusion Matrix')
plt.show()

# Feature importances
importances = rf.feature_importances_
plt.figure(figsize=(6,4))
sns.barplot(x=importances, y=features)
plt.title('Random Forest Feature Importances')
plt.show()

# Cross-validation
cv_scores = cross_val_score(lr, X_scaled, y, cv=5, scoring='roc_auc')
print("Logistic Regression Cross-Validated AUC:", cv_scores.mean())

### 6. Technical Notes for Resume/Projects

"""
- Data wrangling, statistical visualization (seaborn/matplotlib)
- Feature engineering and normalization
- Model training (logistic regression, random forest)
- Cross-validation, classification metrics (accuracy, AUC, confusion matrix)
- Feature importance visualization for interpretability
- Clean and modular project code (suitable for Jupyter Notebook or script)
"""
