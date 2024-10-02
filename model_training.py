# Import Important Modules
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.rcParams['figure.figsize'] = (12, 8)
import seaborn as sns
sns.set(style='whitegrid', color_codes=True)
import warnings
warnings.filterwarnings('ignore')

# Sckit Learn Specific Modules
from sklearn.feature_selection import chi2, mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from yellowbrick.model_selection import CVScores
from sklearn.metrics import confusion_matrix, classification_report

# ML Models
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.tree import DecisionTreeClassifier

# Exploratory Data Analysis (EDA)
df = pd.read_csv('./Dataset/Dis2.csv')
df.head()

# Use only DEM, Slope, and NDVI
# Select the relevant features
X = df[['ELEVATION', 'SLOPE', 'NDVI']]
y = df[['Y']]

# Basic Data Info
print(df.info())

# Summary Statistics for the selected features
print(df[['ELEVATION', 'SLOPE', 'NDVI']].describe())

# Visualize the distribution of target variable (Y)
def count_categories(feature): 
    plt.figure(figsize=(6, 6))
    sns.countplot(
        x=feature,
        data=df
    )
    plt.show()

count_categories('Y')

# Visualize distributions of the selected features
for feature in X.columns:
    count_categories(feature)

# Check balance of the target variable
print(df['Y'].value_counts())

# Correlation Matrix using only the selected features
plt.figure(figsize=(8, 6))
sns.heatmap(
    data=df[['ELEVATION', 'SLOPE', 'NDVI', 'Y']].corr('kendall'),
    annot=True,
    fmt='.0%',
    cmap='coolwarm'
)
plt.title("Kendall Correlation Matrix for DEM, Slope, NDVI, and Target")
plt.show()

# Normality Test using Shapiro-Wilk Test for the selected features
from scipy.stats import shapiro
def normality_test_shapiro(feature):
    stat, p = shapiro(X[feature].values)
    print(f'Statistics={stat:.3f}, p={p:.3f} for {feature}')
    # Interpret results
    alpha = 0.05
    if p > alpha:
        print(f'{feature} looks Gaussian (fail to reject H0)')
    else:
        print(f'{feature} does not look Gaussian (reject H0)')

for feature in X.columns:
    normality_test_shapiro(feature)

# Chi-Square Test for the selected features
chi_scores = chi2(X, y)[0]
chi_pvalues = chi2(X, y)[1]
for feature, score, pvalue in zip(X.columns, chi_scores, chi_pvalues):
    print(f"Chi-Square Test - {feature}: Score={score}, P-Value={pvalue:.5f}")

# Create a DataFrame to show Chi-Square test results
chi_square_df = pd.DataFrame({'Feature': X.columns, 'Chi2 Score': chi_scores, 'P-Value': chi_pvalues})
print(chi_square_df)

# Feature Ranking using Yellowbrick Rank1D
from yellowbrick.features import Rank1D

# Instantiate the 1D visualizer with the Shapiro ranking algorithm
visualizer = Rank1D(algorithm='shapiro', features=X.columns)
visualizer.fit(X, y)  # Fit the data to the visualizer
visualizer.transform(X.values)  # Transform the data
visualizer.show()  # Finalize and render the figure

# Display the correlations between the selected features and the target
print("Correlation between features and the target (Y):")
print(df[['ELEVATION', 'SLOPE', 'NDVI', 'Y']].corr())

# Train a simple Decision Tree Classifier using only these three features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix for Decision Tree Classifier")
plt.show()
