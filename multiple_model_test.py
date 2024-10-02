# Load Important Modules
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
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from yellowbrick.model_selection import CVScores

from sklearn.metrics import confusion_matrix, classification_report

# ML Models
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Load the dataset
df = pd.read_csv('./Dataset/Dis2.csv')

# Feature and Target Variable Split: Only use 'ELEVATION', 'SLOPE', and 'NDVI'
X = df[['ELEVATION', 'SLOPE', 'NDVI']]
y = df[['Y']]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=15, stratify=y)
X_train.shape, X_test.shape

# Cross-validation setup
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)

# Helper function for model tuning
def tune_model(classifier, param_grid, X, y):
    grid = GridSearchCV(classifier, param_grid, refit=True, cv=cv, verbose=3, n_jobs=4) 
    grid = grid.fit(X, y)
    return grid

# Logistic Regression Model
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'penalty': ['l2']  # Only 'l2' because some solvers don't support 'l1' or 'elasticnet'
}

lR = tune_model(LogisticRegression(), param_grid_lr, X_train, y_train)
print(lR.best_params_)
print(lR.best_score_)

lR = lR.best_estimator_
lR.fit(X_train, y_train)

# Evaluation on test set
y_test_pred = lR.predict(X_test)
print(classification_report(y_test, y_test_pred, digits=4))

# Confusion Matrix Visualization
from yellowbrick.classifier import confusion_matrix
plt.figure(figsize=(6, 5))
visualizer = confusion_matrix(lR, X_test, y_test, is_fitted=True)
visualizer.show()

# ROC-AUC Curve
from yellowbrick.classifier import ROCAUC
plt.figure(figsize=(6.4, 4.8))
visualizer = ROCAUC(lR, is_fitted=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

# K-Nearest Neighbors Model
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11, 13],
    'p': [1, 2]
}

knn = tune_model(KNeighborsClassifier(), param_grid_knn, X_train, y_train)
print(knn.best_params_)
print(knn.best_score_)

knn = knn.best_estimator_
knn.fit(X_train, y_train)

y_test_pred_knn = knn.predict(X_test)
print(classification_report(y_test, y_test_pred_knn, digits=4))

# Confusion Matrix for KNN
plt.figure(figsize=(6, 5))
visualizer = confusion_matrix(knn, X_test, y_test, is_fitted=True)
visualizer.show()

# ROC-AUC for KNN
plt.figure(figsize=(6.4, 4.8))
visualizer = ROCAUC(knn, is_fitted=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

# XGBoost Model
param_grid_xgb = {
    'max_depth': [2, 3, 5],
    'n_estimators': [500, 1000, 1500],
    'learning_rate': [0.01, 0.1, 0.05],
    'gamma': [0, 0.1, 0.5],
    'subsample': [0.7, 0.9, 1],
    'n_jobs': [4]
}

xgb_tuned = tune_model(XGBClassifier(), param_grid_xgb, X_train, y_train)
print(xgb_tuned.best_params_)
print(xgb_tuned.best_score_)

xgb_model = xgb_tuned.best_estimator_
xgb_model.fit(X_train, y_train)

# XGBoost Evaluation
y_test_pred_xgb = xgb_model.predict(X_test)
print(classification_report(y_test, y_test_pred_xgb, digits=4))

# Confusion Matrix for XGBoost
plt.figure(figsize=(6, 5))
visualizer = confusion_matrix(xgb_model, X_test, y_test, is_fitted=True)
visualizer.show()

# ROC-AUC for XGBoost
plt.figure(figsize=(6.4, 4.8))
visualizer = ROCAUC(xgb_model, is_fitted=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

# SHAP for XGBoost Model Interpretation
import shap
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_train)

# Visualize SHAP values
shap.plots.beeswarm(shap_values, max_display=3)  # Only 3 features

# Summary Plot
shap.summary_plot(shap_values, X_train)

# Decision Plot
shap.decision_plot(explainer.expected_value, shap_values[1], X_train.columns)

