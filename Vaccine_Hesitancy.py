# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import mutual_info_classif

df = pd.read_csv('/Users/yasudayuuya/AI_ethics/Vaccine_Hesitancy_for_COVID-19__County_and_local_estimates_20250227.csv')
df.head()

# Data preprocessing
# Check for missing values
df.isnull().sum()

# Output columns with missing values
df.isnull().sum()[df.isnull().sum() > 0]

# Drop rows with missing values
df = df.dropna()
df.isnull().sum()
df

# Set target variable (here we predict the rate for Total Prevalence)
target = 'Estimated strongly hesitant'
df[target] = df[target].astype(float)  # Convert string to float

# Set features
features = df.drop(columns=['FIPS Code', 'State', 'Estimated hesitant', 'Estimated hesitant or unsure', 'SVI Category', 'CVAC Level Of Concern', 'Percent adults fully vaccinated against COVID-19 (as of 6/10/21)',  'Geographical Point', 'State Code', 'County Boundary', 'State Boundary', target])

# Display feature column names in a table
features.columns

# Remove special characters from feature column names
# The characters to be removed are [ ] % , and space
features.columns = features.columns.str.replace('[\[\]%, ]', '_', regex=True)

# Label encoding for categorical variables
# This converts categorical variables to numeric values
# The county_name here can be converted to numeric
label_encoder = LabelEncoder()
for col in features.select_dtypes(include=['object']).columns:
    features[col] = label_encoder.fit_transform(features[col])

# Encode County_Name
features['County_Name'] = label_encoder.fit_transform(features['County_Name'])

# Binarize target variable
# The target variable here is a continuous value, but we binarize it to treat it as a classification problem
median_value = df[target].median()
binary_target = (df[target] > median_value).astype(int)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, df[target], test_size=0.2, random_state=42)

X_train['Social_Vulnerability_Index_(SVI)'].head(10)  # This confirms the numerical values and names of the Social Vulnerability Index (SVI)
X_train['Percent_Hispanic'].head(10)  # These values represent the percentage of Hispanic

import xgboost as xgb
# Apply XGBoost and calculate feature importance
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)

# Extract feature importance
xgb_importances = xgb_model.feature_importances_
xgb_indices = xgb_importances.argsort()[::-1]
xgb_top_features = [(features.columns[i], xgb_importances[i]) for i in xgb_indices[:5]]

# Display results
print("XGBoost Top 5 Features:")
for feature, importance in xgb_top_features:
    print(f"{feature}: {importance:.6f}")

# Calculate feature importance with Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(features, binary_target)

# Extract feature importance
importances = model.feature_importances_
feature_importances = pd.DataFrame({'Feature': features.columns, 'Importance': importances})
feature_importances.sort_values(by='Importance', ascending=False, inplace=True)

# Display results
print("Random Forest Top 5 Important Features:")
print(feature_importances.head(5))

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Test multiple alpha values
alphas = [0.1, 0.01, 0.001, 0.0001]
results = []

# Use the defined features and df[target]
X = features
y = df[target]

for alpha in alphas:
    lasso = Lasso(alpha=alpha, random_state=42)
    pipeline = Pipeline([('scaler', StandardScaler()), ('lasso', lasso)])
    pipeline.fit(X, y)
    
    # Get features with non-zero coefficients
    lasso_coef = pipeline.named_steps['lasso'].coef_
    lasso_importances = pd.Series(lasso_coef, index=features.columns)
    non_zero_importances = lasso_importances[lasso_importances != 0].sort_values(ascending=False)
    
    results.append((alpha, non_zero_importances))

# Display results
for alpha, importances in results:
    print(f"Alpha: {alpha}")
    print("LASSO Feature Importances:")
    print(importances.head(5))
    print('-' * 40)

from sklearn.model_selection import GridSearchCV

# Set up Grid Search
param_grid = {'lasso__alpha': [0.1, 0.01, 0.001, 0.0001]}
lasso = Lasso(random_state=42)
pipeline = Pipeline([('scaler', StandardScaler()), ('lasso', lasso)])
grid_search = GridSearchCV(pipeline, param_grid, cv=5)

# Perform Grid Search
grid_search.fit(X, y)

# Get the best parameter
best_alpha = grid_search.best_params_['lasso__alpha']
print(f"Optimal alpha value: {best_alpha}")

# Train LASSO with the optimal alpha value
best_lasso = Lasso(alpha=best_alpha, random_state=42)
pipeline = Pipeline([('scaler', StandardScaler()), ('lasso', best_lasso)])
pipeline.fit(X, y)

# Get features with non-zero coefficients
lasso_coef = pipeline.named_steps['lasso'].coef_
lasso_importances = pd.Series(lasso_coef, index=features.columns)
lasso_importances = lasso_importances[lasso_importances != 0].sort_values(ascending=False)

# Display results
print("LASSO Feature Importances with Optimal Alpha:")
print(lasso_importances.head(5))

from sklearn.svm import SVC

# Create SVM model
svm = SVC(kernel="linear", random_state=42)
svm.fit(X, binary_target)

# Get coefficients from the model
coefficients = svm.coef_[0]

# Create a DataFrame for features and their coefficients
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients
})

# Rank by absolute value of coefficients
feature_importance['Abs_Coefficient'] = feature_importance['Coefficient'].abs()
feature_importance = feature_importance.sort_values(by='Abs_Coefficient', ascending=False)

# Select top 5 features
top_5_features = feature_importance.head(5).copy()

# Add ranking column
top_5_features['Ranking'] = range(1, 6)

# Drop unnecessary columns
top_5_features = top_5_features.drop(columns=['Abs_Coefficient'])

print("SVM Top 5 Features and Ranking, Coefficients:")
for index, row in top_5_features.iterrows():
    print(f"{row['Feature']}: Ranking {row['Ranking']}, Coefficient {row['Coefficient']:.6f}")

from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import mutual_info_classif

# mi represents mutual information
# This calculates the dependency between features and target variable
# This function works for both continuous and categorical features
mi = mutual_info_classif(X, binary_target)
mi_series = pd.Series(mi, index=features.columns)
mi_series.sort_values(ascending=False, inplace=True)

print("Naive Bayes (Mutual Information) Top Features:")
print(mi_series.head(5))

from scipy.stats import spearmanr
# Spearman's Correlation
spearman_scores = []
for feature in features.columns:
    coef, p = spearmanr(features[feature], df[target])
    spearman_scores.append((feature, coef, p))

spearman_scores.sort(key=lambda x: abs(x[1]), reverse=True)
spearman_top_features = spearman_scores[:5]

# Display results
print("Spearman's Correlation Top 5 Features:")
for feature, coef, p in spearman_top_features:
    print(f"{feature}: {coef:.6f} (p-value: {p:.6e})")

from scipy.stats import kendalltau
# Calculate Kendall's Tau and p-values, and create a DataFrame
kendall_results = []
for column in X.columns:
    tau, p_value = kendalltau(X[column], y)
    kendall_results.append((column, tau, p_value))

# Convert to DataFrame
kendall_df = pd.DataFrame(kendall_results, columns=['Feature', 'Kendall_Tau', 'P_value'])

# Sort by absolute value of Kendall's Tau (closer to 1 indicates stronger correlation)
kendall_df['Abs_Tau'] = kendall_df['Kendall_Tau'].abs()
sorted_kendall_df = kendall_df.sort_values(by='Abs_Tau', ascending=False)

# Filter top 5 features
top_kendall_features = sorted_kendall_df.head(5)

print("Kendall's Tau Top 5 Features and P-values:")
for index, row in top_kendall_features.iterrows():
    print(f"{row['Feature']}: Kendall's Tau {row['Kendall_Tau']:.6f}, P-value {row['P_value']:.6e}")