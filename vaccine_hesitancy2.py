import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.feature_selection import chi2
from scipy.stats import spearmanr



from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import mutual_info_classif

# データ(csvファイル)の読み込み
df = pd.read_csv('/Users/yasudayuuya/AI_ethics/Vaccine_Hesitancy_for_COVID-19__County_and_local_estimates_20250227.csv')
df.head()
# データの前処理
# 欠損値の確認
df.isnull().sum()
# 欠損値のあった列のみ出力
df.isnull().sum()[df.isnull().sum()>0]
# 欠損値のある行を削除
df = df.dropna()
df.isnull().sum()
df
# ターゲット変数設定（ここではTotal PrevalenceのRateを予測ターゲットとします）
target = 'Estimated strongly hesitant'
df[target] = df[target].astype(float)  # 文字列を浮動小数点数に変換
# 特徴量設定
features = df.drop(columns=['FIPS Code', 'State', 'Estimated hesitant', 'Estimated hesitant or unsure', 'SVI Category', 'CVAC Level Of Concern', 'Percent adults fully vaccinated against COVID-19 (as of 6/10/21)',  'Geographical Point', 'State Code', 'County Boundary', 'State Boundary', target])

# 説明変数のカラム名を表示
#　表形式で表示
features.columns
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# 特徴量のカラム名から特殊文字を削除
# 今回削除される文字は、[ ] % , です
features.columns = features.columns.str.replace('[\[\]%, ]', '_', regex=True)


# カテゴリ変数をラベルエンコーディング
# これによって、カテゴリ変数を数値に変換することができます
# ここにあるcounty_nameは数値に変換すると
label_encoder = LabelEncoder()
for col in features.select_dtypes(include=['object']).columns:
    features[col] = label_encoder.fit_transform(features[col])

# County_Nameをエンコーディング
features['County_Name'] = label_encoder.fit_transform(features['County_Name'])

# ターゲット変数を2値化
# 今回のターゲット変数は連続値ですが、2値化して分類問題として扱います
median_value = df[target].median()
binary_target = (df[target] > median_value).astype(int)

# データセットの分割
X_train, X_test, y_train, y_test = train_test_split(features, df[target], test_size=0.2, random_state=42)

X_train['Social_Vulnerability_Index_(SVI)'].head(10) # これによって、Social Vulnerability Index (SVI)の数値と名前の対応を確認できます
X_train['Percent_Hispanic'].head(10) # この数値は、ヒスパニックの割合を示しています
# XGBoostの適用と特徴量重要度の計算
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)

# 特徴量重要度の抽出
xgb_importances = xgb_model.feature_importances_
xgb_indices = xgb_importances.argsort()[::-1]
xgb_top_features = [(features.columns[i], xgb_importances[i]) for i in xgb_indices[:5]]

# 結果の表示
print("XGBoost Top 5 Features:")
for feature, importance in xgb_top_features:
    print(f"{feature}: {importance:.6f}")

# DataFrame表示オプションの調整
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

# 各手法の結果を縦一列にまとめる
results = []

# XGBoostの結果を追加
for i, (feature, importance) in enumerate(xgb_top_features, start=1):
    results.append({'Method': 'XGBoost', 'Rank': i, 'Feature': feature, 'Importance': importance})

# Chi-squaredの結果を追加
for i, (feature, score, p) in enumerate(chi2_top_features, start=1):
    results.append({'Method': 'Chi-squared', 'Rank': i, 'Feature': feature, 'Importance': score, 'P-value': p})

# Spearmanの結果を追加
for i, (feature, coef, p) in enumerate(spearman_top_features, start=1):
    results.append({'Method': 'Spearman', 'Rank': i, 'Feature': feature, 'Importance': coef, 'P-value': p})

# 結果をDataFrameに変換
comparison_df = pd.DataFrame(results)

# 表形式でインデックスなしで表示
print(comparison_df.to_string(index=False))
# ランダムフォレストの特徴量重要度の計算
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(features, binary_target)

# 特徴量重要度の抽出
importances = model.feature_importances_
feature_importances = pd.DataFrame({'Feature': features.columns, 'Importance': importances})
feature_importances.sort_values(by='Importance', ascending=False, inplace=True)

# 結果の表示
print("Random Forest Top 5 Important Features:")
print(feature_importances.head(5))
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 複数のアルファ値を試してみる
alphas = [0.1, 0.01, 0.001, 0.0001]
results = []

# 定義済みの features と df[target] を使用
X = features
y = df[target]

for alpha in alphas:
    lasso = Lasso(alpha=alpha, random_state=42)
    pipeline = Pipeline([('scaler', StandardScaler()), ('lasso', lasso)])
    pipeline.fit(X, y)
    
    # 非ゼロの係数を持つ特徴量を取得
    lasso_coef = pipeline.named_steps['lasso'].coef_
    lasso_importances = pd.Series(lasso_coef, index=features.columns)
    non_zero_importances = lasso_importances[lasso_importances != 0].sort_values(ascending=False)
    
    results.append((alpha, non_zero_importances))

# 結果を表示
for alpha, importances in results:
    print(f"アルファ: {alpha}")
    print("LASSO特徴量の重要度:")
    print(importances.head(5))
    print('-' * 40)
from sklearn.model_selection import GridSearchCV

# グリッドサーチの設定
param_grid = {'lasso__alpha': [0.1, 0.01, 0.001, 0.0001]}
lasso = Lasso(random_state=42)
pipeline = Pipeline([('scaler', StandardScaler()), ('lasso', lasso)])
grid_search = GridSearchCV(pipeline, param_grid, cv=5)

# グリッドサーチの実行
grid_search.fit(X, y)

# 最適なパラメータを取得
best_alpha = grid_search.best_params_['lasso__alpha']
print(f"最適なアルファ値: {best_alpha}")

# 最適なアルファ値を用いたLASSOのトレーニング
best_lasso = Lasso(alpha=best_alpha, random_state=42)
pipeline = Pipeline([('scaler', StandardScaler()), ('lasso', best_lasso)])
pipeline.fit(X, y)

# 非ゼロの係数を持つ特徴量を取得
lasso_coef = pipeline.named_steps['lasso'].coef_
lasso_importances = pd.Series(lasso_coef, index=features.columns)
lasso_importances = lasso_importances[lasso_importances != 0].sort_values(ascending=False)

# 結果を表示
print("最適なアルファ値でのLASSO特徴量の重要度:")
print(lasso_importances.head(5))
from sklearn.svm import SVC

# SVMモデルの作成
svm = SVC(kernel="linear", random_state=42)
svm.fit(X, binary_target)

# モデルから係数を取得
coefficients = svm.coef_[0]

# 特徴量とその係数をデータフレームにまとめる
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients
})

# 係数の絶対値でランキング
feature_importance['Abs_Coefficient'] = feature_importance['Coefficient'].abs()
feature_importance = feature_importance.sort_values(by='Abs_Coefficient', ascending=False)

# 上位5つの特徴量を選択
top_5_features = feature_importance.head(5).copy()

# ランキング列を追加
top_5_features['Ranking'] = range(1, 6)

# 不要な列を削除
top_5_features = top_5_features.drop(columns=['Abs_Coefficient'])

print("SVM トップ5特徴量とランキング、係数:")
for index, row in top_5_features.iterrows():
    print(f"{row['Feature']}: ランキング {row['Ranking']}, 係数 {row['Coefficient']:.6f}")
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import mutual_info_classif

# miは相互情報量を表す
# これによって、特徴量とターゲット変数の間の依存関係を計算できます
# この関数は、特徴量が連続値でもカテゴリ値でも機能します
mi = mutual_info_classif(X, binary_target)
mi_series = pd.Series(mi, index=features.columns)
mi_series.sort_values(ascending=False, inplace=True)

print("Naive Bayes (Mutual Information) Top Features:")
print(mi_series.head(5))

# Spearman's Correlation
spearman_scores = []
for feature in features.columns:
    coef, p = spearmanr(features[feature], df[target])
    spearman_scores.append((feature, coef, p))

spearman_scores.sort(key=lambda x: abs(x[1]), reverse=True)
spearman_top_features = spearman_scores[:5]

# 結果の表示
print("Spearman's Correlation Top 5 Features:")
for feature, coef, p in spearman_top_features:
    print(f"{feature}: {coef:.6f} (p-value: {p:.6e})")
from scipy.stats import kendalltau

# Kendall's Tauとp値を計算し、DataFrameを作成
kendall_results = []
for column in X.columns:
    tau, p_value = kendalltau(X[column], y)
    kendall_results.append((column, tau, p_value))

# DataFrameに変換
kendall_df = pd.DataFrame(kendall_results, columns=['Feature', 'Kendall_Tau', 'P_value'])

# Kendall's Tauの絶対値でソート（1に近いほど強い相関）
kendall_df['Abs_Tau'] = kendall_df['Kendall_Tau'].abs()
sorted_kendall_df = kendall_df.sort_values(by='Abs_Tau', ascending=False)

# 上位5つの特徴量をフィルタリング
top_kendall_features = sorted_kendall_df.head(5)

print("Kendall's Tau 上位5特徴量とP値:")
for index, row in top_kendall_features.iterrows():
    print(f"{row['Feature']}: Kendall's Tau {row['Kendall_Tau']:.6f}, P-value {row['P_value']:.6e}")
