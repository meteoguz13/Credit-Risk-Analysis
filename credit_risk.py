import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import shap
from sklearn.metrics import precision_score, recall_score, f1_score

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)


df = pd.read_csv(r"/Data/cs-training.csv")

df = df.drop(columns=["Unnamed: 0"])
df_tr = df.copy()

# age=0 sil (veri hatası) + geliri TR'ye ölçekle
df_tr = df_tr[df_tr["age"] >= 18]
df_tr["MonthlyIncome"] = df_tr["MonthlyIncome"] * 10.83


def check_df(dataframe, head=5):
    print("##############_Shape_##############")
    print(dataframe.shape)
    print("##############_Dtypes_##############")
    print(dataframe.dtypes)
    print("##############_Head_##############")
    print(dataframe.head(head))
    print("##############_Tail_##############")
    print(dataframe.tail(head))
    print("##############_Null_##############")
    print(dataframe.isnull().sum())
    print("##############_Describe_##############")
    print(dataframe.describe([0, 0.05, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99, 1]).T)
#check_df(df)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and df[col].dtypes in ["int64", "float64"]]
    cat_but_car = [col for col in df.columns if df[col].nunique() > car_th and str(df[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car, num_but_cat
cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, cat_th = 10, car_th=20)


def num_summary(dataframe, numeric_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numeric_col].describe(quantiles).T)
    if plot:
        dataframe[numeric_col].hist(bins=50)
        plt.xlabel(numeric_col)
        plt.title(numeric_col)
        plt.show(block=True)
    print("#####################################")
#for col in num_cols:
#    num_summary(df, col, True)



# ÖNCE SPLIT (ham haliyle - temizlik henüz yok)
X = df_tr.drop(columns=["SeriousDlqin2yrs"])
y = df_tr["SeriousDlqin2yrs"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(X_train.shape, X_test.shape, y_train.mean(), y_test.mean())

# === Train'den öğren ===
income_floor = 33030
income_cap = X_train["MonthlyIncome"].quantile(0.99)
realestate_cap = X_train["NumberRealEstateLoansOrLines"].quantile(0.99)
revol_cap = X_train["RevolvingUtilizationOfUnsecuredLines"].quantile(0.99)
opencredit_cap = X_train["NumberOfOpenCreditLinesAndLoans"].quantile(0.99)
debt_median = X_train.loc[X_train["DebtRatio"] <= 2, "DebtRatio"].median()

gelir_capli = X_train["MonthlyIncome"].copy()
gelir_capli[gelir_capli < income_floor] = income_floor
gelir_capli[gelir_capli > income_cap] = income_cap
yas_gruplari = pd.cut(X_train["age"], bins=[18, 30, 45, 60, 120])
income_medians = gelir_capli.groupby(yas_gruplari, observed=False).median()


def temizle(X):
    X = X.copy()
    # Gelir: taban + üst capping
    X.loc[X["MonthlyIncome"] < income_floor, "MonthlyIncome"] = income_floor
    X.loc[X["MonthlyIncome"] > income_cap, "MonthlyIncome"] = income_cap
    # Capping'ler (train eşikleriyle)
    X.loc[X["NumberRealEstateLoansOrLines"] > realestate_cap, "NumberRealEstateLoansOrLines"] = realestate_cap
    X.loc[X["RevolvingUtilizationOfUnsecuredLines"] > revol_cap, "RevolvingUtilizationOfUnsecuredLines"] = revol_cap
    X.loc[X["NumberOfOpenCreditLinesAndLoans"] > opencredit_cap, "NumberOfOpenCreditLinesAndLoans"] = opencredit_cap
    # DebtRatio bozuk değer
    X.loc[X["DebtRatio"] > 2, "DebtRatio"] = debt_median
    # Bağımlı eksik
    X["NumberOfDependents"] = X["NumberOfDependents"].fillna(0)
    # Gelir eksik -> yaş grubu medyanı (train'den)
    gruplar = pd.cut(X["age"], bins=[18, 30, 45, 60, 120])
    X["MonthlyIncome"] = X["MonthlyIncome"].fillna(gruplar.map(income_medians))
    # Feature: toplam gecikme + gecikme var/yok
    gecikme_toplam = (X["NumberOfTime30-59DaysPastDueNotWorse"]
                      + X["NumberOfTime60-89DaysPastDueNotWorse"]
                      + X["NumberOfTimes90DaysLate"])
    X["toplam_gecikme"] = gecikme_toplam
    X["gecikme_var"] = (gecikme_toplam > 0).astype(int)
    X = X.drop(columns=["NumberOfTime30-59DaysPastDueNotWorse",
                        "NumberOfTime60-89DaysPastDueNotWorse",
                        "NumberOfTimes90DaysLate"])
    return X


X_train = temizle(X_train)
X_test = temizle(X_test)


# === LOGISTIC (baseline) ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(class_weight="balanced", max_iter=1000)
log_model.fit(X_train_scaled, y_train)
y_prob_log = log_model.predict_proba(X_test_scaled)[:, 1]
print("LOGISTIC ROC-AUC:", roc_auc_score(y_test, y_prob_log))

# === XGBoost + GridSearch ===
oran = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(scale_pos_weight=oran, random_state=42, eval_metric="logloss")
param_grid = {"max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1], "n_estimators": [100, 200]}
grid = GridSearchCV(xgb, param_grid, scoring="roc_auc", cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
print("En iyi parametreler:", grid.best_params_)

best_xgb = grid.best_estimator_
y_prob_best = best_xgb.predict_proba(X_test)[:, 1]
y_pred_best = best_xgb.predict(X_test)
print("XGBoost TEST ROC-AUC:", roc_auc_score(y_test, y_prob_best))
print(classification_report(y_test, y_pred_best))


# === SHAP ===
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, show=False)
plt.show(block=True)

# === Eşik tablosu ===
for esik in np.arange(0.1, 0.95, 0.1):
    tahmin = (y_prob_best >= esik).astype(int)
    p = precision_score(y_test, tahmin)
    r = recall_score(y_test, tahmin)
    f = f1_score(y_test, tahmin)
    print(f"Esik {esik:.1f} -> Precision: {p:.2f}  Recall: {r:.2f}  F1: {f:.2f}")

# === Kaydet ===
import joblib
joblib.dump(best_xgb, "kredi_model.pkl")
joblib.dump(list(X_train.columns), "feature_isimleri.pkl")
























