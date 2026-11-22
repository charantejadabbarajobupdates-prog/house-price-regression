import numpy as np, pandas as pd, matplotlib, joblib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

out = []
def log(s=""):
    print(s); out.append(str(s))

# ---------- 1. Load and clean ----------
df = pd.read_csv("data/raw/train.csv")
if "Id" in df.columns:
    df = df.drop(columns="Id")
n0 = len(df)
# Two known outliers (huge houses sold very cheaply), recommended by the dataset author
df = df[~((df.GrLivArea > 4000) & (df.SalePrice < 300000))]
log(f"Rows: {n0:,} -> {len(df):,} after removing {n0 - len(df)} outliers | columns: {df.shape[1]}")
log(f"SalePrice skew: {df.SalePrice.skew():.2f} -> after log1p: {np.log1p(df.SalePrice).skew():.2f}")

y = np.log1p(df["SalePrice"])          # model the log price (fixes skew, errors become relative)
X = df.drop(columns="SalePrice").copy()

# ---------- 2. Feature engineering ----------
def add_features(X):
    X = X.copy()
    z = lambda c: X[c].fillna(0)
    X["TotalSF"] = z("TotalBsmtSF") + z("1stFlrSF") + z("2ndFlrSF")
    X["HouseAge"] = (X["YrSold"] - X["YearBuilt"]).clip(lower=0)
    X["RemodAge"] = (X["YrSold"] - X["YearRemodAdd"]).clip(lower=0)
    X["TotalBath"] = z("FullBath") + 0.5 * z("HalfBath") + z("BsmtFullBath") + 0.5 * z("BsmtHalfBath")
    X["PorchSF"] = z("OpenPorchSF") + z("EnclosedPorch") + z("3SsnPorch") + z("ScreenPorch") + z("WoodDeckSF")
    X["HasGarage"] = (z("GarageArea") > 0).astype(int)
    X["HasPool"] = (z("PoolArea") > 0).astype(int)
    X["HasFireplace"] = (z("Fireplaces") > 0).astype(int)
    X["WasRemodeled"] = (X["YearRemodAdd"] != X["YearBuilt"]).astype(int)
    X["MSSubClass"] = X["MSSubClass"].astype(str)   # a category coded as a number
    return X

X = add_features(X)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
num = Xtr.select_dtypes(include="number").columns.tolist()
cat = Xtr.select_dtypes(exclude="number").columns.tolist()
skewed = [c for c in num if Xtr[c].skew() > 0.75 and Xtr[c].min() >= 0]   # chosen on train only
normal = [c for c in num if c not in skewed]
log(f"Numeric: {len(num)} ({len(skewed)} skewed, log-transformed) | categorical: {len(cat)}")

def make_prep():
    return ColumnTransformer([
        ("skew", Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                           ("sc", StandardScaler())]), skewed),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), normal),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="None")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])

kf = KFold(5, shuffle=True, random_state=42)
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))

# ---------- 3. Models (5-fold CV on train only) ----------
log("\n[1] 5-fold CV RMSE on log(SalePrice) (lower is better)")
ols = Pipeline([("prep", make_prep()), ("model", LinearRegression())])
s = -cross_val_score(ols, Xtr, ytr, cv=kf, scoring="neg_root_mean_squared_error")
log(f"{'OLS (no regularisation)':26s} {s.mean():.4f} +- {s.std():.4f}")

ridge = GridSearchCV(Pipeline([("prep", make_prep()), ("model", Ridge())]),
                     {"model__alpha": [1, 3, 10, 30, 100, 300]},
                     scoring="neg_root_mean_squared_error", cv=kf, n_jobs=-1).fit(Xtr, ytr)
log(f"{'Ridge':26s} {-ridge.best_score_:.4f} | best alpha {ridge.best_params_['model__alpha']}")

lasso = GridSearchCV(Pipeline([("prep", make_prep()), ("model", Lasso(max_iter=50000))]),
                     {"model__alpha": [0.0002, 0.0005, 0.001, 0.003, 0.01]},
                     scoring="neg_root_mean_squared_error", cv=kf, n_jobs=-1).fit(Xtr, ytr)
log(f"{'Lasso':26s} {-lasso.best_score_:.4f} | best alpha {lasso.best_params_['model__alpha']}")

xgb = RandomizedSearchCV(
    Pipeline([("prep", make_prep()), ("model", XGBRegressor(random_state=42))]),
    {"model__n_estimators": [400, 800, 1200], "model__learning_rate": [0.01, 0.03, 0.05],
     "model__max_depth": [2, 3, 4], "model__subsample": [0.7, 0.85, 1.0],
     "model__colsample_bytree": [0.4, 0.6, 0.8], "model__min_child_weight": [1, 3, 5],
     "model__reg_lambda": [1, 5, 10]},
    n_iter=15, scoring="neg_root_mean_squared_error", cv=kf, random_state=42, n_jobs=-1).fit(Xtr, ytr)
log(f"{'XGBoost (tuned)':26s} {-xgb.best_score_:.4f}")

# ---------- 4. Test set ----------
best_lasso, best_ridge, best_xgb = lasso.best_estimator_, ridge.best_estimator_, xgb.best_estimator_
preds = {"Ridge": best_ridge.predict(Xte), "Lasso": best_lasso.predict(Xte), "XGBoost": best_xgb.predict(Xte)}
preds["Lasso + XGBoost blend"] = (preds["Lasso"] + preds["XGBoost"]) / 2
actual = np.expm1(yte)
log("\n[2] TEST set (price metrics in dollars)")
for name, p in preds.items():
    d = np.expm1(p)
    log(f"{name:22s} RMSE(log) {rmse(yte, p):.4f} | R2 {r2_score(yte, p):.3f} | "
        f"MAE ${mean_absolute_error(actual, d):,.0f} | MAPE {np.mean(np.abs(d - actual) / actual):.1%}")
log(f"Baseline (predict the median price): MAE ${mean_absolute_error(actual, np.full(len(actual), np.median(np.expm1(ytr)))):,.0f}")

# ---------- 5. Residual analysis on the blend ----------
p = preds["Lasso + XGBoost blend"]
res = yte.values - p
d = np.expm1(p)
log("\n[3] Residual analysis (log scale)")
log(f"Mean residual {res.mean():+.4f} | std {res.std():.4f} | skew {stats.skew(res):.2f} | kurtosis {stats.kurtosis(res):.2f}")
band = pd.qcut(actual, 4, labels=["Q1 cheapest", "Q2", "Q3", "Q4 most expensive"])
by = pd.DataFrame({"band": band, "ape": np.abs(d - actual) / actual, "res": res}).groupby("band", observed=True).agg(
    MAPE=("ape", "mean"), mean_residual=("res", "mean"))
log("Error by price band:\n" + by.round(3).to_string())
worst = pd.DataFrame({"actual": actual.round(0), "predicted": d.round(0), "error_%": ((d - actual) / actual * 100).round(1)},
                     index=yte.index).reindex(pd.Series(np.abs(res), index=yte.index).sort_values(ascending=False).index[:5])
log("5 largest errors:\n" + worst.to_string())

fig, ax = plt.subplots(2, 2, figsize=(12, 9))
ax[0, 0].scatter(actual, d, s=12, alpha=0.6); lim = [actual.min(), actual.max()]
ax[0, 0].plot(lim, lim, "r--"); ax[0, 0].set_title("Predicted vs actual price ($)")
ax[0, 0].set_xlabel("actual"); ax[0, 0].set_ylabel("predicted")
ax[0, 1].scatter(p, res, s=12, alpha=0.6); ax[0, 1].axhline(0, color="r", ls="--")
ax[0, 1].set_title("Residuals vs fitted (log scale)"); ax[0, 1].set_xlabel("fitted")
ax[1, 0].hist(res, bins=30, color="steelblue"); ax[1, 0].set_title("Residual distribution")
stats.probplot(res, dist="norm", plot=ax[1, 1]); ax[1, 1].set_title("Q-Q plot of residuals")
plt.tight_layout(); plt.savefig("reports/figures/residual_analysis.png", dpi=150); plt.close()

# ---------- 6. What drives price (Lasso coefficients) ----------
names = best_lasso.named_steps["prep"].get_feature_names_out()
coef = pd.Series(best_lasso.named_steps["model"].coef_, index=names)
log(f"\n[4] Lasso kept {(coef != 0).sum()} of {len(coef)} features (the rest were set to zero)")
log("Top positive:\n" + coef.sort_values(ascending=False).head(8).round(3).to_string())
log("Top negative:\n" + coef.sort_values().head(5).round(3).to_string())
top = coef.abs().sort_values(ascending=False).head(15).index
coef[top].sort_values().plot.barh(figsize=(8, 6), title="Top 15 Lasso coefficients (log price)")
plt.tight_layout(); plt.savefig("reports/figures/lasso_coefficients.png", dpi=150); plt.close()

joblib.dump({"lasso": best_lasso, "xgb": best_xgb}, "models/house_price_models.joblib")
open("reports/results.txt", "w", encoding="utf-8").write("\n".join(out))
