from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load as a Bunch object, then turn into a DataFrame for easy exploration
housing = fetch_california_housing(as_frame=True)

X = housing.data          # features
y = housing.target        # median house value
df = housing.frame        # features + target in one table

print("Shape (rows, columns):", df.shape)
print("\nFeature names:", list(X.columns))
print("\nTarget name:", housing.target_names)
print("\nFirst 5 rows:")
print(df.head())
print("\nBasic stats:")
print(df.describe())
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nDataset description (short):")
print(housing.DESCR[:600])  # first part of the docs


# --- Step 2: Feature engineering ---
# Create a ratio: more rooms per person often means more living space
df["RoomsPerPerson"] = df["AveRooms"] / df["AveOccup"]

# Update X to include the new feature (still no target in X)
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

print("New column sample:")
print(df[["AveRooms", "AveOccup", "RoomsPerPerson"]].head(10))
print("\nRoomsPerPerson stats:")
print(df["RoomsPerPerson"].describe())
print("\nUpdated feature list:", list(X.columns))
print("X shape:", X.shape)


from sklearn.model_selection import train_test_split

# --- Step 3: Train/test split ---
# 80% train, 20% test is a common default for this size of dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,  # reproducible split
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print("\nTrain target mean:", round(y_train.mean(), 3))
print("Test target mean:", round(y_test.mean(), 3))


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# --- Step 4: Baseline Linear Regression ---
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\n--- Step 4: Linear Regression ---")
print("Model fitted. Test predictions shape:", y_pred_lr.shape)
print("Sample predictions (first 5):", np.round(y_pred_lr[:5], 3))


# --- Step 5: Evaluate with MAE, RMSE, R2 ---
def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{label}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2:   {r2:.4f}")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


print("\n--- Step 5: Linear Regression metrics ---")
lr_metrics = evaluate(y_test, y_pred_lr, "Linear Regression (test)")


# --- Step 6: Random Forest + compare ---
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n--- Step 6: Random Forest vs Linear Regression ---")
rf_metrics = evaluate(y_test, y_pred_rf, "Random Forest (test)")
print("\nComparison (lower MAE/RMSE better; higher R2 better):")
print(f"  {'Model':<20} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
print(f"  {'Linear Regression':<20} {lr_metrics['MAE']:8.4f} {lr_metrics['RMSE']:8.4f} {lr_metrics['R2']:8.4f}")
print(f"  {'Random Forest':<20} {rf_metrics['MAE']:8.4f} {rf_metrics['RMSE']:8.4f} {rf_metrics['R2']:8.4f}")


# --- Step 7: Hyperparameter tuning with GridSearchCV ---
# Search only on train; evaluate the best model once on test
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 2],
}
grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
)
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_
y_pred_best = best_rf.predict(X_test)

print("\n--- Step 7: GridSearchCV (Random Forest) ---")
print("Best params:", grid.best_params_)
print("Best CV MAE:", round(-grid.best_score_, 4))
best_metrics = evaluate(y_test, y_pred_best, "Tuned Random Forest (test)")
print("\nFinal comparison:")
print(f"  {'Model':<24} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
print(f"  {'Linear Regression':<24} {lr_metrics['MAE']:8.4f} {lr_metrics['RMSE']:8.4f} {lr_metrics['R2']:8.4f}")
print(f"  {'Random Forest':<24} {rf_metrics['MAE']:8.4f} {rf_metrics['RMSE']:8.4f} {rf_metrics['R2']:8.4f}")
print(f"  {'Tuned Random Forest':<24} {best_metrics['MAE']:8.4f} {best_metrics['RMSE']:8.4f} {best_metrics['R2']:8.4f}")


# --- Step 8: Feature importance ---
importances = pd.Series(
    best_rf.feature_importances_,
    index=X.columns,
).sort_values(ascending=False)

print("\n--- Step 8: Feature importance (tuned Random Forest) ---")
print(importances.round(4).to_string())
print("\nTop feature:", importances.index[0], f"({importances.iloc[0]:.4f})")