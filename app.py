# Import libraries
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend so plots don't block execution
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("car data.csv")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
df.info()

# EDA
print(df.tail(10))
print(df.describe())
print(df.dtypes)
print("Nulls:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# Correlation heatmap (numeric only) — shown in-memory only
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.tight_layout()
plt.close()

# Missing value handling
# Fill categorical columns with mode
for col in ["Fuel Type", "Transmission", "Owner", "Seller Type",
            "Drivetrain", "Color", "Location"]:
    if col in df.columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Engine / Max Power / Max Torque are stored as strings like "1197 CC", "82 bhp"
# Extract numeric part
def extract_numeric(series):
    return pd.to_numeric(series.astype(str).str.extract(r"([\d.]+)")[0],
                         errors="coerce")

for col in ["Engine", "Max Power", "Max Torque"]:
    if col in df.columns:
        df[col] = extract_numeric(df[col])

# Fill remaining numeric nulls with median
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)

print("Nulls after filling:\n", df.isnull().sum())

# Outlier treatment on target column
target = "Price"

plt.figure(figsize=(8, 5))
sns.boxplot(x=df[target])
plt.tight_layout()
plt.close()

Q1 = df[target].quantile(0.25)
Q3 = df[target].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[
    (df[target] >= lower) &
    (df[target] <= upper)
]
print("Shape after outlier removal:", df.shape)

# Drop columns with too many unique values (high-cardinality)
# 'Make' and 'Model' have many unique values; drop 'Model', keep 'Make'
if "Model" in df.columns:
    df.drop("Model", axis=1, inplace=True)

# Encoding
encoder = LabelEncoder()
for col in df.select_dtypes(include="object"):
    df[col] = encoder.fit_transform(df[col])

# Feature Engineering
df["Car_Age"] = 2025 - df["Year"]
df.drop("Year", axis=1, inplace=True)

# Prepare Features
# XGBoost is tree-based — no feature scaling needed
X = df.drop(target, axis=1)
y = df[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost Regressor
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE  : {mae:.4f}")

mse = mean_squared_error(y_test, y_pred)
print(f"MSE  : {mse:.4f}")

rmse = np.sqrt(mse)
print(f"RMSE : {rmse:.4f}")

r2 = r2_score(y_test, y_pred)
print(f"R²   : {r2:.4f}")

n = X_test.shape[0]
p = X_test.shape[1]
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
print(f"Adj R²: {adj_r2:.4f}")

scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print(f"CV R² scores : {scores}")
print(f"CV R² mean   : {scores.mean():.4f}")

# Compare Actual vs Predicted
result = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
print(result.head(10))

# Scatter Plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.tight_layout()
plt.close()

