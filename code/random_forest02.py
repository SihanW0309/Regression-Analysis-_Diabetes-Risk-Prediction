import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

column_threshold = 0.05
row_threshold = 0.2
lasso_cv = 10

base_dir = os.path.dirname(__file__)
folder_path_Y = os.path.join(base_dir, "Dataset/Y")
df_y = None

for filename in os.listdir(folder_path_Y):
    if filename.endswith(".xpt"):
        file_path = os.path.join(folder_path_Y, filename)
        df = pd.read_sas(file_path, format="xport")

        if "SEQN" not in df.columns:
            continue
        if df_y is None:
            df_y = df
        else:
            df_y = pd.merge(
                df_y, df, on="SEQN", how="outer", suffixes=("_left", "_right")
            )

df_y["Value"] = 0
df_y.Value[df_y.DIQ010 == 3] = 1
df_y.Value[df_y.DIQ010 == 1] = 2

df_y["feat_selection"] = 0
df_y.feat_selection[df_y.DIQ010 == 1] = 1

folder_path_Questionnaire_Data = os.path.join(base_dir, "Dataset/X/Questionnaire Data")
folder_path_Laboratory_Data = os.path.join(base_dir, "Dataset/X/Laboratory Data")
folder_path_Demo_Data = os.path.join(
    base_dir, "Dataset/X/Demographic Variables and Sample Weights"
)
folder_path_Examination_Data = os.path.join(base_dir, "Dataset/X/Examination Data")

folder_path_Data = [
    folder_path_Questionnaire_Data,
    folder_path_Laboratory_Data,
    folder_path_Examination_Data,
    folder_path_Demo_Data,
]

df_x = None
columns_to_rename = ["DRDINT", "DRABF", "WTDR2D", "WTDRD1", "WTPH2YR"]

dropped_columns = []
files_to_exclude = []

for folder_path in folder_path_Data:
    for filename in os.listdir(folder_path):
        if filename.endswith(".xpt"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_sas(file_path, format="xport")
            if df.columns.isin(columns_to_rename).any():
                df = df.rename(
                    columns={
                        col: col + filename
                        for col in columns_to_rename
                        if col in df.columns
                    }
                )

            if "SEQN" not in df.columns:
                continue
            df_sub = df[df["SEQN"].isin(df_y["SEQN"])]
            if df_sub.shape[0] > df_y.shape[0]:
                files_to_exclude.extend([filename])
                continue

            df_sub = df_sub.select_dtypes(exclude=["object", "category"])

            missing = df_sub.isna().mean()
            cols_to_drop = missing[missing > column_threshold].index
            df_sub = df_sub.drop(columns=cols_to_drop)
            dropped_columns.extend([filename, cols_to_drop])

            cols_to_drop = df_sub.columns[df_sub.nunique() == 1]
            dropped_columns.extend([filename, cols_to_drop])
            df_sub = df_sub.drop(columns=cols_to_drop)

            if df_x is None:
                df_x = pd.merge(
                    df_y[["SEQN", "Value", "feat_selection"]],
                    df_sub,
                    how="left",
                    on="SEQN",
                    suffixes=("_x", "_y"),
                )
            else:
                df_x = pd.merge(
                    df_x, df_sub, how="left", on="SEQN", suffixes=("_x", "_y")
                )

df_x = df_x.drop_duplicates()

missing = df_x.isna().mean()
cols_to_drop = missing[missing > 0.5].index
df_x = df_x.drop(columns=cols_to_drop)
dropped_columns.extend([filename, cols_to_drop])

threshold = (1 - row_threshold) * df_x.shape[1]
df_x = df_x.dropna(thresh=threshold)
df_x.reset_index(drop=True, inplace=True)

X = df_x.drop(columns=["SEQN", "feat_selection", "Value"])
y = df_x["feat_selection"]

y = pd.DataFrame({"y": y})

plt.figure(figsize=(10, 5))
sns.countplot(x="y", data=y, palette="viridis")
plt.title("Distribution of Diabetes")
plt.xlabel("Diabetes")
plt.ylabel("Count")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

X["Age_Group"] = ["<40" if age < 40 else "≥40" for age in X["RIDAGEYR"]]
under_40_mask = X["Age_Group"] == "<40"

plt.figure(figsize=(10, 5))
sns.countplot(x="Age_Group", data=X, palette="viridis")
plt.title("Distribution of Age Groups")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
X_numeric = X[numeric_cols]

print("1. CORRELATION MATRIX:")
corr = X_numeric.corr().round(2)
print("Checking for correlation coefficients > 0.7 or < -0.7...\n")

high_corr = []
for i in range(len(corr.columns)):
    for j in range(i):
        if abs(corr.iloc[i, j]) > 0.7:
            high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

if high_corr:
    print("High correlations detected:")
    for var1, var2, val in high_corr:
        print(f"  {var1} and {var2}: {val}")
else:
    print("No high correlations detected.")

X_filtered = X[under_40_mask].drop(columns=["Age_Group"])
y_filtered = y[under_40_mask]

missing_counts = X_filtered.isna().sum()

X_dropped = X.drop(columns=["Age_Group"])

X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, y_filtered, test_size=0.2, random_state=1
)

train_weights = compute_sample_weight(class_weight="balanced", y=y_train)

pipeline = Pipeline(
    [
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lassoCV", LassoCV(cv=lasso_cv, random_state=1)),
    ]
)

pipeline.fit(X_train, y_train, lassoCV__sample_weight=train_weights)

lassoCV = pipeline.named_steps["lassoCV"]
sns.set_theme()
plt.plot(lassoCV.alphas_, lassoCV.mse_path_.mean(axis=1), color="green", marker="o")
plt.axvline(lassoCV.alpha_, label=f"Best alpha: {round(lassoCV.alpha_, 3)}")
plt.xlabel("alpha")
plt.ylabel("MSE")
plt.title("LassoCV Feature Selection - MSE")
plt.legend()
plt.grid(True)
plt.show()

alpha = lassoCV.alpha_
coeffs = lassoCV.coef_
features = X_dropped.columns

plt.figure(figsize=(14, 12))
plt.bar(features[np.abs(coeffs > 0.0001)], np.abs(coeffs[np.abs(coeffs > 0.0001)]))
plt.xticks(rotation=75)
plt.grid()
plt.title("LassoCV Feature Selection - Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.ylim(0, 0.05)
plt.grid(True)
plt.show()

X_test_transformed = pipeline.named_steps["scaler"].transform(
    pipeline.named_steps["imp"].transform(X_test)
)

y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE on test data: {mse}")

r2 = r2_score(y_test, y_pred)

print("LassoCV CV n:", lasso_cv)
print("LassoCV alpha:", round(alpha, 4))
print("LassoCV MSE:", round(mse, 4))
print("LassoCV R²:", round(r2, 4))

plt.figure(figsize=(10, 6))
feature_importance = pd.DataFrame({"Feature": features, "Importance": np.abs(coeffs)})
feature_importance = feature_importance.sort_values("Importance", ascending=False)
sns.barplot(x="Importance", y="Feature", data=feature_importance.head(15))
plt.title("Top 15 Feature Importances (Balanced LassoCV)")
plt.tight_layout()
plt.show()

model = SelectFromModel(pipeline.named_steps["lassoCV"], prefit=True)
features = X_dropped.columns[model.get_support()]

df_y = df_x["Value"]
df_x = df_x[features]

n_selected = sum(coeffs != 0)
print(f"Number of features selected by Lasso: {n_selected} out of {len(coeffs)}")

missing = df_x.isna().mean()
cols_to_drop = missing[missing > column_threshold].index
df_x = df_x.drop(columns=cols_to_drop)
dropped_columns.extend([filename, cols_to_drop])

if hasattr(y_filtered, "values"):
    y_filtered = y_filtered.values
if len(y_filtered.shape) > 1:
    y_filtered = y_filtered.ravel()

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
    X_filtered[features],
    y_filtered,
    test_size=0.2,
    random_state=123,
    stratify=y_filtered,
)

imputer = SimpleImputer(strategy="median")
X_train_lr_imputed = imputer.fit_transform(X_train_lr)
X_test_lr_imputed = imputer.transform(X_test_lr)

scaler = StandardScaler()
X_train_lr_scaled = scaler.fit_transform(X_train_lr_imputed)
X_test_lr_scaled = scaler.transform(X_test_lr_imputed)

smote = SMOTE(random_state=123)
X_train_lr_resampled, y_train_lr_resampled = smote.fit_resample(
    X_train_lr_scaled, y_train_lr
)

print("After SMOTE, training set class distribution:")
print(pd.Series(y_train_lr_resampled).value_counts(normalize=True))

log_reg = LogisticRegression(
    max_iter=1000, random_state=123, class_weight="balanced", C=1.0
)

scoring = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
cv_results = cross_validate(
    log_reg, X_train_lr_resampled, y_train_lr_resampled, cv=cv, scoring=scoring
)

print("\nCross-Validation Results (after SMOTE):")
for metric in scoring:
    print(f"{metric.capitalize()} (mean): {np.mean(cv_results['test_' + metric]):.4f}")

log_reg.fit(X_train_lr_resampled, y_train_lr_resampled)

y_pred_lr = log_reg.predict(X_test_lr_scaled)
y_pred_lr_proba = log_reg.predict_proba(X_test_lr_scaled)[:, 1]

print("\nTest Set Results:")
print(f"Accuracy: {accuracy_score(y_test_lr, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test_lr, y_pred_lr, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test_lr, y_pred_lr, zero_division=0):.4f}")
print(f"F1 Score: {f1_score(y_test_lr, y_pred_lr, zero_division=0):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test_lr, y_pred_lr_proba):.4f}")

thresholds = np.arange(0.1, 0.91, 0.1)
best_f1 = 0
best_threshold = 0.5

print("\nTesting different classification thresholds:")
for threshold in thresholds:
    y_pred_lr_threshold = (y_pred_lr_proba >= threshold).astype(int)
    f1 = f1_score(y_test_lr, y_pred_lr_threshold, zero_division=0)
    precision = precision_score(y_test_lr, y_pred_lr_threshold, zero_division=0)
    recall = recall_score(y_test_lr, y_pred_lr_threshold, zero_division=0)

    print(
        f"Threshold: {threshold:.1f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

y_pred_lr_best = (y_pred_lr_proba >= best_threshold).astype(int)

print(f"\nBest threshold: {best_threshold:.2f}")
print("\nTest Set Results with Best Threshold:")
print(f"Accuracy: {accuracy_score(y_test_lr, y_pred_lr_best):.4f}")
print(f"Precision: {precision_score(y_test_lr, y_pred_lr_best, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test_lr, y_pred_lr_best, zero_division=0):.4f}")
print(f"F1 Score: {f1_score(y_test_lr, y_pred_lr_best, zero_division=0):.4f}")

print("\nClassification Report (Best Threshold):")
print(classification_report(y_test_lr, y_pred_lr_best, zero_division=0))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test_lr, y_pred_lr_best)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test_lr, y_pred_lr_proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()

plt.figure(figsize=(8, 6))
precision, recall, _ = precision_recall_curve(y_test_lr, y_pred_lr_proba)
avg_precision = average_precision_score(y_test_lr, y_pred_lr_proba)
plt.plot(
    recall,
    precision,
    color="blue",
    lw=2,
    label=f"Precision-Recall curve (AP = {avg_precision:.2f})",
)

y_test_numeric = np.array(y_test_lr, dtype=float)
prevalence = np.mean(y_test_numeric)
plt.axhline(
    y=prevalence,
    color="red",
    linestyle="--",
    label=f"No Skill (prevalence = {prevalence:.4f})",
)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="best")
plt.show()

if hasattr(log_reg, "coef_"):
    coef_df = pd.DataFrame({"Feature": features, "Coefficient": log_reg.coef_[0]})

    coef_df["Abs_Coefficient"] = np.abs(coef_df["Coefficient"])
    coef_df = coef_df.sort_values("Abs_Coefficient", ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x="Coefficient", y="Feature", data=coef_df)
    plt.title("Feature Importances (Coefficients)")
    plt.tight_layout()
    plt.show()

    print("\nFeature Importances:")
    print(coef_df[["Feature", "Coefficient"]].to_string(index=False))
