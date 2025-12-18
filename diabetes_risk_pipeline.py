import warnings

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,

)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from imblearn.combine import SMOTEENN

warnings.filterwarnings("ignore")


file_path = "diabetes_012_health_indicators_BRFSS2015.csv"
data = pd.read_csv(file_path)


# ============================================================
# 2. Приведення типів, опис, дублі
# ============================================================

int_columns = [
    "Diabetes_012",
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]

for col in int_columns:
    data[col] = data[col].astype(int)

print(data.info())
print(f"Num rows: {len(data)}")
print(f"Num columns: {len(data.columns)}")

print("Розподіл Diabetes_012 (нормалізований):")
print(data["Diabetes_012"].value_counts(normalize=True))

print("\nОписова статистика:")
print(data.describe())

print("\nПропущені значення по стовпцях:")
print(data.isnull().sum())
print("Чи є пропуски?:", data.isnull().sum().any())

print("\nКількість дублікатів:", data.duplicated().sum())
data.drop_duplicates(inplace=True)
print("Форма після видалення дублікатів:", data.shape)

# ============================================================
# 3. Кореляції та базові візуалізації
# ============================================================

corr_matrix = data.corr()

plt.figure(figsize=(20, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    linewidths=0.5
)
plt.title("Кореляційна матриця ознак (з числовими значеннями)")
plt.tight_layout()
plt.show()

df_vis = data.copy()

df_vis["Diabetes_012"] = df_vis["Diabetes_012"].map(
    {0: "No Diabetes", 1: "Pre Diabetes", 2: "Diabetes"}
)

df_vis["HighBP"] = df_vis["HighBP"].map({0: "No High", 1: "High BP"})
df_vis["HighChol"] = df_vis["HighChol"].map(
    {0: "No High Cholesterol", 1: "High Cholesterol"}
)
df_vis["CholCheck"] = df_vis["CholCheck"].map(
    {0: "No Cholesterol Check in 5 Years", 1: "Cholesterol Check in 5 Years"}
)

binary_yes_no_cols = [
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "DiffWalk",
]

for c in binary_yes_no_cols:
    df_vis[c] = df_vis[c].map({0: "No", 1: "Yes"})

df_vis["GenHlth"] = df_vis["GenHlth"].map(
    {
        1: "Excellent",
        2: "Very Good",
        3: "Good",
        4: "Fair",
        5: "Poor",
    }
)
df_vis["Sex"] = df_vis["Sex"].map({0: "Female", 1: "Male"})

df_vis["Education"] = df_vis["Education"].map(
    {
        1: "Never Attended School",
        2: "Elementary",
        3: "Some high school",
        4: "High school graduate",
        5: "Some college/tech school",
        6: "College graduate",
    }
)

df_vis["Income"] = df_vis["Income"].map(
    {
        1: "Less Than $10,000",
        2: "Less Than $10,000",
        3: "Less Than $10,000",
        4: "Less Than $10,000",
        5: "Less Than $35,000",
        6: "Less Than $35,000",
        7: "Less Than $35,000",
        8: "$75,000 or More",
    }
)

plt.figure(figsize=(8, 6))
labels = ["No Diabetes", "Pre Diabetes", "Diabetes"]
counts = df_vis["Diabetes_012"].value_counts()
sizes = [counts.get(l, 0) for l in labels]
explode = (0.05, 0.05, 0.05)
plt.pie(
    sizes,
    labels=["Не має діабет", "Перед діабет", "Є діабет"],
    explode=explode,
    autopct="%.1f%%",
)
plt.title("Співвідношення трьох можливих стадій діабету")
plt.show()

corr_with_target = data.drop("Diabetes_012", axis=1).corrwith(data["Diabetes_012"])

corr_sorted = corr_with_target.sort_values(ascending=False)


plt.figure(figsize=(15, 6))
corr_sorted.plot(
    kind="bar",
    grid=True,
    title="Кореляція факторів з діабетом (від найсильніших до найслабших)",
)
plt.ylabel("Коефіцієнт кореляції")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 4))
ax = sns.countplot(data=df_vis, x="Diabetes_012", hue="Sex")
ax.set_xlabel("Наявність діабету")
ax.set_ylabel("Кількість")
ax.set_title("Стовпчаста діаграма розподілу діабету за статтю")
ax.set_xticklabels(["Не має діабету", "Перед діабет", "Має діабет"])
plt.legend(title="Стать", labels=["Жінка", "Чоловік"])
for p in ax.patches:
    height = p.get_height()
    ax.annotate(
        f"{(height / df_vis.shape[0]) * 100:.2f}%",
        (p.get_x() + 0.05, height + 0.01),
        fontsize=8,
    )
plt.show()

# Приклад: вплив куріння
plt.figure(figsize=(12, 5))
ax = sns.countplot(data=df_vis, x="Smoker", hue="Diabetes_012")
ax.set_xlabel("Куріння")
ax.set_ylabel("Кількість")
ax.set_title("Вплив фактора куріння на виявлення цукрового діабету")
ax.set_xticklabels(["Не курить", "Курить"])
plt.legend(
    title="Наявність діабету",
    labels=["Не має діабету", "Перед діабет", "Має діабет"],
)
for p in ax.patches:
    height = p.get_height()
    ax.annotate(
        f"{(height / df_vis.shape[0]) * 100:.2f}%",
        (p.get_x() + 0.05, height + 0.01),
        fontsize=8,
    )
plt.show()

# (за бажанням можна додати всі інші твої діаграми аналогічно – Stroke, HeartDiseaseorAttack,
# Fruits, Veggies, HvyAlcoholConsump, PhysActivity, BMI, Age, MentHlth, PhysHlth, GenHlth, Income, Education тощо)

# ============================================================
# 4. Обробка викидів, масштабування
# ============================================================

# Обмежимо BMI < 70, як у твоєму коді
df = data[data["BMI"] < 70].copy()

# Виявлення викидів IsolationForest
iso_model = IsolationForest(random_state=0)
iso_model.fit(df)
df["anomaly"] = iso_model.predict(df)

print("Кількість викидів за IsolationForest:", (df["anomaly"] == -1).sum())
df = df[df["anomaly"] != -1].copy()
df.drop(columns=["anomaly"], inplace=True)
print("Shape після прибирання викидів:", df.shape)

# Масштабування ознак
X = df.drop("Diabetes_012", axis=1)
y = df["Diabetes_012"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Розбиття train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.35,
    random_state=0,
    shuffle=True,
)

# ============================================================
# 5. K-Nearest Neighbors (без ресемплінгу)
# ============================================================

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print("\n=== KNN без ресемплінгу ===")
print(classification_report(y_test, y_pred_knn))
print("Confusion matrix:")
print(pd.crosstab(y_test, y_pred_knn, rownames=["Actual"], colnames=["Predicted"]))

# ============================================================
# 6. Ресемплінг SMOTEENN + KNN
# ============================================================

sm = SMOTEENN()
X_resampled, y_resampled = sm.fit_resample(X, y)

# -------------------------------------------
# Очистка синтетичних даних після SMOTEENN
# -------------------------------------------

X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
y_resampled_df = pd.Series(y_resampled, name="Diabetes_012")

# 1. Фільтр BMI < 70
mask_bmi = X_resampled_df["BMI"] < 70
X_resampled_df = X_resampled_df[mask_bmi]
y_resampled_df = y_resampled_df[mask_bmi]

print("Після фільтру BMI < 70:", X_resampled_df.shape)

# 2. Видалення дублікатів
df_res = pd.concat([X_resampled_df, y_resampled_df], axis=1)
duplicates = df_res.duplicated().sum()
print("Кількість дублікатів у синтетиці:", duplicates)

df_res = df_res.drop_duplicates()
print("Форма після видалення дублікатів:", df_res.shape)

# Розділяємо назад
X_resampled = df_res.drop("Diabetes_012", axis=1).values
y_resampled = df_res["Diabetes_012"].values


Xre_train, Xre_test, yre_train, yre_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.3,
    random_state=42,
)

knn_smote = KNeighborsClassifier(n_neighbors=5)
knn_smote.fit(Xre_train, yre_train)
yre_pred_knn = knn_smote.predict(Xre_test)

print("\n=== KNN з SMOTEENN ===")
print(classification_report(yre_test, yre_pred_knn, labels=[0, 1, 2]))
print(
    "Confusion matrix SMOTEENN:\n",
    pd.crosstab(yre_test, yre_pred_knn, rownames=["Actual"], colnames=["Predicted"]),
)

# ============================================================
# 7. Decision Tree
# ============================================================

dt = DecisionTreeClassifier(criterion="entropy", max_depth=40, random_state=0)
dt.fit(Xre_train, yre_train)

y_pred_train_dt = dt.predict(Xre_train)
y_pred_test_dt = dt.predict(Xre_test)

print("\n=== Decision Tree ===")
print("Train score:", dt.score(Xre_train, yre_train))
print("Test score:", dt.score(Xre_test, yre_test))
print("Train accuracy:", accuracy_score(yre_train, y_pred_train_dt))
print("Test accuracy:", accuracy_score(yre_test, y_pred_test_dt))
print(classification_report(yre_test, y_pred_test_dt))
print("Precision: %.3f" % precision_score(yre_test, y_pred_test_dt, average="micro"))
print("Recall: %.3f" % recall_score(yre_test, y_pred_test_dt, average="micro"))
print("F1-score: %.3f" % f1_score(yre_test, y_pred_test_dt, average="micro"))

# ============================================================
# 8. Random Forest
# ============================================================

rf = RandomForestClassifier(
    n_estimators=800,
    max_features=16,
    max_depth=50,
    random_state=0,
)
rf.fit(Xre_train, yre_train)

import joblib

joblib.dump(rf, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")


y_pred_train_rf = rf.predict(Xre_train)
y_pred_test_rf = rf.predict(Xre_test)

print("\n=== Random Forest ===")
print("Train score:", rf.score(Xre_train, yre_train))
print("Test score:", rf.score(Xre_test, yre_test))
print("Train accuracy:", accuracy_score(yre_train, y_pred_train_rf))
print("Test accuracy:", accuracy_score(yre_test, y_pred_test_rf))
print(classification_report(yre_test, y_pred_test_rf))
print("Precision: %.3f" % precision_score(yre_test, y_pred_test_rf, average="micro"))
print("Recall: %.3f" % recall_score(yre_test, y_pred_test_rf, average="micro"))
print("F1-score: %.3f" % f1_score(yre_test, y_pred_test_rf, average="micro"))

# ============================================================
# 8.1. Важливість ознак (feature importance) для Random Forest
# ============================================================

feature_names = X.columns  # імена ознак до SMOTEENN

importances = pd.Series(rf.feature_importances_, index=feature_names)

# Сортуємо від найважливіших до найменш важливих
importances_sorted = importances.sort_values(ascending=False)

print("\nТоп ознак за важливістю (Random Forest):")
print(importances_sorted)

# Виберемо топ-N для візуалізації
top_n = 15
top_features = importances_sorted.head(top_n)

plt.figure(figsize=(10, 6))
sns.barplot(
    x=top_features.values,
    y=top_features.index,
)
plt.xlabel("Важливість ознаки (feature_importance)")
plt.ylabel("Ознака")
plt.title(f"Топ-{top_n} найважливіших ознак за Random Forest")
plt.tight_layout()
plt.show()

# ============================================================
# 9. ROC-AUC, ROC-криві та матриці плутанини
# ============================================================

classes = [0, 1, 2]
y_test_bin = label_binarize(yre_test, classes=classes)


def plot_multiclass_roc(y_true_bin, y_score, classes, model_name):
    """
    Малює ROC-криві по класах + micro-average для мультикласової задачі.
    """
    n_classes = y_true_bin.shape[1]

    fpr = {}
    tpr = {}
    roc_auc_vals = {}

    # ROC для кожного класу
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc_vals[i] = auc(fpr[i], tpr[i])

    # micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_true_bin.ravel(),
        y_score.ravel(),
    )
    roc_auc_vals["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(7, 6))
    for i in range(n_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            label=f"Клас {classes[i]} (AUC = {roc_auc_vals[i]:.3f})",
        )

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        linestyle="--",
        linewidth=2,
        label=f"Micro-average (AUC = {roc_auc_vals['micro']:.3f})",
    )

    plt.plot([0, 1], [0, 1], "k--", label="Випадковий класифікатор")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC-криві (One-vs-Rest) для {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    return roc_auc_vals["micro"]


# ----- KNN (SMOTEENN) -----
y_score_knn = knn_smote.predict_proba(Xre_test)
auc_knn_macro = roc_auc_score(
    yre_test,
    y_score_knn,
    multi_class="ovr",
    average="macro",
)
print(f"\nKNN (SMOTEENN) ROC-AUC (macro, OVR): {auc_knn_macro:.4f}")
auc_knn_micro = plot_multiclass_roc(
    y_test_bin,
    y_score_knn,
    classes,
    model_name="KNN (SMOTEENN)",
)

cm_knn = confusion_matrix(yre_test, yre_pred_knn, labels=classes)
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm_knn,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes,
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – KNN (SMOTEENN)")
plt.show()

# ----- Decision Tree -----
y_score_dt = dt.predict_proba(Xre_test)
auc_dt_macro = roc_auc_score(
    yre_test,
    y_score_dt,
    multi_class="ovr",
    average="macro",
)
print(f"Decision Tree ROC-AUC (macro, OVR): {auc_dt_macro:.4f}")
auc_dt_micro = plot_multiclass_roc(
    y_test_bin,
    y_score_dt,
    classes,
    model_name="Decision Tree",
)

cm_dt = confusion_matrix(yre_test, y_pred_test_dt, labels=classes)
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm_dt,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=classes,
    yticklabels=classes,
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Decision Tree")
plt.show()

# ----- Random Forest -----
y_score_rf = rf.predict_proba(Xre_test)
auc_rf_macro = roc_auc_score(
    yre_test,
    y_score_rf,
    multi_class="ovr",
    average="macro",
)
print(f"Random Forest ROC-AUC (macro, OVR): {auc_rf_macro:.4f}")
auc_rf_micro = plot_multiclass_roc(
    y_test_bin,
    y_score_rf,
    classes,
    model_name="Random Forest",
)

cm_rf = confusion_matrix(yre_test, y_pred_test_rf, labels=classes)
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Oranges",
    xticklabels=classes,
    yticklabels=classes,
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Random Forest")
plt.show()

# Порівняння micro-average ROC моделей на одному графіку
fpr_knn_micro, tpr_knn_micro, _ = roc_curve(
    y_test_bin.ravel(),
    y_score_knn.ravel(),
)
roc_knn_micro = auc(fpr_knn_micro, tpr_knn_micro)

fpr_dt_micro, tpr_dt_micro, _ = roc_curve(
    y_test_bin.ravel(),
    y_score_dt.ravel(),
)
roc_dt_micro = auc(fpr_dt_micro, tpr_dt_micro)

fpr_rf_micro, tpr_rf_micro, _ = roc_curve(
    y_test_bin.ravel(),
    y_score_rf.ravel(),
)
roc_rf_micro = auc(fpr_rf_micro, tpr_rf_micro)

plt.figure(figsize=(8, 6))
plt.plot(
    fpr_knn_micro,
    tpr_knn_micro,
    label=f"KNN micro (AUC = {roc_knn_micro:.3f})",
)
plt.plot(
    fpr_dt_micro,
    tpr_dt_micro,
    label=f"Decision Tree micro (AUC = {roc_dt_micro:.3f})",
)
plt.plot(
    fpr_rf_micro,
    tpr_rf_micro,
    label=f"Random Forest micro (AUC = {roc_rf_micro:.3f})",
)
plt.plot([0, 1], [0, 1], "k--", label="Випадковий класифікатор")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Порівняння micro-average ROC-кривих моделей")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
