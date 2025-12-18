import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import joblib

FILE_PATH = "diabetes_012_health_indicators_BRFSS2015.csv"

# 1. Завантаження даних
data = pd.read_csv(FILE_PATH)

# Очікувані колонки з "малого" BRFSS
# Diabetes_012, HighBP, HighChol, CholCheck, BMI, Smoker, Stroke,
# HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump,
# AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk,
# Sex, Age, Education, Income

# 2. Створюємо "опитувальні" ознаки в тій самій шкалі, що й у Flask

def make_survey_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # -------- головні речі, що у нас дійсно є --------
    # GENHLTH 1–5
    out["GENHLTH"] = df["GenHlth"].astype(int)

    # X_AGEG5YR – 6 вікових категорій з Age (1–13)
    age = df["Age"].astype(int)
    bins = [0, 3, 5, 7, 9, 11, 13]
    labels = [1, 2, 3, 4, 5, 6]
    out["X_AGEG5YR"] = pd.cut(age, bins=bins, labels=labels).astype(int)

    # X_BMI5CAT – 4 категорії BMI
    bmi = df["BMI"].astype(float)
    # 1: Underweight, 2: Normal, 3: Overweight, 4: Obese
    conds = [
        bmi < 18.5,
        (bmi >= 18.5) & (bmi < 25),
        (bmi >= 25) & (bmi < 30),
        bmi >= 30,
    ]
    cats = [1, 2, 3, 4]
    out["X_BMI5CAT"] = np.select(conds, cats, default=2).astype(int)

    # INCOME2 – беремо прямо з Income (1–8)
    out["INCOME2"] = df["Income"].astype(int)

    # SEX – у датасеті: 0=Female,1=Male; у формі: 1=Male,2=Female
    sex = df["Sex"].astype(int)
    out["SEX"] = np.where(sex == 1, 1, 2).astype(int)

    # MENTHLTH – 1: 0–5, 2: 6–30
    ment = df["MentHlth"].astype(int)
    out["MENTHLTH"] = np.where(ment <= 5, 1, 2).astype(int)

    # X_TOTINDA – з PhysActivity: 1 – була активність, 2 – ні
    phys = df["PhysActivity"].astype(int)
    out["X_TOTINDA"] = np.where(phys == 1, 1, 2).astype(int)

    # _SMOKER3 – з бінарного Smoker (0/1)
    # дуже груба апроксимація:
    # 1: Current every day, 4: Never smoked
    sm = df["Smoker"].astype(int)
    out["_SMOKER3"] = np.where(sm == 1, 1, 4).astype(int)

    # DIABETE3 – з Diabetes_012 (0,1,2)
    # 0: No diabetes, 1: prediabetes, 2: diabetes
    # Map: 3: No, 4: Prediabetes, 1: Yes
    d = df["Diabetes_012"].astype(int)
    out["DIABETE3"] = np.select(
        [d == 0, d == 1, d == 2],
        [3, 4, 1],
        default=3,
    ).astype(int)

    # -------- інші змінні, яких немає в csv -> ставимо типове значення --------
    # Вони ВСЮДИ однакові, модель їх просто проігнорує
    const_maps = {
        "CHECKUP1": 1,   # <1 year
        "X_RACE": 1,     # White
        "MSCODE": 1,     # Center city
        "FLUSHOT6": 2,   # No
        "EMPLOY1": 1,    # Employed
        "MARITAL": 1,    # Married
        "X_EDUCAG": 3,   # Attended college (грубо)
        "SLEPTIM1": 2,   # 7–8 hours
        "CVDCRHD4": 2,   # No
        "HLTHCVR1": 1,   # Yes
        "CHCKIDNY": 2,   # No
        "USEEQUIP": 2,   # No
        "ADDEPEV2": 2,   # No
        "RENTHOM1": 1,   # Own
        "EXERANY2": 1,   # Yes
        "BLIND": 2,      # No
        "DECIDE": 2,     # No
        "HLTHPLN1": 1,   # Yes
    }

    for col, val in const_maps.items():
        out[col] = val

    # Переконуємося, що порядок колонок збігається з encode_survey в Flask
    feature_order = [
        "GENHLTH",
        "X_AGEG5YR",
        "X_BMI5CAT",
        "CHECKUP1",
        "INCOME2",
        "X_RACE",
        "MSCODE",
        "FLUSHOT6",
        "EMPLOY1",
        "SEX",
        "MARITAL",
        "X_EDUCAG",
        "SLEPTIM1",
        "CVDCRHD4",
        "HLTHCVR1",
        "MENTHLTH",
        "CHCKIDNY",
        "USEEQUIP",
        "X_TOTINDA",
        "ADDEPEV2",
        "RENTHOM1",
        "EXERANY2",
        "BLIND",
        "DECIDE",
        "HLTHPLN1",
        "DIABETE3",
        "_SMOKER3",
    ]

    out = out[feature_order]
    return out, feature_order


# 3. Будуємо ознаки і ціль
X_survey, feature_order = make_survey_features(data)
y = data["Diabetes_012"].astype(int)

# 4. Трейн / тест + масштабування
X_train, X_test, y_train, y_test = train_test_split(
    X_survey,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Навчання Random Forest
rf = RandomForestClassifier(
    n_estimators=20,
    max_depth=5,
    random_state=42,
    class_weight="balanced_subsample",
)
rf.fit(X_train_scaled, y_train)

# 6. Оцінка
y_pred = rf.predict(X_test_scaled)
y_proba = rf.predict_proba(X_test_scaled)

print("\n=== RandomForest на опитувальних ознаках ===")
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

auc_macro = roc_auc_score(
    y_test,
    y_proba,
    multi_class="ovr",
    average="macro",
)
print(f"ROC-AUC (macro, OVR): {auc_macro:.4f}")

# 7. Важливість ознак
importances = pd.Series(rf.feature_importances_, index=feature_order).sort_values(ascending=False)
print("\nТоп важливих ознак:")
print(importances.head(20))

plt.figure(figsize=(10, 6))
sns.barplot(x=importances.head(15).values, y=importances.head(15).index)
plt.xlabel("Важливість (feature_importance)")
plt.ylabel("Ознака")
plt.title("Топ-15 найважливіших ознак (RandomForest, опитувальні)")
plt.tight_layout()
plt.show()

# 8. ROC-криві
classes = [0, 1, 2]
y_test_bin = label_binarize(y_test, classes=classes)

def plot_multiclass_roc(y_true_bin, y_score, classes, model_name):
    n_classes = y_true_bin.shape[1]
    fpr = {}
    tpr = {}
    roc_auc_vals = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc_vals[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc_vals["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(7, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"Клас {classes[i]} (AUC = {roc_auc_vals[i]:.3f})")
    plt.plot(fpr["micro"], tpr["micro"], linestyle="--", linewidth=2,
             label=f"Micro-average (AUC = {roc_auc_vals['micro']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Випадковий класифікатор")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC-криві (One-vs-Rest) для {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

plot_multiclass_roc(y_test_bin, y_proba, classes, "RandomForest (survey features)")

# 9. Збереження моделі й scaler
joblib.dump(rf, "survey_rf_model.pkl")
joblib.dump(scaler, "survey_scaler.pkl")
print("\nМодель збережено у survey_rf_model.pkl, scaler у survey_scaler.pkl")

