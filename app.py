from flask import Flask, render_template, request, redirect, url_for, session
import matplotlib
matplotlib.use("Agg")  # без монітора
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import joblib
import pandas as pd

app = Flask(__name__)
app.secret_key = "super_secret_key_for_demo"

# ==============================
# 0. Завантаження моделі, scaler та датасету
# ==============================

rf = joblib.load("survey_rf_model.pkl")
scaler = joblib.load("survey_scaler.pkl")

# датасет для глобальних графіків (як у train.py)
data = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")

# ==============================
# 1. Дані для вибору лікарів
# ==============================

DOCTORS = [
    "д-р Іваненко (ендокринолог)",
    "д-р Петрова (терапевт)",
    "д-р Шевченко (кардіолог)",
    "д-р Ковалюк (сімейний лікар)",
]

# ==============================
# 2. Мапінги відповідей → кодів
# ==============================

def encode_survey(form_data):

    genhlth_map = {
        "Відмінно": 1,
        "Дуже добре": 2,
        "Добре": 3,
        "Задовільно": 4,
        "Погано": 5,
    }

    age_map = {
        "31–40 років": 1,
        "41–50 років": 2,
        "51–60 років": 3,
        "61–70 років": 4,
        "71–80 років": 5,
        "більше 81 року": 6,
    }

    bmi_map = {
        "Недостатня вага": 1,
        "Нормальна вага": 2,
        "Надмірна вага": 3,
        "Ожиріння": 4,
    }

    checkup_map = {
        "Менше 1 року": 1,
        "1–2 роки": 2,
        "3–5 років": 3,
        "Більше 5 років": 4,
        "Ніколи": 6,
    }

    income_map = {
        "Менше $10 тис.": 1,
        "$10–15 тис.": 2,
        "$15–20 тис.": 3,
        "$20–25 тис.": 4,
        "$25–35 тис.": 5,
        "$35–50 тис.": 6,
        "$50–75 тис.": 7,
        "Більше $75 тис.": 8,
    }

    race_map = {
        "Білий": 1,
        "Чорний": 2,
        "Американець індіанського походження або коренем жителем Аляски": 3,
        "Азіат": 4,
        "Цільносоціальні гавайці або інші мешканці Тихого океану": 5,
        "Інша раса": 6,
        "Змішаної раси": 7,
        "Іспанець": 8,
    }

    mscode_map = {
        "Центр міста": 1,
        "Райони округи": 2,
        "Передмістя": 3,
        "Не в межах MSA": 5,
    }

    yes_no_map = {"Так": 1, "Ні": 2}

    employ_map = {
        "Працюю": 1,
        "Індивідуальний підприємець": 2,
        "Без роботи більше 1 року": 3,
        "Без роботи менше 1 року": 4,
        "Домогосподарка": 5,
        "Студент": 6,
        "На пенсії": 7,
        "Не можу працювати": 8,
    }

    sex_map = {
        "Чоловік": 1,
        "Жінка": 2,
    }

    marital_map = {
        "Одружений(а)": 1,
        "Розлучений(а)": 2,
        "Вдівець/вдова": 3,
        "Розлучені разом": 4,
        "Ніколи не одружувався(лась)": 5,
        "Неолюблена пара": 6,
    }

    educ_map = {
        "Не закінчив(ла) середню школу": 1,
        "Закінчив(ла) середню школу": 2,
        "Відвідував(а) коледж": 3,
        "Закінчив(ла) коледж": 4,
    }

    sleep_map = {
        "1–6 годин": 1,
        "7–8 годин": 2,
        "9 або більше годин": 3,
    }

    menthlth_map = {
        "0–5 днів": 1,
        "6–30 днів": 2,
    }

    smoker_map = {
        "Курю кожен день": 1,
        "Курю деякі дні": 2,
        "Колишній палій": 3,
        "Ніколи не палив(а)": 4,
    }

    diabete3_map = {
        "Так": 1,
        "Так, але тільки під час вагітності": 2,
        "Ні": 3,
        "Предіабет / межа": 4,
    }

    phys_map = {
        "Мав(а) фізичну активність або вправи": 1,
        "Немає фізичної активності за останні 30 днів": 2
    }

    rent_map = {
        "Власна": 1,
        "Орендована": 2,
        "Інша": 3
    }

    encoded = {
        "GENHLTH": genhlth_map[form_data["GENHLTH"]],
        "X_AGEG5YR": age_map[form_data["X_AGEG5YR"]],
        "X_BMI5CAT": bmi_map[form_data["X_BMI5CAT"]],
        "CHECKUP1": checkup_map[form_data["CHECKUP1"]],
        "INCOME2": income_map[form_data["INCOME2"]],
        "X_RACE": race_map[form_data["X_RACE"]],
        "MSCODE": mscode_map[form_data["MSCODE"]],
        "FLUSHOT6": yes_no_map[form_data["FLUSHOT6"]],
        "EMPLOY1": employ_map[form_data["EMPLOY1"]],
        "SEX": sex_map[form_data["SEX"]],
        "MARITAL": marital_map[form_data["MARITAL"]],
        "X_EDUCAG": educ_map[form_data["X_EDUCAG"]],
        "SLEPTIM1": sleep_map[form_data["SLEPTIM1"]],
        "CVDCRHD4": yes_no_map[form_data["CVDCRHD4"]],
        "HLTHCVR1": yes_no_map[form_data["HLTHCVR1"]],
        "MENTHLTH": menthlth_map[form_data["MENTHLTH"]],
        "CHCKIDNY": yes_no_map[form_data["CHCKIDNY"]],
        "USEEQUIP": yes_no_map[form_data["USEEQUIP"]],
        "X_TOTINDA": phys_map[form_data["X_TOTINDA"]],
        "ADDEPEV2": yes_no_map[form_data["ADDEPEV2"]],
        "RENTHOM1": rent_map[form_data["RENTHOM1"]],
        "EXERANY2": yes_no_map[form_data["EXERANY2"]],
        "BLIND": yes_no_map[form_data["BLIND"]],
        "DECIDE": yes_no_map[form_data["DECIDE"]],
        "HLTHPLN1": yes_no_map[form_data["HLTHPLN1"]],
        "DIABETE3": diabete3_map[form_data["DIABETE3"]],
        "_SMOKER3": smoker_map[form_data["_SMOKER3"]],
    }

    return encoded

# ==============================
# 3. Перетворення відповідей → фічі моделі RandomForest
# ==============================

def build_model_features(encoded):
    """
    Проста обгортка: беремо закодовані відповіді опитувальника
    і формуємо DataFrame в тому ж порядку, що й при тренуванні survey-моделі.
    """

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

    row = [encoded[name] for name in feature_order]
    X_df = pd.DataFrame([row], columns=feature_order)
    return X_df



# ==============================
# 4. Функції для побудови графіків
# ==============================

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64


def build_risk_bar(score):
    fig, ax = plt.subplots(figsize=(4, 1.5))
    ax.barh(["Ризик діабету"], [score], color="#2A6DC4")
    ax.set_xlim(0, 100)
    ax.set_xlabel("0–100")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    return fig_to_base64(fig)


def build_radar_plot(encoded):
    labels = ["Здоров'я", "Вік", "ІМТ", "Фіз.активність", "Куріння"]

    gen = encoded["GENHLTH"] / 5.0
    age = encoded["X_AGEG5YR"] / 6.0
    bmi = encoded["X_BMI5CAT"] / 4.0
    phys = 0 if encoded["X_TOTINDA"] == 1 else 1
    smoke = (encoded["_SMOKER3"] - 1) / 3.0

    values = [gen, age, bmi, phys, smoke]
    values += values[:1]

    angles = [n / float(len(labels)) * 2 * 3.14159 for n in range(len(labels))]
    angles += angles[:1]

    fig = plt.figure(figsize=(4, 4))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids([a * 180 / 3.14159 for a in angles[:-1]], labels)
    ax.set_title("Профіль факторів ризику", pad=20)
    return fig_to_base64(fig)


def build_proba_bar(proba):
    classes = ["0 – немає діабету", "1 – переддіабет", "2 – діабет"]
    percents = [p * 100 for p in proba]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(classes, percents, color="#2A6DC4")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Ймовірність, %")
    plt.xticks(rotation=15, ha="right")
    ax.set_title("Ймовірності діабету")
    return fig_to_base64(fig)


def build_diabetes_pie():
    counts = data["Diabetes_012"].value_counts().sort_index()
    labels = ["0 – немає діабету", "1 – переддіабет", "2 – діабет"]
    sizes = [counts.get(0, 0), counts.get(1, 0), counts.get(2, 0)]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(sizes, labels=labels, autopct="%.1f%%", explode=(0.03, 0.03, 0.03))
    ax.set_title("Розподіл діабету в датасеті")
    return fig_to_base64(fig)

# ==============================
# 5. Маршрути
# ==============================

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        session["first_name"] = request.form["first_name"]
        session["last_name"] = request.form["last_name"]
        session["doctor"] = request.form["doctor"]
        session["wants_check"] = request.form["wants_check"]

        if session["wants_check"] == "Yes":
            return redirect(url_for("diabetes_survey"))
        else:
            return render_template("no_check.html")

    return render_template("register.html", doctors=DOCTORS)


@app.route("/diabetes-survey", methods=["GET", "POST"])
def diabetes_survey():
    if request.method == "POST":
        encoded = encode_survey(request.form)

        # фічі для моделі
        X_model = build_model_features(encoded)
        X_scaled = scaler.transform(X_model)
        proba = rf.predict_proba(X_scaled)[0]
        y_pred = int(rf.predict(X_scaled)[0])

        # ризик на основі ймовірностей моделі
        risk_score = float((proba[1] * 0.5 + proba[2]) * 100)
        risk_score = max(0.0, min(100.0, risk_score))

        if risk_score < 30:
            level = "Низький"
            color = "success"
        elif risk_score < 60:
            level = "Помірний"
            color = "warning"
        else:
            level = "Високий"
            color = "danger"

        classes_labels = {
            0: "Ймовірно немає діабету (клас 0)",
            1: "Ймовірно переддіабет (клас 1)",
            2: "Ймовірно діабет (клас 2)",
        }
        pred_label = classes_labels.get(y_pred, "Невідомий клас")

        # Графіки
        risk_img = build_risk_bar(risk_score)
        radar_img = build_radar_plot(encoded)
        proba_img = build_proba_bar(proba)
        pie_img = build_diabetes_pie()

        return render_template(
            "result.html",
            first_name=session.get("first_name"),
            last_name=session.get("last_name"),
            doctor=session.get("doctor"),
            score=risk_score,
            level=level,
            color=color,
            pred_class=y_pred,
            pred_label=pred_label,
            proba0=proba[0] * 100,
            proba1=proba[1] * 100,
            proba2=proba[2] * 100,
            risk_img=risk_img,
            radar_img=radar_img,
            proba_img=proba_img,
            pie_img=pie_img,
        )

    return render_template("survey.html")


if __name__ == "__main__":
    app.run(debug=True)