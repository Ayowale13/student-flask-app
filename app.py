from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("student_performance_model.pkl")
scaler = joblib.load("scaler.pkl")

# Label map
LABEL_MAP = {0: "At-Risk / Low", 1: "Average / Medium", 2: "High Performance"}
LABEL_CLASS = {0: "at-risk", 1: "average", 2: "high"}
ADVICE_MAP = {
    0: "This student shows significant risk indicators. Immediate academic counselling and attendance review is strongly recommended.",
    1: "This student is performing at an average level. An advisor check-in and review of study habits is advised.",
    2: "This student is on a strong academic trajectory. Continue current support and consider scholarship referral."
}

# Encode helpers (must match training encoding)
SEX_MAP     = {"Male": 1, "Female": 0}
ADDR_MAP    = {"Urban": 1, "Rural": 0}
FAM_MAP     = {"Greater than 3 (GT3)": 0, "Less or equal to 3 (LE3)": 1}
PSTAT_MAP   = {"Living Together (T)": 1, "Apart (A)": 0}
JOB_MAP     = {"At Home": 0, "Health": 1, "Other": 2, "Services": 3, "Teacher": 4}
REASON_MAP  = {"Course Preference": 0, "Home Proximity": 1, "Other": 2, "Reputation": 3}
GUARD_MAP   = {"Father": 0, "Mother": 1, "Other": 2}
BIN_MAP     = {"Yes": 1, "No": 0}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        f = request.form

        row = [
            SEX_MAP[f["sex"]],
            int(f["age"]),
            ADDR_MAP[f["address"]],
            FAM_MAP[f["famsize"]],
            PSTAT_MAP[f["Pstatus"]],
            int(f["Medu"]),
            int(f["Fedu"]),
            JOB_MAP[f["Mjob"]],
            JOB_MAP[f["Fjob"]],
            REASON_MAP[f["reason"]],
            GUARD_MAP[f["guardian"]],
            int(f["traveltime"]),
            int(f["studytime"]),
            int(f["failures"]),
            BIN_MAP[f["schoolsup"]],
            BIN_MAP[f["famsup"]],
            BIN_MAP[f["paid"]],
            BIN_MAP[f["activities"]],
            BIN_MAP[f["nursery"]],
            BIN_MAP[f["higher"]],
            BIN_MAP[f["internet"]],
            BIN_MAP[f["romantic"]],
            int(f["famrel"]),
            int(f["freetime"]),
            int(f["goout"]),
            int(f["Dalc"]),
            int(f["Walc"]),
            int(f["health"]),
            int(f["absences"]),
        ]

        arr = np.array([row])
        arr_scaled = scaler.transform(arr)
        pred = model.predict(arr_scaled)[0]
        proba = model.predict_proba(arr_scaled)[0]

        # Risk flags
        flags = []
        if int(f["absences"]) > 20:
            flags.append("🔴 High Absence Rate — Attendance risk detected")
        if int(f["studytime"]) <= 1:
            flags.append("🟠 Low Study Time — Low engagement flag")
        if int(f["Walc"]) >= 4:
            flags.append("🟡 High Weekend Alcohol Consumption — Wellbeing alert")
        if int(f["failures"]) >= 2:
            flags.append("🔴 Multiple Past Failures — Academic risk flag")
        if int(f["Medu"]) == 0 and f["internet"] == "No":
            flags.append("🔵 Low Parental Education + No Internet — Support needed")

        result = {
            "label": LABEL_MAP[pred],
            "label_class": LABEL_CLASS[pred],
            "advice": ADVICE_MAP[pred],
            "prob_low": round(proba[0] * 100, 1),
            "prob_med": round(proba[1] * 100, 1),
            "prob_high": round(proba[2] * 100, 1),
            "confidence": round(max(proba) * 100, 1),
            "flags": flags,
        }

        return render_template("index.html", result=result, form_data=f)

    except Exception as e:
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True)