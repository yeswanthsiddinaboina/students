from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

class StudentPerformanceApp:
    def __init__(self):
        with open("model.pkl", "rb") as file:
            self.model = pickle.load(file)

    def predict(self, data):
        prediction = self.model.predict(data)
        return round(float(prediction[0]), 2)

student_app = StudentPerformanceApp()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        hours = float(request.form.get("hours_studied"))
        scores = float(request.form.get("previous_scores"))
        activity = request.form.get("extracurricular")
        sleep = float(request.form.get("sleep_hours"))
        papers = float(request.form.get("question_papers"))

        activity_value = 1 if activity.lower() == "yes" else 0

        data = pd.DataFrame([[
            hours, scores, activity_value, sleep, papers
        ]], columns=[
            "Hours Studied",
            "Previous Scores",
            "Extracurricular Activities",
            "Sleep Hours",
            "Sample Question Papers Practiced"
        ])

        result = student_app.predict(data)

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)