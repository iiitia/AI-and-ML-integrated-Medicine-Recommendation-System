import os
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user

# ----------------------------------------------------
# App Setup
# ----------------------------------------------------
app = Flask(__name__)
app.secret_key = "super_secret_key"

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------
# User Storage
# ----------------------------------------------------
USERS_FILE = os.path.join(BASE_DIR, "users.json")

if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w") as f:
        json.dump({}, f)

def load_users():
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

class User(UserMixin):
    def __init__(self, username):
        self.id = username

@login_manager.user_loader
def load_user(user_id):
    users = load_users()
    if user_id in users:
        return User(user_id)
    return None

# ----------------------------------------------------
# Load Model (SAFE)
# ----------------------------------------------------
model_path = os.path.join(BASE_DIR, "svc.pkl")
svc = None

if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        svc = pickle.load(f)
else:
    print("Model file not found!")

# ----------------------------------------------------
# Load Datasets (SAFE)
# ----------------------------------------------------
def safe_read_csv(filename):
    path = os.path.join(BASE_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"{filename} not found!")
    return pd.DataFrame()

sym_des = safe_read_csv("symtoms_df.csv")
precautions = safe_read_csv("precautions_df.csv")
workout = safe_read_csv("workout_df.csv")
description = safe_read_csv("description.csv")
medications = safe_read_csv("medications.csv")
diets = safe_read_csv("diets.csv")

# ----------------------------------------------------
# Dummy Dictionaries (Replace with your real ones)
# ----------------------------------------------------
symptoms_dict = {
    "itching": 0
}

diseases_list = {
    0: "Fungal infection"
}

# ----------------------------------------------------
# Prediction Logic
# ----------------------------------------------------
def get_predicted_value(patient_symptoms):
    if svc is None:
        return "Model not loaded"

    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1

    prediction = svc.predict([input_vector])[0]
    return diseases_list.get(prediction, "Unknown Disease")

# ----------------------------------------------------
# Routes
# ----------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():  # 🔥 Removed login_required
    symptoms = request.form.get("symptoms")

    if not symptoms:
        return render_template("index.html", message="Please enter symptoms.")

    user_symptoms = [
        s.strip().lower().replace(" ", "_")
        for s in symptoms.split(",")
    ]

    predicted_disease = get_predicted_value(user_symptoms)

    return render_template(
        "index.html",
        predicted_disease=predicted_disease
    )

# ----------------------------------------------------
# Auth Routes
# ----------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        users = load_users()
        username = request.form["username"].strip()
        password = request.form["password"].strip()

        if username in users:
            flash("Username already exists!", "danger")
            return redirect(url_for("register"))

        users[username] = password
        save_users(users)
        flash("Registration successful!", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    users = load_users()

    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()

        if username in users and users[username] == password:
            login_user(User(username))
            return redirect(url_for("index"))

        flash("Invalid credentials!", "danger")

    return render_template("login.html")

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("login"))

# ----------------------------------------------------
# Run App
# ----------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

