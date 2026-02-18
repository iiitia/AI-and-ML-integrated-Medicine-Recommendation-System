import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from PIL import Image

# Safe OCR import (won't crash on Render)
try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except:
    OCR_AVAILABLE = False

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
# Load Model
# ----------------------------------------------------
model_path = os.path.join(BASE_DIR, "svc.pkl")
svc = pickle.load(open(model_path, "rb"))

# ----------------------------------------------------
# Load Datasets
# ----------------------------------------------------
sym_des = pd.read_csv(os.path.join(BASE_DIR, "symtoms_df.csv"))
precautions = pd.read_csv(os.path.join(BASE_DIR, "precautions_df.csv"))
workout = pd.read_csv(os.path.join(BASE_DIR, "workout_df.csv"))
description = pd.read_csv(os.path.join(BASE_DIR, "description.csv"))
medications = pd.read_csv(os.path.join(BASE_DIR, "medications.csv"))
diets = pd.read_csv(os.path.join(BASE_DIR, "diets.csv"))

# ----------------------------------------------------
# Prediction Logic
# ----------------------------------------------------
symptoms_dict = { ... }  # KEEP YOUR FULL DICTIONARY HERE
diseases_list = { ... }  # KEEP YOUR FULL DICTIONARY HERE

def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][
        ['Precaution_1','Precaution_2','Precaution_3','Precaution_4']
    ]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [m for m in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [d for d in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']

    return desc, pre, med, die, wrkout

# ----------------------------------------------------
# Routes
# ----------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    symptoms = request.form.get("symptoms")
    if not symptoms:
        return render_template("index.html", message="Please enter symptoms.")

    user_symptoms = [s.strip().lower().replace(" ", "_") for s in symptoms.split(",")]
    predicted_disease = get_predicted_value(user_symptoms)

    dis_des, pre_list, med_list, diet_list, workout_list = helper(predicted_disease)

    return render_template(
        "index.html",
        predicted_disease=predicted_disease,
        dis_des=dis_des,
        precautions=pre_list,
        medications=med_list,
        rec_diet=diet_list,
        workout=workout_list
    )

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
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# ----------------------------------------------------
# OCR Upload (Safe Version)
# ----------------------------------------------------
@app.route("/upload_prescription", methods=["GET", "POST"])
@login_required
def upload_prescription():
    if not OCR_AVAILABLE:
        return "OCR not available in deployed version"

    if request.method == "POST":
        file = request.files["prescription_file"]
        upload_folder = os.path.join(BASE_DIR, "uploads")
        os.makedirs(upload_folder, exist_ok=True)

        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        return render_template("prescription_results.html", medicines=["OCR Feature Active"])

    return render_template("upload_prescription.html")

# ----------------------------------------------------
# Run App
# ----------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


