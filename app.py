import streamlit as st
st.set_page_config(page_title="Medicine Finder & Reminder", layout="centered")
from dotenv import load_dotenv
load_dotenv()

import os
twilio_sid='YOUR SID'
auth_token = 'YOUR TOKEN'
API_KEY = 'YOUR KEY'

TWILIO_PHONE = '+18507895491'
import pandas as pd
from fuzzywuzzy import process
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from streamlit_javascript import st_javascript
import requests
from twilio.rest import Client
import json
import os
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# ========== CONFIGURATION ==========


# ========== LOAD DATA ==========
medicine_df = pd.read_csv("medicine.csv")
if not os.path.exists("inventory.csv"):
    pd.DataFrame(columns=["medicine_name", "quantity", "added_by"]).to_csv("inventory.csv", index=False)
inventory_df = pd.read_csv("inventory.csv")

# ========== Load BioBERT ==========
@st.cache_resource
def load_biobert():
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    return tokenizer, model

tokenizer, model = load_biobert()

@st.cache_data
def get_medicine_embeddings(medicine_names):
    embeddings = {}
    for name in medicine_names:
        inputs = tokenizer(name, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze()
        embeddings[name] = emb
    return embeddings

medicine_embeddings = get_medicine_embeddings(medicine_df['medicine_name'].unique())

def get_semantic_matches(user_input, top_n=5):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        user_emb = model(**inputs).last_hidden_state.mean(dim=1).squeeze()
    similarities = {
        name: cosine_similarity(user_emb.unsqueeze(0), emb.unsqueeze(0)).item()
        for name, emb in medicine_embeddings.items()  # ✅ fixed bug here
    }
    sorted_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_matches[:top_n]

# ========== PHARMACY STOCK LOGIN PANEL ==========
st.sidebar.title("\U0001F3EA Pharmacy Inventory")
ph_user = st.sidebar.text_input("Username", key="pharm_user")
ph_pass = st.sidebar.text_input("Password", type="password", key="pharm_pass")

def load_pharmacies():
    with open("pharmacies.json", "r") as f:
        return json.load(f)

def save_stock_data(df):
    df.to_csv("inventory.csv", index=False)

if st.sidebar.button("Login"):
    pharmacies = load_pharmacies()
    user = next((u for u in pharmacies if u["username"] == ph_user and u["password"] == ph_pass), None)
    if user:
        st.session_state["pharmacy_logged_in"] = True
        st.session_state["pharmacy_user"] = user
        st.sidebar.success(f"Logged in as {user['name']}")
    else:
        st.sidebar.error("Invalid credentials.")

if st.session_state.get("pharmacy_logged_in"):
    st.header(f"\U0001F4E6 Edit Inventory - {st.session_state['pharmacy_user']['name']}")
    pharmacy_username = st.session_state['pharmacy_user']['username']
    pharmacy_inventory = inventory_df[inventory_df['added_by'] == pharmacy_username]
    editable_df = st.data_editor(pharmacy_inventory, num_rows="dynamic", use_container_width=True)

    if st.button("\U0001F4BE Save Stock Updates"):
        inventory_df = inventory_df[inventory_df['added_by'] != pharmacy_username]
        updated_df = pd.concat([inventory_df, editable_df], ignore_index=True)
        save_stock_data(updated_df)
        st.success("Stock updated successfully!")

# ========== UTILITY FUNCTIONS ==========
def get_coordinates(place):
    geolocator = Nominatim(user_agent="pharmacy_locator")
    location = geolocator.geocode(place)
    if location:
        return f"{location.latitude},{location.longitude}", (location.latitude, location.longitude)
    return None, (None, None)

def get_pharmacies_nearby(location, radius=5000):
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        'location': location,
        'radius': radius,
        'type': 'pharmacy',
        'key': API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()
    if data.get("status") != "OK":
        return pd.DataFrame()
    pharmacies = []
    for place in data.get('results', []):
        pharmacies.append({
            'name': place.get('name'),
            'address': place.get('vicinity'),
            'latitude': place['geometry']['location']['lat'],
            'longitude': place['geometry']['location']['lng']
        })
    return pd.DataFrame(pharmacies)

def get_google_maps_link(name, address):
    return f"https://www.google.com/maps/search/{name.replace(' ', '+')}+{address.replace(' ', '+')}"

def send_sms(to_phone, message):
    if not to_phone.startswith('+'):
        to_phone = f"+{to_phone}"
    client = Client(TWILIO_SID, TWILIO_AUTH)
    try:
        message = client.messages.create(body=message, from_=TWILIO_PHONE, to=to_phone)
        print("SMS sent:", message.sid)
    except Exception as e:
        st.error(f"SMS error: {e}")

# ========== UI START ==========
st.title("\U0001F48A Medicine Finder & Reminder")
st.subheader("\U0001F4CD Choose Your Location")

use_current = st.checkbox("Use my current location")
coordinates, user_lat, user_lon = None, None, None

if use_current:
    st.info("Please allow location access in your browser.")
    result = st_javascript("""await new Promise((resolve) => {
        navigator.geolocation.getCurrentPosition(
            (pos) => resolve({latitude: pos.coords.latitude, longitude: pos.coords.longitude}),
            (err) => resolve({error: err.message})
        );
    });""")
    if result and "latitude" in result:
        user_lat = result["latitude"]
        user_lon = result["longitude"]
        coordinates = f"{user_lat},{user_lon}"
        st.success(f"Location: {user_lat}, {user_lon}")
    elif result and "error" in result:
        st.error(f"Geolocation Error: {result['error']}")
    else:
        st.warning("Waiting for location...")
else:
    city = st.text_input("Enter your city/locality:")
    if city:
        coordinates, (user_lat, user_lon) = get_coordinates(city)

# ========== PHARMACY SEARCH ==========
if coordinates:
    pharmacy_df = get_pharmacies_nearby(coordinates)
    if pharmacy_df.empty:
        st.warning("No pharmacies found nearby.")
    else:
        st.success(f"{len(pharmacy_df)} pharmacies found.")
        st.write("### \U0001F3E5 Nearby Pharmacies")
        for _, row in pharmacy_df.iterrows():
            st.markdown(f"**{row['name']}** - {row['address']}")
            st.markdown(f"[\U0001F4CD View on Google Maps]({get_google_maps_link(row['name'], row['address'])})")
            st.markdown("---")

        st.subheader("\U0001F50D Search for Medicine")
        med_query = st.text_input("Enter medicine name or symptom:")
        if med_query:
            sem_matches = get_semantic_matches(med_query)
            if sem_matches:
                for med, score in sem_matches:
                    # ✅ Smaller font version of alternative medicines
                    st.markdown(
                        f"<span style='font-size:16px;'><b>{med}</b> <i>(similarity: {score:.2f})</i></span>",
                        unsafe_allow_html=True
                    )
                    matching_inventory = inventory_df[inventory_df['medicine_name'].str.lower() == med.lower()]
                    if not matching_inventory.empty:
                        st.write("Available in:")
                        matching_pharmacies = matching_inventory['added_by'].unique().tolist()
                        for pharmacy_name in matching_pharmacies:
                            matched_pharmacy = process.extractOne(pharmacy_name, pharmacy_df['name'].tolist(), score_cutoff=80)
                            if matched_pharmacy:
                                match_row = pharmacy_df[pharmacy_df['name'] == matched_pharmacy[0]].iloc[0]
                                distance_km = geodesic((user_lat, user_lon), (match_row['latitude'], match_row['longitude'])).km
                                st.markdown(f"**{match_row['name']}** - {match_row['address']}")
                                st.markdown(f"\U0001F4CD Distance: `{distance_km:.2f} km`")
                                st.markdown(f"[View on Maps]({get_google_maps_link(match_row['name'], match_row['address'])})")
                                st.markdown("---")
                            else:
                                st.markdown(f"\U0001F4E6 Pharmacy: `{pharmacy_name}` - stock available (location not matched on map)")
                    else:
                        st.warning(f"No pharmacy has '{med}' in inventory.")

# ========== REMINDER SECTION ==========
st.subheader("\u23F0 Set SMS Medicine Reminder")
with st.form("reminder_form"):
    phone = st.text_input("Phone Number (with country code, e.g., +91...)")
    reminder_med = st.text_input("Medicine Name")
    reminder_time = st.text_input("Reminder Time (e.g., 9:00 AM)")
    if st.form_submit_button("Send SMS Reminder"):
        if phone and reminder_med and reminder_time:
            msg = f"Reminder: Take your medicine '{reminder_med}' at {reminder_time}."
            send_sms(phone, msg)
            st.success("\U0001F4F2 SMS Reminder Sent!")
        else:
            st.error("Please fill all fields.")
