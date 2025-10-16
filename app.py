from flask import Flask, render_template, request, jsonify, url_for
import os, sys, pickle
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import tensorflow as tf 
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# ======================================================================
# RULE-BASED ENGINE (UNCHANGED)
# ======================================================================
RULES_ENGINE = {
    # Existing rules remain here...
    ('fever',): {"disease": "Common Fever", "meds": ["Paracetamol 500mg (PRN)", "Rest"], "cause": "Non-specific systemic response to minor infection or stress."},
    ('headache',): {"disease": "Tension Headache", "meds": ["Ibuprofen 200mg (PRN)", "Hydration"], "cause": "Muscle contraction and strain in the head and neck, often stress-induced."},
    ('fatigue', 'body pain'): {
        "disease": "Muscular Aches",
        "meds": ["Paracetamol (PRN)", "Magnesium Supplement"],
        "cause": "Generalized aches, often due to physical exertion or onset of a mild viral illness."
    },
    ('dizziness', 'fatigue'): {
        "disease": "Mild Dehydration",
        "meds": ["Oral Rehydration Salts (ORS)", "Increased Water Intake"],
        "cause": "Low fluid intake leading to reduced blood volume and lightheadedness."
    },
    ('fever', 'fatigue'): {
        "disease": "Viral Syndrome (Mild)",
        "meds": ["Paracetamol 500mg (TID)", "Rest"],
        "cause": "Early stage of a viral infection (e.g., flu or common cold)."
    },
    ('fever', 'fatigue', 'headache'): {
        "disease": "Early Flu",
        "meds": ["Paracetamol (TID)", "Fluids", "Rest"],
        "cause": "Classic combination indicating a systemic viral infection, requiring symptom management."
    },
    ('fever', 'fatigue', 'headache', 'body pain'): {
        "disease": "Influenza (Flu)",
        "meds": ["Paracetamol (TID)", "Fluids", "Oseltamivir (if prescribed)"],
        "cause": "Severe systemic symptoms indicating a definite influenza infection."
    },
    ('body pain', 'dizziness', 'fatigue'): {
        "disease": "Post-Viral Fatigue",
        "meds": ["ORS", "Vitamin B Complex", "Sleep Hygiene"],
        "cause": "Persistent systemic symptoms often remaining after an acute infection has passed."
    },
    ('cough',): {
        "disease": "Irritant Cough",
        "meds": ["Dextromethorphan syrup (TID)", "Throat Lozenges"],
        "cause": "Minor irritation or inflammation of the upper respiratory tract."
    },
    ('cold',): {
        "disease": "Minor Viral Rhinitis",
        "meds": ["Pseudoephedrine (PRN)", "Zinc Supplements"],
        "cause": "Rhinovirus infection causing nasal congestion and mild systemic symptoms."
    },
    ('sore throat',): {
        "disease": "Pharyngitis (Mild)",
        "meds": ["Throat Lozenges", "Warm Saltwater Gargle"],
        "cause": "Inflammation of the throat lining, most often due to a viral infection."
    },
    ('shortness of breath',): {
        "disease": "Respiratory Observation",
        "meds": ["Avoid Exertion", "Seek Medical Attention IMMEDIATELY"],
        "cause": "Difficulty breathing, requires urgent assessment to rule out severe issues like asthma or heart problems."
    },
    ('chest pain',): {
        "disease": "Musculoskeletal Pain",
        "meds": ["Ibuprofen 400mg (PRN)", "Rest"],
        "cause": "Pain originating from the chest wall muscles or rib cage (costochondritis)."
    },
    ('cough', 'cold'): {
        "disease": "Acute Nasal Drip",
        "meds": ["Decongestant Spray", "Antihistamine"],
        "cause": "Post-nasal drip irritation following nasal congestion."
    },
    ('cough', 'sore throat'): {
        "disease": "Bronchial Irritation",
        "meds": ["Dextromethorphan syrup", "Steam Inhalation"],
        "cause": "Inflammation of the airways, commonly accompanying a sore throat."
    },
    ('cough', 'chest pain'): {
        "disease": "Chest Muscle Strain",
        "meds": ["Rest", "Topical Pain Relief", "Avoid Heavy Lifting"],
        "cause": "Muscular discomfort exacerbated by forceful coughing."
    },
    ('cold', 'cough', 'sore throat'): {
        "disease": "Common Cold Syndrome",
        "meds": ["Paracetamol", "Decongestant Spray", "Throat Lozenges"],
        "cause": "Standard combination of upper respiratory tract infection symptoms."
    },
    ('chest pain', 'shortness of breath'): {
        "disease": "Cardiopulmonary Warning",
        "meds": ["IMMEDIATE EMERGENCY CALL", "Do Not Exert"],
        "cause": "Critical warning; requires urgent medical assessment to rule out heart or lung issues."
    },
    ('shortness of breath', 'sore throat'): {
        "disease": "Laryngeal Edema",
        "meds": ["Seek Emergency Care"],
        "cause": "Swelling or inflammation restricting the airway near the throat."
    },
    ('cold', 'shortness of breath'): {
        "disease": "Severe Congestion",
        "meds": ["Decongestant", "Hot Shower/Steam", "Doctor Consult"],
        "cause": "Nasal congestion severe enough to impact air intake."
    },
    ('diarrhea',): {
        "disease": "Acute Bowel Upset",
        "meds": ["Oral Rehydration Salts (ORS)", "Loperamide (if severe)"],
        "cause": "Sudden inflammation or irritation of the large intestine."
    },
    ('loss of appetite',): {
        "disease": "Non-specific Anorexia",
        "meds": ["Nutritional Supplements", "Monitor Energy Levels"],
        "cause": "Lack of desire to eat, often due to stress, illness, or fatigue."
    },
    ('nausea',): {
        "disease": "Gastric Discomfort",
        "meds": ["Ginger Capsules", "Antiemetic Tablet (PRN)"],
        "cause": "Feeling of sickness/unrest in the stomach."
    },
    ('stomach pain',): {
        "disease": "Mild Cramping",
        "meds": ["Antispasmodic (PRN)", "Warm Compress"],
        "cause": "Muscular contractions in the abdomen due to trapped gas or minor irritation."
    },
    ('vomiting',): {
        "disease": "Gastric Irritation",
        "meds": ["Antiemetic Tablet", "Avoid Solid Foods"],
        "cause": "Forceful expulsion of stomach contents, requiring fluid replacement."
    },
    ('nausea', 'vomiting', 'diarrhea'): {
        "disease": "Viral Gastroenteritis",
        "meds": ["ORS", "Bland Diet (BRAT)", "Rest"],
        "cause": "Highly contagious inflammation of the stomach and small intestine."
    },
    ('loss of appetite', 'stomach pain'): {
        "disease": "Gastritis/Peptic Irritation",
        "meds": ["Omeprazole 20mg (OD)", "Antacids"],
        "cause": "Inflammation or mild ulceration of the stomach lining."
    },
    ('diarrhea', 'stomach pain', 'vomiting'): {
        "disease": "Food Poisoning",
        "meds": ["ORS", "Barrowed Antibiotic (Rx)", "Bland Diet"],
        "cause": "Acute intestinal illness caused by consuming contaminated food or water."
    },
    ('loss of appetite', 'nausea', 'stomach pain'): {
        "disease": "Chronic Dyspepsia",
        "meds": ["Omeprazole 20mg", "Dietary Modification"],
        "cause": "Persistent or recurrent pain or discomfort centered in the upper abdomen."
    },
    ('diarrhea', 'loss of appetite', 'nausea', 'stomach pain', 'vomiting'): {
        "disease": "Severe Acute Infection",
        "meds": ["Ciprofloxacin (Rx)", "IV Fluids (Urgent)"],
        "cause": "Generalized severe illness often caused by a serious bacterial infection."
    },
    ('rash',): {
        "disease": "Non-specific Rash",
        "meds": ["Avoid Sunlight", "Topical Moisturizer"],
        "cause": "General skin irritation or allergic reaction of unknown origin."
    },
    ('itching',): {
        "disease": "Minor Pruritus",
        "meds": ["Calamine Lotion", "Hydration"],
        "cause": "Localized dryness or mild skin irritation causing the urge to scratch."
    },
    ('redness',): {
        "disease": "Localized Inflammation",
        "meds": ["Cool Compress", "Aloe Vera Gel"],
        "cause": "Increased blood flow to a specific area due to injury or infection."
    },
    ('swelling',): {
        "disease": "Edema (Localized)",
        "meds": ["Elevation of Area", "Cold Compress"],
        "cause": "Fluid buildup in tissue, often due to injury or mild allergic reaction."
    },
    ('itching', 'rash'): {
        "disease": "Contact Dermatitis (Mild)",
        "meds": ["Hydrocortisone Cream", "Avoid Irritant"],
        "cause": "Inflammation where skin directly contacted an allergen (e.g., detergent, jewelry)."
    },
    ('itching', 'redness', 'rash'): {
        "disease": "Eczema Flare-up",
        "meds": ["Moisturizing Cream (BD)", "Antihistamine (PRN)"],
        "cause": "Common chronic skin condition causing redness and flaking."
    },
    ('redness', 'swelling'): {
        "disease": "Inflammatory Reaction",
        "meds": ["Ibuprofen (PRN)", "Monitor for Fever"],
        "cause": "Sign of localized infection or a strong inflammatory response."
    },
    ('itching', 'redness', 'swelling', 'rash'): {
        "disease": "Severe Urticaria (Hives)",
        "meds": ["Cetirizine 10mg (OD)", "Emergency Consult"],
        "cause": "Acute, systemic allergic reaction requiring immediate medical guidance."
    },
    ('itching', 'swelling'): {
        "disease": "Insect Bite/Sting",
        "meds": ["Topical Antiseptic", "Cold Compress"],
        "cause": "Immediate reaction to venom or saliva introduced by a bug bite."
    },
}
# ==========================
# Helper Functions
# ==========================
def load_pickle(filepath):
    """Safely loads a pickle file."""
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[ERROR] Unable to load {filepath}: {str(e)}", file=sys.stderr)
        return None

# 🌟 FIX: Updated to support new .keras format 🌟
def load_cnn_model(model_filepath_base, label_filepath):
    """Safely loads a Keras/TensorFlow model and its label map."""
    model_filepath = model_filepath_base + '.keras'
    if not os.path.exists(model_filepath):
        # Fallback to check for legacy .h5 file if the .keras file isn't found
        model_filepath = model_filepath_base + '.h5'
        if not os.path.exists(model_filepath):
            print(f"[ERROR] CNN Model not found at either {model_filepath_base}.keras or .h5.", file=sys.stderr)
            return None, None
            
    try:
        # tf.keras.models.load_model handles both .keras and .h5 formats
        model = tf.keras.models.load_model(model_filepath)
        label_map = load_pickle(label_filepath) 
        print(f"[INFO] CNN Model loaded from {os.path.basename(model_filepath)}.")
        return model, label_map
    except Exception as e:
        print(f"[ERROR] Unable to load CNN model or label map: {str(e)}", file=sys.stderr)
        return None, None


# ==========================
# Load Models and Encoders
# ==========================
models_dir = "models"
datasets_dir = "datasets"

# Symptom Description Lookup
try:
    DESCRIPTION_DF = pd.read_csv(os.path.join(datasets_dir, "symptom_Description.csv"))
    DESCRIPTION_DF.columns = ['Disease', 'Description'] 
    DESCRIPTION_DICT = DESCRIPTION_DF.set_index('Disease').to_dict()['Description']
except Exception as e:
    DESCRIPTION_DICT = {}
    print(f"[ERROR] Could not load symptom_Description.csv: {e}", file=sys.stderr)


q_models = {
    "disease_model": load_pickle(os.path.join(models_dir, "questionnaire_disease_model.pkl")),
    "med_model": load_pickle(os.path.join(models_dir, "questionnaire_med_model.pkl")),
    "gender_encoder": load_pickle(os.path.join(models_dir, "questionnaire_gender_encoder.pkl")),
    "symptom_encoder": load_pickle(os.path.join(models_dir, "questionnaire_symptom_encoder.pkl")),
    "disease_encoder": load_pickle(os.path.join(models_dir, "questionnaire_disease_encoder.pkl")),
    "meds_encoder": load_pickle(os.path.join(models_dir, "questionnaire_meds_encoder.pkl"))
}

s_models = {
    "disease_model": load_pickle(os.path.join(models_dir, "symptom_disease_model.pkl")),
    "med_model": load_pickle(os.path.join(models_dir, "symptom_med_model.pkl")),
    "symptom_encoder": load_pickle(os.path.join(models_dir, "symptom_symptom_encoder.pkl")),
    "disease_encoder": load_pickle(os.path.join(models_dir, "symptom_disease_encoder.pkl")),
    "meds_encoder": load_pickle(os.path.join(models_dir, "symptom_meds_encoder.pkl"))
}

# --- Load IMAGE Models (UPDATED FILENAME) ---
CHEST_MODEL, CHEST_LABELS = load_cnn_model(
    os.path.join(models_dir, "chest_xray_model"), os.path.join(models_dir, "chest_xray_labels.pkl")
)
SKIN_MODEL, SKIN_LABELS = load_cnn_model(
    os.path.join(models_dir, "skin_model"), os.path.join(models_dir, "skin_labels.pkl")
)
TONGUE_MODEL, TONGUE_LABELS = load_cnn_model(
    os.path.join(models_dir, "tongue_model"), os.path.join(models_dir, "tongue_labels.pkl")
)
# -------------------------

if not all(q_models.values()) or not all(s_models.values()):
    print("[WARNING] Core ML models failed to load. Run train_models.py.", file=sys.stderr)
if not CHEST_MODEL:
    print("[WARNING] Chest X-ray model failed to load.", file=sys.stderr)
if not SKIN_MODEL:
    print("[WARNING] Skin model failed to load.", file=sys.stderr)
if not TONGUE_MODEL:
    print("[WARNING] Tongue model failed to load.", file=sys.stderr)
if CHEST_MODEL or SKIN_MODEL or TONGUE_MODEL:
    print("[INFO] At least one Image Model loaded successfully.")

# ==========================
# ROUTES (UNCHANGED)
# ==========================
@app.route("/")
def home():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/symptom")
def symptom():
    return render_template("symptom.html")

# ==========================
# IMAGE HELPER FUNCTIONS (UNCHANGED)
# ==========================
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(filepath):
    """Resizes and normalizes the image for the CNN model."""
    try:
        img = Image.open(filepath).convert('RGB')
        img = img.resize((150, 150))
        img_array = np.array(img, dtype='float32')
        img_array = img_array / 255.0 
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"[ERROR] Image preprocessing failed: {e}", file=sys.stderr)
        return None

# ==========================
# IMAGE UPLOAD & PREDICT ROUTE (UNCHANGED LOGIC)
# ==========================
@app.route('/upload_and_predict', methods=['POST'])
def upload_image_and_predict():
    if 'image' not in request.files:
        return jsonify({"error": "400 BAD REQUEST: No image file part found in request."}), 400
        
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type or no file selected (only png/jpg/jpeg allowed)."}), 400
        
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    model_type = request.form.get('model_type', 'CHEST').upper()
    
    model_to_use, labels_to_use = None, None
    model_key = 'CHEST' 

    if model_type == 'SKIN' and SKIN_MODEL:
        model_to_use, labels_to_use = SKIN_MODEL, SKIN_LABELS
        model_key = 'SKIN'
    elif model_type == 'TONGUE' and TONGUE_MODEL:
        model_to_use, labels_to_use = TONGUE_MODEL, TONGUE_LABELS
        model_key = 'TONGUE'
    elif CHEST_MODEL: # Default fallback to CHEST if it exists
        model_to_use, labels_to_use = CHEST_MODEL, CHEST_LABELS
        model_key = 'CHEST'
    
    if not model_to_use:
        return jsonify({"error": f"{model_key} model is not loaded. Please run model.py training for this type."}), 503
        
    processed_array = preprocess_image(filepath)
    if processed_array is None:
        return jsonify({"error": "Failed to process image."}), 500
        
    try:
        predictions = model_to_use.predict(processed_array)
        predicted_class_index = np.argmax(predictions)
        confidence = float(np.max(predictions) * 100)
        
        predicted_disease = "No Diagnosis Found"
        if predicted_class_index < len(labels_to_use):
            predicted_disease = labels_to_use[predicted_class_index]
            
        if model_key == 'CHEST':
            recommendation = ["Seek medical attention immediately." if predicted_disease.upper() == 'PNEUMONIA' else "Result is normal."]
            cause = "Lungs inflammation (Pneumonia)." if predicted_disease.upper() == 'PNEUMONIA' else "No major abnormality."
        elif model_key == 'TONGUE':
            recommendation = ["Tongue analysis requires specialist evaluation.", "Maintain good oral hygiene."]
            cause = f"Predicted {predicted_disease}."
        else: # SKIN
             recommendation = ["Apply topical cream as prescribed.", "Consult a dermatologist."]
             cause = f"Predicted {predicted_disease}."


        return jsonify({
            "status": "success",
            "possible_cause": f"Model: {model_key} | {cause} | Confidence: {confidence:.2f}%",
            "recommendation": recommendation,
            "path": url_for('static', filename=f'uploads/{filename}')
        })

    except Exception as e:
        print(f"[ERROR] CNN Prediction failed ({model_key}): {e}", file=sys.stderr)
        return jsonify({"error": f"Prediction failed ({model_key}): {e}"}), 500

# ==========================
# SYMPTOM PREDICTION HELPER FUNCTIONS (UNCHANGED)
# ==========================

def predict_disease_and_meds_symptom_only(symptoms, models_dict):
    """Prediction for the manual symptom checker, uses RULES_ENGINE first."""
    
    standardized_symptoms = sorted([s.strip().lower() for s in symptoms])
    symptom_key = tuple(standardized_symptoms) 

    if symptom_key in RULES_ENGINE:
        result = RULES_ENGINE[symptom_key]
        return {"disease": result["disease"], "meds": result["meds"], "cause": f"Rule-Based Match: {result['cause']}"}

    if not all([models_dict["disease_model"], models_dict["med_model"], models_dict["symptom_encoder"]]):
        return {"error": "ML models are not loaded. Please run training script."}
        
    try:
        symptom_vector = models_dict["symptom_encoder"].transform([standardized_symptoms])
        disease_pred = models_dict["disease_model"].predict(symptom_vector)
        disease = models_dict["disease_encoder"].inverse_transform(disease_pred)[0]

        med_pred = models_dict["med_model"].predict(symptom_vector)
        meds = models_dict["meds_encoder"].inverse_transform(med_pred)
        
        meds_list = list(meds[0]) if meds is not None and len(meds[0]) > 0 else ["No specific recommendation found."] 
        
        cause_explanation = DESCRIPTION_DICT.get(disease, f"Predicted: {disease}. Detailed cause explanation not found.")

        return {"disease": disease, "meds": meds_list, "cause": cause_explanation}
    except Exception as e:
        print(f"[ERROR] Manual Symptom Prediction failed: {str(e)}", file=sys.stderr)
        return {"error": f"Prediction Error: {str(e)}."}

def predict_disease_and_meds_questionnaire(data, models_dict):
    """Prediction for the questionnaire, returning disease and cause."""
    if not all([models_dict["disease_model"], models_dict["gender_encoder"]]):
        return {"error": "ML models are not loaded for questionnaire. Please run training script."}
        
    X_symptoms = None 
    X_gender = None 

    try:
        symptoms_str_raw = data.get('symptoms', '').strip()
        symptoms_str_cleaned = symptoms_str_raw.replace("'", "").replace('"', '')
        symptom_list_raw = [s.strip() for s in symptoms_str_cleaned.split(',') if s.strip()] 
        symptom_list_standardized = [s.lower() for s in symptom_list_raw] 
        
        if not symptom_list_standardized:
             return {"error": "No valid symptoms were extracted from your input. Please check spelling/format.", 
                     "disease": "Observation", "meds": ["No specific medicine recommended."], "cause": "Input data incomplete."}
            
        X_symptoms = models_dict["symptom_encoder"].transform([symptom_list_standardized])
        
        gender_input = data.get('gender', 'unknown').lower()
        if gender_input not in models_dict["gender_encoder"].classes_:
             gender_input = models_dict["gender_encoder"].classes_[0] 
        
        X_gender = models_dict["gender_encoder"].transform([gender_input])
        
        X_combined = np.hstack([X_symptoms, X_gender.reshape(-1, 1)])
        
        disease_pred = models_dict["disease_model"].predict(X_combined)
        predicted_disease = models_dict["disease_encoder"].inverse_transform(disease_pred)[0]
        
        med_pred = models_dict["med_model"].predict(X_combined)
        meds = models_dict["meds_encoder"].inverse_transform(med_pred)
        meds_list = list(meds[0]) if meds is not None and len(meds[0]) > 0 else ["No specific recommendation found."]

        cause_explanation = DESCRIPTION_DICT.get(
            predicted_disease, 
            f"Predicted: {predicted_disease}. Detailed cause explanation not found."
        )

        return {"disease": predicted_disease, "meds": meds_list, "cause": cause_explanation}
    except Exception as e:
        print(f"[ERROR] Questionnaire Prediction failed: {str(e)}", file=sys.stderr)
        return {"error": f"Prediction Error (Feature or Label Mismatch): {str(e)}"}


# ==========================
# PREDICTION ROUTES (Call Helper Functions)
# ==========================
@app.route('/get_questionnaire_recommendation', methods=['POST'])
def get_questionnaire_recommendation():
    data = request.get_json(force=True)
    if not data.get('age') or not data.get('gender') or not data.get('symptoms'):
        return jsonify({"error": "Age, gender, and symptoms are required."}), 400

    result = predict_disease_and_meds_questionnaire(data, q_models) 
    if "error" in result:
        return jsonify({"error": result["error"]}), 500
    
    return jsonify({
        "status": "success", 
        "possible_cause": result["cause"], 
        "disease": result["disease"], 
        "recommendation": result["meds"]
    })

@app.route('/get_symptom_recommendation', methods=['POST'])
def get_symptom_recommendation():
    data = request.get_json(force=True)
    symptoms = data.get('symptoms', [])
    if not isinstance(symptoms, list) or len(symptoms) == 0:
        return jsonify({"error": "Please select at least one symptom."}), 400

    result = predict_disease_and_meds_symptom_only(symptoms, s_models) 
    if "error" in result:
        return jsonify({"error": result["error"]}), 500
    return jsonify({"status": "success", "possible_cause": result["cause"], "recommendation": result["meds"]})


if __name__ == '__main__':
    if not os.path.exists(models_dir):
        print(f"\n[CRITICAL] Models directory '{models_dir}' not found. Run train_models.py first.", file=sys.stderr)
        sys.exit(1)
        
    app.run(debug=True, port = 8000)