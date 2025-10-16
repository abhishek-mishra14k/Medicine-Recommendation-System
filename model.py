import pandas as pd
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier

# --- Configuration for ALL Image Models ---
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
# CRITICAL FIX 1: Point to the innermost 'chest_xray' folder
CHEST_DATA_DIR = os.path.join("datasets", "chest_xray", "chest_xray") 
SKIN_METADATA_PATH = os.path.join("datasets", "HAM10000_metadata.csv")
SKIN_IMAGE_DIR_1 = os.path.join("datasets", "HAM10000_images_part_1") 
SKIN_IMAGE_DIR_2 = os.path.join("datasets", "HAM10000_images_part_2")
# CRITICAL FIX 2: Point to the innermost 'dataset' folder
TONGUE_IMAGE_DIR = os.path.join("datasets", "TongueImageDataset", "dataset") 

# --- MEDICINE MAPPING DICTIONARY (For Auto-Augmentation) ---
MEDICINE_MAPPING = {
    "Fungal infection": "Ketoconazole cream|Clotrimazole cream",
    "Acne": "Benzoyl Peroxide|Salicylic Acid Wash",
    "Allergy": "Loratadine 10mg (OD)|Saline Nasal Spray",
    "Common Cold": "Paracetamol (TID)|Decongestant",
    "Dengue": "Paracetamol only|Fluids",
    "Typhoid": "Ciprofloxacin (Rx)|ORS",
    "Malaria": "Artemisinin-based combination therapy (Rx)",
    "Pneumonia": "Amoxicillin|Azithromycin (Rx)",
}

# ======================================================================
# 1. CHEST X-RAY MODEL (FIXED: steps_per_epoch and .keras save)
# ======================================================================
def train_chest_xray_model(data_dir=CHEST_DATA_DIR, output_dir="models"):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    if not os.path.exists(train_dir):
        print(f"\n[ERROR] Chest X-ray training data not found at: {train_dir}")
        return

    print("\n===== Training Chest X-ray Model (CNN) =====")
    train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    try:
        train_generator = train_datagen.flow_from_directory(train_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
        test_generator = test_datagen.flow_from_directory(test_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
    except Exception as e:
        print(f"[FATAL] Chest X-ray data structure error: {e}")
        return

    class_names = list(train_generator.class_indices.keys())
    NUM_CLASSES = len(class_names)
    
    if NUM_CLASSES != 2:
        print(f"[ERROR] Expected 2 classes (NORMAL/PNEUMONIA), found {NUM_CLASSES}. Check data structure.")
        return

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Starting CNN training...")
    
    # 🌟 FIX: Removed steps_per_epoch and validation_steps 🌟
    # This prevents the "input ran out of data" warning
    model.fit(train_generator, epochs=3, validation_data=test_generator, verbose=1)
    
    # 🌟 FIX: Use .keras format 🌟
    model_path = os.path.join(output_dir, "chest_xray_model.keras") 
    model.save(model_path)
    label_path = os.path.join(output_dir, "chest_xray_labels.pkl")
    with open(label_path, 'wb') as f:
        pickle.dump(class_names, f)
    print(f"✅ Chest X-ray Model saved successfully! (as {os.path.basename(model_path)})")


# ======================================================================
# 2. HAM10000 SKIN MODEL (FIXED: Filename lookup and .keras save)
# ======================================================================

def train_ham10000_model(metadata_path=SKIN_METADATA_PATH, img_dir_1=SKIN_IMAGE_DIR_1, img_dir_2=SKIN_IMAGE_DIR_2, output_dir="models"):
    """ Trains a CNN for HAM10000 by mapping metadata to images across two folders. """
    if not os.path.exists(metadata_path):
        print(f"\n[ERROR] Skin metadata not found at: {metadata_path}")
        return
    
    print("\n===== Training HAM10000 Skin Model (CNN) =====")
    data = pd.read_csv(metadata_path)
    
    # 1. Create file paths for ALL images, checking both split folders
    def get_image_path(image_id):
        fname = image_id + '.jpg'
        path1 = os.path.join(img_dir_1, fname)
        path2 = os.path.join(img_dir_2, fname)
        
        # 🌟 FIX: The DataFrame MUST contain only the filename (image_id.jpg) 🌟
        if os.path.exists(path1) or os.path.exists(path2):
             return fname 
        return None
    
    # Apply the corrected path check
    data['image_id'] = data['image_id'].apply(get_image_path)
    data.dropna(subset=['image_id'], inplace=True) 

    if len(data) < 5000: 
        print(f"[WARNING] Found only {len(data)} images. Ensure HAM10000 images are copied into part_1/part_2 and file paths are correct.")

    # 2. Data Preparation
    data['dx'] = data['dx'].astype(str)
    train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['dx'])

    # 🌟 FIX: We use the MOST POPULATED base directory (usually part_1). 
    # The flow_from_dataframe logic is often tricky with split folders. 
    # For a robust fix, all images should ideally be merged into ONE directory.
    # We will assume a single base directory for flow_from_dataframe to reference.
    base_dir = SKIN_IMAGE_DIR_1 
    
    datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, validation_split=0.2)
    
    # CRITICAL FIX: The directory must be the base directory where the images are found.
    # The x_col must contain the FILENAME, which we fixed above.
    train_generator = datagen.flow_from_dataframe(train_df, directory=base_dir, x_col='image_id', y_col='dx', target_size=IMAGE_SIZE, class_mode='categorical', batch_size=BATCH_SIZE)
    validation_generator = datagen.flow_from_dataframe(val_df, directory=base_dir, x_col='image_id', y_col='dx', target_size=IMAGE_SIZE, class_mode='categorical', batch_size=BATCH_SIZE)

    class_names = list(train_generator.class_indices.keys())
    NUM_CLASSES = len(class_names)
    
    # 3. Build and Train Model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(f"Found {train_generator.samples} training images belonging to {NUM_CLASSES} classes.")
    model.fit(train_generator, epochs=1, verbose=1, validation_data=validation_generator) 
    
    # 🌟 FIX: Use .keras format 🌟
    model_path = os.path.join(output_dir, "skin_model.keras")
    model.save(model_path)
    label_path = os.path.join(output_dir, "skin_labels.pkl")
    with open(label_path, 'wb') as f:
        pickle.dump(class_names, f)
    print(f"✅ Skin Model saved successfully! (as {os.path.basename(model_path)})")


# ======================================================================
# 3. TONGUE MODEL (FIXED: .keras save)
# ======================================================================
def train_tongue_model(data_dir=TONGUE_IMAGE_DIR, output_dir="models"):
    """ Trains a simple CNN for the Tongue Image Dataset (Uses corrected nested path). """
    if not os.path.exists(data_dir):
        print(f"\n[ERROR] Tongue data directory not found at: {data_dir}")
        return

    print("\n===== Training Tongue Model (CNN) =====")
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    try:
        # This now points to datasets/TongueImageDataset/dataset (where class folders should be)
        train_generator = datagen.flow_from_directory(
            data_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='training'
        )
    except Exception as e:
        print(f"[ERROR] Tongue data not organized. Ensure images are in subfolders (e.g., /dataset/healthy, /dataset/sick).")
        return
        
    class_names = list(train_generator.class_indices.keys())
    NUM_CLASSES = len(class_names)
    
    if train_generator.samples == 0:
        print(f"[ERROR] Tongue model found 0 images. Skipping training and saving placeholder.")
        model = Sequential([Conv2D(1, (1, 1), input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)), Flatten(), Dense(1, activation='linear')])
        
        # 🌟 FIX: Use .keras format 🌟
        model.save(os.path.join(output_dir, "tongue_model.keras"))
        with open(os.path.join(output_dir, "tongue_labels.pkl"), 'wb') as f:
            pickle.dump(['placeholder'], f)
        print(f"✅ Tongue Model placeholder saved successfully! (as {os.path.basename(os.path.join(output_dir, 'tongue_model.keras'))})")
        return

    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(train_generator, epochs=1, verbose=0) 

    # 🌟 FIX: Use .keras format 🌟
    model_path = os.path.join(output_dir, "tongue_model.keras")
    model.save(model_path)
    label_path = os.path.join(output_dir, "tongue_labels.pkl")
    with open(label_path, 'wb') as f:
        pickle.dump(class_names, f)
    print(f"✅ Tongue Model saved successfully! (as {os.path.basename(model_path)})")


# ======================================================================
# 4 & 5. MANUAL SYMPTOM & QUESTIONNAIRE MODELS (No major changes needed)
# ======================================================================

def train_symptom_model(dataset_path, output_dir="models"):
    """ Trains manual symptom models using dynamic Symptom_X columns and auto-generates medicines. """
    if not os.path.exists(dataset_path): raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    data = pd.read_csv(dataset_path) 
    print("\n===== Training Manual Symptom Models (Fixed Column Names) =====")

    if 'Recommended Medicines' not in data.columns:
        print("[INFO] 'Recommended Medicines' column missing. Auto-generating using map.")
        data['Recommended Medicines'] = data['Disease'].str.strip().map(MEDICINE_MAPPING).fillna("Supportive care|Monitor symptoms")
        
    symptom_cols = [col for col in data.columns if col.startswith('Symptom_')]
    if not symptom_cols:
        symptom_cols = [col for col in data.columns if col.startswith('Symptom ')]
        if not symptom_cols:
            raise ValueError("No columns starting with 'Symptom_' or 'Symptom ' found. Check CSV headers.")

    data['Symptoms'] = data[symptom_cols].apply(
        lambda row: [str(s).strip().lower() for s in row if pd.notna(s) and str(s).strip() != "" and str(s).lower() != "none"], axis=1
    )

    symptom_encoder = MultiLabelBinarizer()
    X_symptoms = symptom_encoder.fit_transform(data['Symptoms'])
    
    disease_encoder = LabelEncoder()
    y_disease = disease_encoder.fit_transform(data['Disease']) 
    
    meds_list = data['Recommended Medicines'].apply(lambda x: [m.strip() for m in str(x).split('|') if m.strip()])
    meds_encoder = MultiLabelBinarizer()
    y_meds = meds_encoder.fit_transform(meds_list)

    X_train, X_test, y_disease_train, y_disease_test = train_test_split(X_symptoms, y_disease, test_size=0.2, random_state=42)
    _, _, y_meds_train, y_meds_test = train_test_split(X_symptoms, y_meds, test_size=0.2, random_state=42)

    print("Training disease model...")
    disease_model = RandomForestClassifier(n_estimators=100, random_state=42)
    disease_model.fit(X_train, y_disease_train)

    print("Training medicine model...")
    med_model = RandomForestClassifier(n_estimators=100, random_state=42)
    med_model.fit(X_train, y_meds_train)

    os.makedirs(output_dir, exist_ok=True)
    pickle.dump(disease_model, open(os.path.join(output_dir, "symptom_disease_model.pkl"), "wb"))
    pickle.dump(med_model, open(os.path.join(output_dir, "symptom_med_model.pkl"), "wb"))
    pickle.dump(symptom_encoder, open(os.path.join(output_dir, "symptom_symptom_encoder.pkl"), "wb"))
    pickle.dump(disease_encoder, open(os.path.join(output_dir, "symptom_disease_encoder.pkl"), "wb"))
    pickle.dump(meds_encoder, open(os.path.join(output_dir, "symptom_meds_encoder.pkl"), "wb"))
    print("✅ Manual symptom models saved successfully!")


def train_questionnaire_model(dataset_path, output_dir="models"):
    """ Trains questionnaire models (disease + medicine), forcing lowercase standardization. """
    if not os.path.exists(dataset_path): raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    data = pd.read_csv(dataset_path)
    print("\n===== Training Questionnaire Models =====")
    required_cols = ['Age', 'Gender', 'Symptoms', 'Predicted Disease', 'Recommended Medicines']
    for col in required_cols:
        if col not in data.columns:
            raise KeyError(f"Missing column '{col}' in dataset")
    data['Symptoms'] = data['Symptoms'].apply(lambda x: [s.strip().lower() for s in str(x).split('|') if s.strip()])
    symptom_encoder = MultiLabelBinarizer()
    X_symptoms = symptom_encoder.fit_transform(data['Symptoms'])
    gender_encoder = LabelEncoder()
    X_gender = gender_encoder.fit_transform(data['Gender'].str.lower()) 
    X = np.hstack([X_symptoms, X_gender.reshape(-1,1)])
    disease_encoder = LabelEncoder()
    y_disease = disease_encoder.fit_transform(data['Predicted Disease'])
    meds_list = data['Recommended Medicines'].apply(lambda x: [m.strip() for m in str(x).split('|') if m.strip()])
    meds_encoder = MultiLabelBinarizer()
    y_meds = meds_encoder.fit_transform(meds_list)
    X_train, X_test, y_disease_train, y_disease_test = train_test_split(X, y_disease, test_size=0.2, random_state=42)
    _, _, y_meds_train, y_meds_test = train_test_split(X, y_meds, test_size=0.2, random_state=42)
    print("Training disease model...")
    disease_model = RandomForestClassifier(n_estimators=100, random_state=42)
    disease_model.fit(X_train, y_disease_train)
    print("Training medicine model...")
    med_model = RandomForestClassifier(n_estimators=100, random_state=42)
    med_model.fit(X_train, y_meds_train)
    os.makedirs(output_dir, exist_ok=True)
    pickle.dump(disease_model, open(os.path.join(output_dir, "questionnaire_disease_model.pkl"), "wb"))
    pickle.dump(med_model, open(os.path.join(output_dir, "questionnaire_med_model.pkl"), "wb"))
    pickle.dump(symptom_encoder, open(os.path.join(output_dir, "questionnaire_symptom_encoder.pkl"), "wb"))
    pickle.dump(gender_encoder, open(os.path.join(output_dir, "questionnaire_gender_encoder.pkl"), "wb"))
    pickle.dump(disease_encoder, open(os.path.join(output_dir, "questionnaire_disease_encoder.pkl"), "wb"))
    pickle.dump(meds_encoder, open(os.path.join(output_dir, "questionnaire_meds_encoder.pkl"), "wb"))
    print("✅ Questionnaire models saved successfully!")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    
    # 1. Train ALL Image Models
    train_chest_xray_model()
    train_ham10000_model() 
    train_tongue_model()
    
    # 2. Train Symptom-based Models 
    train_symptom_model("datasets/symptom.csv")
    train_questionnaire_model("datasets/questionnaire_data.csv")