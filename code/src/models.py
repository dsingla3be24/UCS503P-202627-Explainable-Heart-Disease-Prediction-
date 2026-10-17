import json
import pymysql
import uuid
from datetime import datetime

# --- RAW MYSQL DATABASE FUNCTIONS ---
def get_db_connection():
    return pymysql.connect(
        host='localhost',
        port=3307,
        user='root',
        password='',
        db='cardio_ai_db',
        cursorclass=pymysql.cursors.DictCursor
    )

def get_or_create_patient(patient_name, session_id, provided_patient_id=None):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # If a specific patient ID is provided, look it up first
            if provided_patient_id and provided_patient_id.strip():
                cursor.execute("SELECT id, hospital_patient_id, name FROM patients WHERE hospital_patient_id = %s LIMIT 1", (provided_patient_id.strip(),))
                existing = cursor.fetchone()
                if existing:
                    # Update patient name if they provided a new one for the same ID
                    if patient_name and patient_name.strip() and patient_name != existing['name'] and not patient_name.startswith("Patient "):
                        cursor.execute("UPDATE patients SET name = %s WHERE id = %s", (patient_name, existing['id']))
                        final_name = patient_name
                    else:
                        final_name = existing['name']
                    return existing['id'], existing['hospital_patient_id'], final_name

            # Create new patient (if no ID provided or ID not found)
            hospital_patient_id = provided_patient_id.strip() if provided_patient_id and provided_patient_id.strip() else f"PT-{session_id.upper()}"
            if not patient_name or not patient_name.strip():
                patient_name = f"Patient {session_id.upper()}"
                
            sql_patient = "INSERT INTO patients (hospital_patient_id, name, created_at) VALUES (%s, %s, NOW())"
            cursor.execute(sql_patient, (hospital_patient_id, patient_name))
            patient_id = cursor.lastrowid
            
        conn.commit()
        return patient_id, hospital_patient_id, patient_name
    finally:
        conn.close()

def get_patient_last_demographics(patient_db_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT age, sex, recorded_at FROM clinical_records WHERE patient_id = %s ORDER BY recorded_at DESC LIMIT 1", (patient_db_id,))
            record = cursor.fetchone()
            if record:
                return record['age'], record['sex'], record['recorded_at']
            return None, None, None
    finally:
        conn.close()

def insert_patient_record(patient_id, raw_features, prob, top_features_str, pdf_filename):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # 2. Insert into clinical_records
            sql_record = """INSERT INTO clinical_records 
                (patient_id, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, recorded_at) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())"""
            cursor.execute(sql_record, (
                patient_id, raw_features['age'], raw_features['sex'], raw_features['cp'],
                raw_features['trestbps'], raw_features['chol'], raw_features['fbs'],
                raw_features['restecg'], raw_features['thalach'], raw_features['exang'],
                raw_features['oldpeak'], raw_features['slope'], raw_features['ca'], raw_features['thal']
            ))
            record_id = cursor.lastrowid
            
            # 3. Insert into ai_diagnoses
            prediction_class = 1 if prob >= 0.5 else 0
            
            sql_diagnosis = """INSERT INTO ai_diagnoses 
                (record_id, risk_probability, prediction_class, primary_drivers, pdf_report_path, diagnosed_at) 
                VALUES (%s, %s, %s, %s, %s, NOW())"""
            cursor.execute(sql_diagnosis, (
                record_id, float(prob), prediction_class, top_features_str[:255], pdf_filename
            ))
        
        conn.commit()
    finally:
        conn.close()

def search_patient_history(search_query):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Search by name (LIKE) or exact hospital_patient_id
            if search_query:
                sql = """
                    SELECT 
                        p.hospital_patient_id, 
                        p.name, 
                        p.created_at,
                        cr.age,
                        cr.sex,
                        cr.recorded_at,
                        ad.risk_probability,
                        ad.prediction_class,
                        ad.pdf_report_path
                    FROM patients p
                    LEFT JOIN clinical_records cr ON p.id = cr.patient_id
                    LEFT JOIN ai_diagnoses ad ON cr.id = ad.record_id
                    WHERE p.name LIKE %s OR p.hospital_patient_id = %s
                    ORDER BY cr.recorded_at DESC
                """
                like_query = f"%{search_query}%"
                cursor.execute(sql, (like_query, search_query))
            else:
                sql = """
                    SELECT 
                        p.hospital_patient_id, 
                        p.name, 
                        p.created_at,
                        cr.age,
                        cr.sex,
                        cr.recorded_at,
                        ad.risk_probability,
                        ad.prediction_class,
                        ad.pdf_report_path
                    FROM patients p
                    LEFT JOIN clinical_records cr ON p.id = cr.patient_id
                    LEFT JOIN ai_diagnoses ad ON cr.id = ad.record_id
                    ORDER BY cr.recorded_at DESC LIMIT 100
                """
                cursor.execute(sql)
                
            results = cursor.fetchall()
            
            # Group by patient to format nicely for the frontend
            patients_dict = {}
            for row in results:
                pid = row['hospital_patient_id']
                if pid not in patients_dict:
                    patients_dict[pid] = {
                        'hospital_patient_id': pid,
                        'name': row['name'],
                        'created_at': row['created_at'].isoformat() if row['created_at'] else None,
                        'history': []
                    }
                
                # If they have a clinical record, add it to their history
                if row['recorded_at']:
                    patients_dict[pid]['history'].append({
                        'age': row['age'],
                        'sex': 'Male' if row['sex'] == 0 else 'Female',
                        'recorded_at': row['recorded_at'].isoformat(),
                        'risk_probability': row['risk_probability'],
                        'prediction_class': row['prediction_class'],
                        'pdf_report_path': row['pdf_report_path']
                    })
            
            return list(patients_dict.values())
    finally:
        conn.close()

# --- MACHINE LEARNING MODELS ---
# TensorFlow/Keras layers resolve to ./lib/tensorflow.py by default (a lightweight
# NumPy / scikit-learn emulator), so every builder below runs on CPU without the
# native TensorFlow wheel. `hp` is only supplied by the --tune flow and is ignored
# when hyper-parameter tuning is disabled.

def build_cnn(input_shape, hp=None):
    """Spatial baseline: 2D convolutional branch over the GAF (GASF) image."""
    import tensorflow as tf
    filters = int(hp.get('cnn_filters', 32)) if hp else 32
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.models.Model(inputs=inputs, outputs=outputs, name="CNN")


def build_cnn_lstm(input_shape, hp=None):
    """Temporal baseline: 2D convolution + LSTM over the GAF (GASF) image."""
    import tensorflow as tf
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.models.Model(inputs=inputs, outputs=outputs, name="CNN_LSTM")


def build_cnn_lstm_attention(input_shape, hp=None):
    """Full model: 2D convolution + LSTM + multi-head self-attention."""
    import tensorflow as tf
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.models.Model(inputs=inputs, outputs=outputs, name="CNN_LSTM_Attention")


def compile_model(model):
    return model

