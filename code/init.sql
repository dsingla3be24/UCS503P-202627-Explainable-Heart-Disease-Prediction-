CREATE DATABASE IF NOT EXISTS cardio_ai_db;
USE cardio_ai_db;

CREATE TABLE IF NOT EXISTS patients (
    id INT AUTO_INCREMENT PRIMARY KEY,
    hospital_patient_id VARCHAR(100) UNIQUE,
    name VARCHAR(255),
    created_at DATETIME
);

CREATE TABLE IF NOT EXISTS clinical_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT,
    age FLOAT,
    sex FLOAT,
    cp FLOAT,
    trestbps FLOAT,
    chol FLOAT,
    fbs FLOAT,
    restecg FLOAT,
    thalach FLOAT,
    exang FLOAT,
    oldpeak FLOAT,
    slope FLOAT,
    ca FLOAT,
    thal FLOAT,
    recorded_at DATETIME,
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS ai_diagnoses (
    id INT AUTO_INCREMENT PRIMARY KEY,
    record_id INT,
    risk_probability FLOAT,
    prediction_class INT,
    primary_drivers VARCHAR(255),
    pdf_report_path VARCHAR(255),
    diagnosed_at DATETIME,
    FOREIGN KEY (record_id) REFERENCES clinical_records(id) ON DELETE CASCADE
);
