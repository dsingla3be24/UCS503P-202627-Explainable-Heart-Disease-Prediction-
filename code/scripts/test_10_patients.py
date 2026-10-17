import pandas as pd
import urllib.request
import json
import time

def test_10_patients():
    df = pd.read_csv('heart_disease_cleveland.csv')
    
    # Take 10 patients for testing
    test_df = df.head(10)
    
    url = "http://127.0.0.1:5001/api/predict"
    
    success_count = 0
    
    for i, row in test_df.iterrows():
        patient_name = f"Test Patient {i+1}"
        
        payload = {
            "patient_name": patient_name,
            "patient_type": "new",
            "age": row['age'],
            "sex": row['sex'],
            "cp": row['cp'],
            "trestbps": row['trestbps'],
            "chol": row['chol'],
            "fbs": row['fbs'],
            "restecg": row['restecg'],
            "thalach": row['thalach'],
            "exang": row['exang'],
            "oldpeak": row['oldpeak'],
            "slope": row['slope'],
            "ca": row['ca'] if str(row['ca']) != '?' else 0.0,
            "thal": row['thal'] if str(row['thal']) != '?' else 3.0
        }
        
        try:
            print(f"Testing {patient_name}...")
            
            req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    prob = data.get('probability', 0)
                    label = data.get('prediction_label', '')
                    pdf = data.get('pdf_url')
                    print(f"  -> Success! Prediction: {label} ({prob:.2%}), PDF Generated: {bool(pdf)}")
                    success_count += 1
                else:
                    print(f"  -> Failed! Status: {resp.status}")
        except urllib.error.HTTPError as e:
            print(f"  -> HTTP Error: {e.code}, {e.read().decode()[:100]}")
        except Exception as e:
            print(f"  -> Request Error: {e}")
            
        time.sleep(0.5) 
        
    print(f"\nVerification Complete: {success_count}/10 passed successfully.")

if __name__ == "__main__":
    test_10_patients()
