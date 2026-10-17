import urllib.request
import os

js_dir = r"c:\Users\ISTC\Downloads\Heart Disease\static\js"
os.makedirs(js_dir, exist_ok=True)

url = "https://cdn.plot.ly/plotly-2.24.1.min.js"
dest = os.path.join(js_dir, "plotly.min.js")

print(f"Downloading {url} to {dest}...")
try:
    urllib.request.urlretrieve(url, dest)
    print("Successfully downloaded plotly.min.js")
except Exception as e:
    print(f"Failed to download plotly.min.js: {e}")
