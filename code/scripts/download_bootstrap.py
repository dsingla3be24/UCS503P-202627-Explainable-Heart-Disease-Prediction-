import urllib.request
import os

css_dir = r"c:\Users\ISTC\Downloads\Heart Disease\static\css"
js_dir = r"c:\Users\ISTC\Downloads\Heart Disease\static\js"
os.makedirs(css_dir, exist_ok=True)
os.makedirs(js_dir, exist_ok=True)

urls = {
    'bootstrap.min.css': 'https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css',
    'bootstrap.bundle.min.js': 'https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js'
}

print("=== Downloading Bootstrap ===")
for name, url in urls.items():
    dest_dir = css_dir if name.endswith('.css') else js_dir
    dest = os.path.join(dest_dir, name)
    print(f"Downloading {url} to {dest}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"Successfully downloaded {name}")
    except Exception as e:
        print(f"Failed to download {name}: {e}")
