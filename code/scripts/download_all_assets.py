import urllib.request
import os

js_dir = r"c:\Users\ISTC\Downloads\Heart Disease\static\js"
assets_dir = r"c:\Users\ISTC\Downloads\Heart Disease\static\assets"
os.makedirs(js_dir, exist_ok=True)
os.makedirs(assets_dir, exist_ok=True)

libs = {
    'three.min.js': 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js',
    'vanta.net.min.js': 'https://cdn.jsdelivr.net/npm/vanta@0.5.24/dist/vanta.net.min.js',
    'plotly.min.js': 'https://cdn.plot.ly/plotly-2.24.1.min.js',
    'lottie-player.js': 'https://unpkg.com/@lottiefiles/lottie-player@1.7.1/dist/lottie-player.js'
}

print("=== Downloading JS Libraries ===")
for name, url in libs.items():
    dest = os.path.join(js_dir, name)
    print(f"Downloading {url} to {dest}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"Successfully downloaded {name}")
    except Exception as e:
        print(f"Failed to download {name}: {e}")

# Download a public domain heartbeat Lottie JSON animation for offline use
lottie_url = "https://assets5.lottiefiles.com/packages/lf20_0vdgq0kn.json" # standard heartbeat
lottie_dest = os.path.join(assets_dir, "heartbeat.json")
print(f"\nDownloading Lottie heartbeat animation to {lottie_dest}...")
try:
    urllib.request.urlretrieve(lottie_url, lottie_dest)
    print("Successfully downloaded heartbeat.json")
except Exception as e:
    print(f"Failed to download heartbeat.json: {e}")
