import zipfile
import re
import xml.etree.ElementTree as ET

def get_docx_text(path):
    try:
        with zipfile.ZipFile(path) as z:
            xml_content = z.read('word/document.xml')
            root = ET.fromstring(xml_content)
            texts = []
            for child in root.iter():
                if child.tag.endswith('t'):
                    texts.append(child.text)
            return " ".join(t for t in texts if t)
    except Exception as e:
        return f"Error reading {path}: {e}"

import os
downloads_dir = r"c:\Users\ISTC\Downloads"
files = [os.path.join(downloads_dir, f) for f in os.listdir(downloads_dir) if f.lower().endswith(".docx")]
for f in files:
    print(f"=== Searching {os.path.basename(f)} ===")
    text = get_docx_text(f)
    # Find sentences or context containing 90, 90%, percent, improve, increase, tuning, parameter
    matches = re.finditer(r'([^.]{0,120}(?:90|percent|improve|increase|tune|tuning|hyperparameter)[^.]{0,120})', text, re.IGNORECASE)
    count = 0
    for match in matches:
        count += 1
        print(f"{count}: {match.group(0).strip()}")
        if count >= 40:
            print("... truncated matches ...")
            break
    print("\n")
