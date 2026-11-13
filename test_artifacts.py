
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "Artifacts"

print("Project path:", BASE_DIR)
print("Artifacts path:", ARTIFACTS_DIR)

print("\nChecking files...")

for filename in ["nlp_model.pkl", "tranform.pkl", "main_data.csv"]:
    file_path = ARTIFACTS_DIR / filename
    print(filename, "Exists:", file_path.exists())

print("\nTesting model loading...")

try:
    with open(ARTIFACTS_DIR / "nlp_model.pkl", "rb") as f:
        clf = pickle.load(f)
    print("Model loaded successfully!")

except Exception as e:
    print("MODEL ERROR:", repr(e))

print("\nTesting vectorizer loading...")

try:
    with open(ARTIFACTS_DIR / "tranform.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print("Vectorizer loaded successfully!")

except Exception as e:
    print("VECTORIZER ERROR:", repr(e))
