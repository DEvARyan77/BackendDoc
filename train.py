import pandas as pd
import numpy as np
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

print("Loading clean.csv...")

def load_training_data():
    # First attempt: read normally
    try:
        df = pd.read_csv('clean.csv', encoding='utf-8')
        # Check if first column is unnamed (index)
        if df.columns[0].startswith('Unnamed') or df.columns[0] == '':
            df = df.iloc[:, 1:]
        if df.shape[1] == 2:
            df.columns = ['label', 'text']
            return df
    except Exception as e:
        print(f"Normal read failed: {e}")

    # Second attempt: specify index_col=0
    try:
        df = pd.read_csv('clean.csv', index_col=0, encoding='utf-8')
        if df.shape[1] == 2:
            df.columns = ['label', 'text']
            return df
    except Exception as e:
        print(f"Read with index_col=0 failed: {e}")

    raise RuntimeError("Could not load clean.csv in a usable format.")

df = load_training_data()

X = df['text'].fillna('')
y = df['label']

print(f"Training samples: {len(X)}")
print(f"Unique labels: {y.nunique()}")

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_vec = vectorizer.fit_transform(X)  # keep sparse

# Encode labels
label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)
classes = label_encoder.classes_.tolist()
num_classes = len(classes)

print(f"Number of classes: {num_classes}")

# Train logistic regression
model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
model.fit(X_vec, y_enc)

print("Model training complete.")

# Save vectorizer parameters – convert numpy types to Python native
vocab = {k: int(v) for k, v in vectorizer.vocabulary_.items()}
idf = [float(x) for x in vectorizer.idf_]

with open('vectorizer_params.json', 'w') as f:
    json.dump({'vocabulary': vocab, 'idf': idf}, f)
print("vectorizer_params.json saved")

# Save model coefficients and intercept – convert to lists
coef = [row.tolist() for row in model.coef_]
intercept = model.intercept_.tolist()

with open('model_coef.json', 'w') as f:
    json.dump({
        'coef': coef,
        'intercept': intercept,
        'classes': classes
    }, f)
print("model_coef.json saved")

# Save classes separately (optional)
with open('classes.json', 'w') as f:
    json.dump(classes, f)
print("classes.json saved")

print("All done.")