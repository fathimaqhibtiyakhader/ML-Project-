# Run:  pip install flask pandas scikit-learn
#       put heart_noisy_for_eda.csv in this folder, then: python app.py
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifier, LogisticRegression

df = pd.read_csv("heart_noisy_for_eda.csv")
df = df.drop(columns=[c for c in df.columns if str(c).lower().startswith("unnamed") or c == "index"])
df = df.drop_duplicates()
X, y = df.drop(columns=["target"]), df["target"].astype(int)
cols = X.columns.tolist()
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

def pipe(m):
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", m)])

models = {
    "Ridge (L2)": pipe(RidgeClassifier(alpha=1.0)),
    "LASSO (L1)": pipe(LogisticRegression(penalty="l1", C=1.0, solver="liblinear", max_iter=5000, random_state=42)),
    "Elastic Net": pipe(LogisticRegression(penalty="elasticnet", C=1.0, l1_ratio=0.5, solver="saga", max_iter=10000, random_state=42)),
}
for m in models.values():
    m.fit(X_train, y_train)

app = Flask(__name__)

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    d = request.get_json()
    row = pd.DataFrame([[d.get(c) if d.get(c) is not None else np.nan for c in cols]], columns=cols)
    out = []
    for name, m in models.items():
        pred = int(m.predict(row)[0])
        prob = float(m.predict_proba(row)[0][1]) if hasattr(m, "predict_proba") else None
        out.append({"model": name, "disease": pred == 1, "prob": prob})
    return jsonify(results=out)

if __name__ == "__main__":
    app.run(debug=True)
