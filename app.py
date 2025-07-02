from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)
CORS(app)

df = pd.read_csv("dialogs.txt",
                 sep=":::",
                 header=None,
                 names=["prompt", "response"])

# Vectorization and model training
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['prompt'])
y = df['response']
model = MultinomialNB()
model.fit(X, y)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_input = data["message"]

    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)[0]
    return jsonify({"response": prediction})


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=3000)
