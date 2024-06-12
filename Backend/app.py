from flask import Flask, render_template, jsonify, request
from flask_restful import reqparse, Api, Resource
import pickle
from model import SGDModel
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)  
api = Api(app)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize model
model = SGDModel()

# Load model and vectorizer
sgd_path = 'Backend/lib/models/SentimentClassifier.pkl'
vec_path = 'Backend/lib/models/TFIDFVectorizer.pkl'

try:
    with open(sgd_path, 'rb') as f:
        model.sgd = pickle.load(f)
    logging.info(f"Model loaded from {sgd_path}")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise

try:
    with open(vec_path, 'rb') as f:
        model.vectorizer = pickle.load(f)
    logging.info(f"Vectorizer loaded from {vec_path}")
except Exception as e:
    logging.error(f"Failed to load vectorizer: {e}")
    raise

# Argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query', type=str, location='args', required=True, help="Query string cannot be blank!")

class PredictSentiment(Resource):
    def get(self):
        args = parser.parse_args()
        user_query = args.get("query")

        if not user_query:
            return {'error': 'No query provided'}, 400

        try:
            uq_vectorized = model.vectorizer.transform([user_query])
            prediction = model.sgd.predict(uq_vectorized)
            pred_proba = model.sgd.predict_proba(uq_vectorized)

            confidence = round(pred_proba[0].max(), 3)
            output = {
                "prediction": prediction[0],
                "confidence": confidence
            }
            return jsonify(output)

        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return {'error': str(e)}, 500

# API
api.add_resource(PredictSentiment, '/predict')

@app.route('/')
def hello():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)