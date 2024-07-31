from flask import Flask, request, jsonify
from flask_cors import CORS
from data import predict_chord

app = Flask(__name__)
CORS(app, resources={r"/predict_chord": {"origins": "http://localhost:5173"}})

@app.route('/predict_chord', methods=['POST'])
def predict_chord_endpoint():
    midi_keys = request.json['midi_keys']
    chord_name = predict_chord(midi_keys)
    return jsonify({'chord_name': chord_name})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)