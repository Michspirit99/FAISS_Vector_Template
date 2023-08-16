from flask import Flask, request, jsonify
import faiss
import numpy as np

app = Flask(__name__)

# Initialize a FAISS index
dimension = 128  # Example dimension size
index = faiss.IndexFlatL2(dimension)

@app.route('/add_vector', methods=['POST'])
def add_vector():
    vector = np.array(request.json['vector']).astype('float32')
    index.add(np.array([vector]))
    return jsonify({"message": "Vector added successfully!"})

@app.route('/search', methods=['POST'])
def search():
    vector = np.array(request.json['vector']).astype('float32')
    k = 5  # Number of search results
    distances, indices = index.search(np.array([vector]), k)
    return jsonify({"distances": distances.tolist(), "indices": indices.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
