from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv

# loading environment variables from .env file
load_dotenv()

from haystack_setup import em_pipeline, rag_pipeline, splitter, ask, document_store
from haystack import Document

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part in the request."}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected."}), 400

    if file and file.filename.endswith(".txt"):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)
        with open(filepath, "r") as f:
            sql_text = f.read()
        doc = Document(content = sql_text)
        # running embedding pipeline to store documents
        print("Running em_pipeline...", flush=True)
        em_pipeline.run({"document_splitter": {"documents": [doc]}})
        return jsonify({"status": "success", "message": "Schema uploaded and processed successfully."})
    else:
        return jsonify({"status": "error", "message": "Please upload a .txt file."})

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"status": "error", "message": "Question is empty."})
    print("Running rag_pipeline...", flush = True)
    result = ask(question)
    return jsonify({"status": "success", "prompt": result["final_prompt"], "sql": result["sql"]})


@app.route("/clear", methods=["POST"])
def clear():
    doc_ids = list(document_store.storage.keys())
    if doc_ids:
        document_store.delete_documents(doc_ids)
    print('Document store refreshed succesfully.', flush=True)
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
