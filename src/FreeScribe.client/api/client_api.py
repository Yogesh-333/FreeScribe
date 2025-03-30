import os
import json
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from werkzeug.utils import secure_filename
import torch
from faster_whisper import WhisperModel
import logging
import threading
import time
import sys
from api.ModelManagerApi import ModelManagerAPI

# Placeholder functions - replace with your actual implementations
def faster_whisper_transcribe(audio_file):
    """
    Transcribe audio using faster-whisper
    """
    model = WhisperModel("base", device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Transcribe the audio file
    segments, info = model.transcribe(audio_file, beam_size=5)
    
    # Combine segments into full text
    transcription = " ".join([segment.text for segment in segments])
    
    return transcription

def send_text_to_localmodel(edited_text):
    """
    Send prompt to local model and get response
    """
    if ModelManagerAPI.local_model is None:
        logging.error("Model is not loaded. Please load the model before generating responses.")
        return "Model is not loaded. Please load the model before generating responses."

    try:
        response = ModelManagerAPI.local_model.generate_response(
            edited_text,
            temperature=0.1,
            top_p=0.95,
            repeat_penalty=1.1
        )
        return response
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return f"Error generating response: {str(e)}"

class APIConfig:
    """
    Configuration class for API settings
    """
    ENABLED = True  # Default API status
    PORT = 5000    # Default port
    UPLOAD_FOLDER = 'uploads'  # Folder for uploaded audio files
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg'}

# Ensure upload folder exists
os.makedirs(APIConfig.UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = APIConfig.UPLOAD_FOLDER
api = Api(app)

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in APIConfig.ALLOWED_EXTENSIONS

class WhisperTranscriptionResource(Resource):
    def post(self):
        """
        Endpoint for audio transcription
        """
        # Check if API is enabled
        if not APIConfig.ENABLED:
            return {"message": "API is currently disabled"}, 503
        
        # Check if file is present in the request
        if 'audio' not in request.files:
            return {"message": "No audio file uploaded"}, 400
        
        audio_file = request.files['audio']
        
        # Check if filename is empty
        if audio_file.filename == '':
            return {"message": "No selected file"}, 400
        
        # Check if file is allowed
        if audio_file and allowed_file(audio_file.filename):
            # Secure the filename and save to upload folder
            filename = secure_filename(audio_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            audio_file.save(filepath)
            
            try:
                # Transcribe the audio
                transcription = faster_whisper_transcribe(filepath)
                
                # Optional: Remove the uploaded file after processing
                os.remove(filepath)
                
                return {"transcription": transcription}, 200
            
            except Exception as e:
                # Optional: Remove the file in case of error
                if os.path.exists(filepath):
                    os.remove(filepath)
                return {"message": f"Transcription error: {str(e)}"}, 500
        
        return {"message": "Invalid file type"}, 400

class LocalModelResource(Resource):
    def post(self):
        """
        Endpoint for sending text to local model
        """
        # Check if API is enabled
        if not APIConfig.ENABLED:
            return {"message": "API is currently disabled"}, 503
        
        # Get JSON data
        data = request.get_json()
        
        # Validate input
        if not data or 'text' not in data:
            return {"message": "No text provided"}, 400
        
        try:
            # Process text with local model
            result = send_text_to_localmodel(data['text'])
            return {"result": result}, 200
        
        except Exception as e:
            return {"message": f"Processing error: {str(e)}"}, 500

class APIControlResource(Resource):
    def get(self):
        """
        Get current API configuration
        """
        return {
            "api_enabled": APIConfig.ENABLED,
            "current_port": APIConfig.PORT
        }
    
    def post(self):
        """
        Update API configuration
        """
        data = request.get_json()
        
        if 'enabled' in data:
            APIConfig.ENABLED = bool(data['enabled'])
        
        if 'port' in data:
            try:
                new_port = int(data['port'])
                APIConfig.PORT = new_port
            except ValueError:
                return {"message": "Invalid port number"}, 400
        
        return {
            "message": "Configuration updated",
            "api_enabled": APIConfig.ENABLED,
            "current_port": APIConfig.PORT
        }

# Add resources to API
api.add_resource(WhisperTranscriptionResource, '/transcribe')
api.add_resource(LocalModelResource, '/process')
api.add_resource(APIControlResource, '/config')

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "api_enabled": APIConfig.ENABLED,
        "current_port": APIConfig.PORT
    }), 200

def initialize_model():
    """
    Initialize the local model when the API starts.
    """
    model_path = "././models/gemma-2-2b-it-Q8_0.gguf"
    context_size = 4096
    gpu_layers = -1 if torch.cuda.is_available() else 0
    main_gpu = 0
    n_batch = 512
    n_threads = None
    seed = 1337

    ModelManagerAPI.setup_model(
        model_path=model_path,
        context_size=context_size,
        gpu_layers=gpu_layers,
        main_gpu=main_gpu,
        n_batch=n_batch,
        n_threads=n_threads,
        seed=seed
    )

    if ModelManagerAPI.local_model is None:
        logging.error("Model failed to load.")
        raise Exception("Model failed to load.")

def run_api():
    try:
        # Reset sys.stdout to ensure a standard output stream
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        
        # Initialize the model
        initialize_model()
        
        # Run the Flask app with minimal logging
        app.run(
            host='0.0.0.0', 
            port=APIConfig.PORT, 
            debug=False, 
            use_reloader=False,
            threaded=True
        )
    except Exception as e:
        print(f"Error starting Flask server: {e}")
        sys.stdout.flush()

# Optional: Allow running directly
if __name__ == '__main__':
    run_api()