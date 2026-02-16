from flask import Flask, render_template, request, jsonify
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import whisper
from pydub import AudioSegment

app = Flask(__name__)

# ----------------------
# Paths
# ----------------------
TEXT_MODEL_PATH = './emotion_detection_model'

# ----------------------
# Load Models
# ----------------------
print("Loading emotion detection model...")
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)
text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_PATH)
text_model.eval()
print("Emotion model loaded successfully!")

print("Loading Whisper model (first time will download ~74MB)...")
whisper_model = whisper.load_model("base")
print("Whisper model loaded successfully!")

TEXT_EMOTIONS = {
    0: {'name': 'anger', 'emoji': '😠', 'color': '#ff4444'},
    1: {'name': 'fear', 'emoji': '😨', 'color': '#9c27b0'},
    2: {'name': 'joy', 'emoji': '😊', 'color': '#4caf50'},
    3: {'name': 'sadness', 'emoji': '😢', 'color': '#2196f3'},
    4: {'name': 'surprise', 'emoji': '😮', 'color': '#ff9800'}
}

# ----------------------
# Routes
# ----------------------
@app.route('/')
def index():
    return render_template('index.html')

# ----------------------
# Text Emotion Analysis
# ----------------------
@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    data = request.json
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = text_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        predicted_idx = int(torch.argmax(probs))
    
    all_emotions = []
    for i, p in enumerate(probs):
        all_emotions.append({
            'emotion': TEXT_EMOTIONS[i]['name'],
            'emoji': TEXT_EMOTIONS[i]['emoji'],
            'color': TEXT_EMOTIONS[i]['color'],
            'confidence': round(float(p)*100, 2)
        })
    all_emotions.sort(key=lambda x: x['confidence'], reverse=True)

    return jsonify({
        'type': 'text',
        'emotion': TEXT_EMOTIONS[predicted_idx]['name'],
        'emoji': TEXT_EMOTIONS[predicted_idx]['emoji'],
        'confidence': all_emotions[0]['confidence'],
        'all_emotions': all_emotions,
        'input_text': text
    })

# ----------------------
# Speech Emotion Analysis with Whisper
# ----------------------
@app.route('/analyze_speech', methods=['POST'])
def analyze_speech():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Get selected language from the form
    selected_language = request.form.get('language', 'auto')
    
    tmp_input = None
    tmp_output = None
    
    try:
        # Save uploaded audio
        file_ext = audio_file.filename.split('.')[-1] if '.' in audio_file.filename else 'webm'
        tmp_input = f'temp_audio_input.{file_ext}'
        tmp_output = 'temp_audio_output.wav'
        
        audio_file.save(tmp_input)

        # Convert to WAV format
        try:
            audio = AudioSegment.from_file(tmp_input)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(tmp_output, format='wav')
        except Exception as conv_error:
            if tmp_input and os.path.exists(tmp_input):
                os.remove(tmp_input)
            return jsonify({'error': f'Audio conversion failed: {str(conv_error)}'}), 500

        # Transcribe and translate using Whisper
        try:
            # If user selected a specific language, use it. Otherwise auto-detect
            whisper_language = None if selected_language == 'auto' else selected_language
            
            # task="translate" - Automatically translates to English
            # Using selected language improves accuracy significantly!
            result = whisper_model.transcribe(
                tmp_output,
                task="translate",          # Translate any language to English
                language=whisper_language, # Use selected language (None = auto-detect)
                fp16=False,                # Set True if you have NVIDIA GPU
                temperature=0.0,           # More deterministic
                best_of=5,                 # Try 5 times and pick best result
                beam_size=5                # Better accuracy with beam search
            )
            
            english_text = result["text"].strip()
            detected_lang = result.get("language", "unknown")
            
            # Map language codes to names
            language_names = {
                'hi': 'Hindi', 'en': 'English', 'es': 'Spanish', 'fr': 'French',
                'de': 'German', 'te': 'Telugu', 'ta': 'Tamil', 'bn': 'Bengali',
                'mr': 'Marathi', 'ur': 'Urdu', 'pa': 'Punjabi', 'gu': 'Gujarati',
                'kn': 'Kannada', 'ml': 'Malayalam', 'zh': 'Chinese', 'ja': 'Japanese',
                'ko': 'Korean', 'ar': 'Arabic', 'ru': 'Russian', 'pt': 'Portuguese'
            }
            detected_language_name = language_names.get(detected_lang, detected_lang.upper())
            
            if not english_text:
                return jsonify({'error': 'No speech detected in audio'}), 400
                
        except Exception as whisper_error:
            return jsonify({'error': f'Whisper transcription failed: {str(whisper_error)}'}), 500
        finally:
            # Clean up temporary files
            if tmp_input and os.path.exists(tmp_input):
                os.remove(tmp_input)
            if tmp_output and os.path.exists(tmp_output):
                os.remove(tmp_output)

        # Predict emotion using the English text
        inputs = text_tokenizer(english_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = text_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            predicted_idx = int(torch.argmax(probs))

        all_emotions = []
        for i, p in enumerate(probs):
            all_emotions.append({
                'emotion': TEXT_EMOTIONS[i]['name'],
                'emoji': TEXT_EMOTIONS[i]['emoji'],
                'color': TEXT_EMOTIONS[i]['color'],
                'confidence': round(float(p)*100, 2)
            })
        all_emotions.sort(key=lambda x: x['confidence'], reverse=True)

        return jsonify({
            'type': 'speech',
            'transcription': english_text,  # Always in English
            'detected_language': detected_language_name,
            'emotion': TEXT_EMOTIONS[predicted_idx]['name'],
            'emoji': TEXT_EMOTIONS[predicted_idx]['emoji'],
            'confidence': all_emotions[0]['confidence'],
            'all_emotions': all_emotions
        })

    except Exception as e:
        # Clean up on any error
        if tmp_input and os.path.exists(tmp_input):
            os.remove(tmp_input)
        if tmp_output and os.path.exists(tmp_output):
            os.remove(tmp_output)
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

# ----------------------
# Run the app
# ----------------------
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🎭 Emotion Detection App")
    print("🌍 Whisper: Any Language → English Translation")
    print("✅ 99+ Languages Supported")
    print("✅ Works Offline")
    print("="*50 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000)