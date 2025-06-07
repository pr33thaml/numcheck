from flask import Flask, request, jsonify, send_from_directory
import pytesseract
from PIL import Image
import pandas as pd
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import time
import threading
from playsound import playsound
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='static')

# Configure upload folders
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
EXCEL_FOLDER = os.getenv('EXCEL_FOLDER', 'excel_files')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'xlsx', 'xls'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['EXCEL_FOLDER'] = EXCEL_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXCEL_FOLDER, exist_ok=True)

# Global variable to store the Excel data
excel_data = None
excel_lock = threading.Lock()

# Configure Tesseract path for different environments
if os.getenv('TESSERACT_PATH'):
    pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_PATH')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )
    
    # Perform morphological operations to clean up the image
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def find_number_region(img):
    # Convert to grayscale if not already
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find the largest contour (assuming it's the number)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add some padding
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2*padding)
    h = min(img.shape[0] - y, h + 2*padding)
    
    return (x, y, w, h)

def process_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")
    
    # Find the region containing the number
    region = find_number_region(img)
    if region:
        x, y, w, h = region
        # Crop the image to the number region
        img = img[y:y+h, x:x+w]
    
    # Preprocess the image
    processed = preprocess_image(img)
    
    # Try different OCR configurations
    configs = [
        '--psm 7 -c tessedit_char_whitelist=0123456789',  # Single line
        '--psm 8 -c tessedit_char_whitelist=0123456789',  # Single word
        '--psm 13 -c tessedit_char_whitelist=0123456789'  # Raw line
    ]
    
    best_result = None
    highest_confidence = 0
    
    for config in configs:
        # Get OCR result with confidence
        data = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
        
        # Find the result with highest confidence
        for i in range(len(data['text'])):
            if float(data['conf'][i]) > highest_confidence and data['text'][i].strip():
                highest_confidence = float(data['conf'][i])
                best_result = data['text'][i].strip()
    
    if not best_result:
        # If no result found, try one more time with different preprocessing
        processed = cv2.bitwise_not(processed)  # Invert the image
        best_result = pytesseract.image_to_string(processed, config=configs[0]).strip()
    
    return best_result if best_result else ""

def check_excel(number):
    global excel_data
    
    if excel_data is None:
        return {
            'found': False,
            'message': 'Please upload an Excel file first'
        }
    
    try:
        start_time = time.time()
        with excel_lock:
            # Check if number exists in any column
            found = number in excel_data.values
        
        processing_time = time.time() - start_time
        
        if found:
            # Play sound notification
            threading.Thread(target=playsound, args=('static/notification.mp3',)).start()
            
        return {
            'found': bool(found),
            'message': f'Number found successfully! (Search time: {processing_time:.2f}s)' if found else f'Number not found. (Search time: {processing_time:.2f}s)'
        }
    except Exception as e:
        return {
            'found': False,
            'message': f'Error checking Excel: {str(e)}'
        }

def load_excel_file(filepath):
    global excel_data
    try:
        # Read Excel file with optimized settings
        df = pd.read_excel(
            filepath,
            engine='openpyxl',
            dtype=str,  # Read all columns as strings for faster comparison
            na_filter=False  # Disable NA filtering for better performance
        )
        
        with excel_lock:
            excel_data = df
            
        return True, "Excel file loaded successfully"
    except Exception as e:
        return False, f"Error loading Excel file: {str(e)}"

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/upload-excel', methods=['POST'])
def upload_excel():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['EXCEL_FOLDER'], filename)
        file.save(filepath)
        
        success, message = load_excel_file(filepath)
        return jsonify({'success': success, 'message': message})
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/check-number', methods=['POST'])
def check_number():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process the image and extract number
            number = process_image(filepath)
            
            if not number:
                return jsonify({
                    'error': 'Could not detect any number in the image. Please try again with a clearer image.'
                }), 400
            
            # Check the number in Excel
            result = check_excel(number)
            
            # Clean up the uploaded file
            os.remove(filepath)
            
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug) 