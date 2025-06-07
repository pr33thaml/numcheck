import os
import pandas as pd
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Get the absolute path to the templates directory
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))

app = Flask(__name__, 
           template_folder=template_dir,
           static_folder=static_dir)
app.secret_key = 'n8x#mP9$vL2@qR5&jK7*wY3'  # Unique secret key for session management
app.config['EXCEL_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['EXCEL_FOLDER'], exist_ok=True)

# Load Excel data
excel_data = None
excel_file_path = None

ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_number(number):
    """Validate if the extracted number is a valid 5-digit number."""
    if not number:
        return False
    
    # Remove any non-digit characters
    number = ''.join(filter(str.isdigit, number))
    
    # Check if it's a 5-digit number
    if len(number) != 5:
        return False
    
    return number

# Initialize search history
def init_search_history():
    if 'search_history' not in session:
        session['search_history'] = []

def save_search_history(number, status, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    history = session.get('search_history', [])
    history.append({
        'number': number,
        'status': status,
        'timestamp': timestamp
    })
    session['search_history'] = history[-50:]  # Keep last 50 searches

@app.route('/')
def index():
    init_search_history()
    return render_template('index.html')

@app.route('/upload-excel', methods=['POST'])
def upload_excel():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if not file.filename.endswith(('.xlsx', '.xls')):
        return jsonify({'error': 'Please upload an Excel file (.xlsx or .xls)'})
    
    try:
        # Save the file
        file_path = os.path.join(app.config['EXCEL_FOLDER'], 'numbers.xlsx')
        file.save(file_path)
        
        # Read the Excel file
        df = pd.read_excel(file_path, dtype=str, na_filter=False)
        
        # Store the data in session
        session['excel_data'] = df.to_dict('records')
        
        # Get some basic statistics
        total_rows = len(df)
        total_columns = len(df.columns)
        column_names = df.columns.tolist()
        
        return jsonify({
            'message': 'Excel file uploaded successfully',
            'stats': {
                'total_rows': total_rows,
                'total_columns': total_columns,
                'column_names': column_names
            }
        })
    except Exception as e:
        logger.error(f"Error processing Excel file: {str(e)}")
        return jsonify({'error': f'Error processing Excel file: {str(e)}'})

@app.route('/check-number', methods=['POST'])
def check_number():
    data = request.get_json()
    number = data.get('number', '').strip()
    
    if not number:
        return jsonify({'error': 'No number provided'})
    
    if not number.isdigit():
        return jsonify({'error': 'Please speak a valid number'})
    
    if 'excel_data' not in session:
        return jsonify({'error': 'Please upload an Excel file first'})
    
    try:
        excel_data = session['excel_data']
        found = False
        found_row = None
        found_column = None
        
        # Check each column in the Excel data
        for row_idx, row in enumerate(excel_data):
            for col_name, value in row.items():
                if str(value).strip() == number:
                    found = True
                    found_row = row_idx
                    found_column = col_name
                    break
            if found:
                break
        
        # Save to search history
        save_search_history(number, 'found' if found else 'not_found')
        
        if found:
            # Get the row with the found number and up to 3 columns before and after
            columns = list(excel_data[0].keys())
            col_idx = columns.index(found_column)
            start_col = max(0, col_idx - 1)
            end_col = min(len(columns), col_idx + 2)
            relevant_columns = columns[start_col:end_col]
            
            # Create a preview with just the relevant data
            preview_data = {
                'headers': relevant_columns,
                'row': {col: excel_data[found_row][col] for col in relevant_columns},
                'row_index': found_row + 2  # Add 2 to match Excel's 1-based indexing
            }
            
            return jsonify({
                'status': 'found',
                'number': number,
                'preview': preview_data
            })
        else:
            return jsonify({
                'status': 'not_found',
                'number': number
            })
            
    except Exception as e:
        logger.error(f"Error checking number: {str(e)}")
        return jsonify({'error': f'Error checking number: {str(e)}'})

@app.route('/get-history', methods=['GET'])
def get_history():
    init_search_history()
    history = session.get('search_history', [])
    
    # Calculate statistics
    total_searches = len(history)
    found_count = sum(1 for item in history if item['status'] == 'found')
    not_found_count = total_searches - found_count
    
    return jsonify({
        'history': history,
        'stats': {
            'total_searches': total_searches,
            'found_count': found_count,
            'not_found_count': not_found_count,
            'success_rate': (found_count / total_searches * 100) if total_searches > 0 else 0
        }
    })

@app.route('/clear-history', methods=['POST'])
def clear_history():
    session['search_history'] = []
    return jsonify({'message': 'Search history cleared'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 