import os
from dotenv import load_dotenv
import pandas as pd
from flask import Flask, request, jsonify, render_template, session, send_file
from werkzeug.utils import secure_filename
import logging
from datetime import datetime
import json
import re
import io

# Load environment variables from .env file
load_dotenv()

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

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY')  # Unique secret key for session management
app.config['EXCEL_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['EXCEL_FOLDER'], exist_ok=True)

# Load Excel data (this will now be handled by session file paths)
excel_data = None
excel_file_path = None

ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def normalize_input(input_str):
    """Normalize input by removing spaces and dashes and converting to lowercase."""
    if not input_str:
        return None
    # Remove spaces, dashes, and convert to lowercase
    normalized = input_str.replace(' ', '').replace('-', '').lower()
    # Remove any leading/trailing whitespace
    normalized = normalized.strip()
    return normalized

def validate_number(number):
    """Validate if the extracted number is a valid alphanumeric string."""
    if not number:
        return False
    
    # Normalize the input
    normalized = normalize_input(number)
    if not normalized:
        return False
    
    # Allow alphanumeric characters
    return bool(re.match(r'^[a-zA-Z0-9]+$', normalized))

# Initialize search history
@app.before_request
def init_search_history():
    if 'search_history' not in session:
        session['search_history'] = []

def save_search_history(number, status, matches=None, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    history_entry = {
        'number': number,
        'status': status,
        'timestamp': timestamp,
        'matches': matches if matches is not None else [] # Store matches if available
    }
    
    history = session.get('search_history', [])
    history.append(history_entry)
    session['search_history'] = history[-50:]  # Keep last 50 searches

@app.route('/')
def index():
    # This ensures search_history is initialized on every request if not present
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
        # Secure filename and create a unique path for the Excel file
        filename = secure_filename(file.filename)
        original_file_path = os.path.join(app.config['EXCEL_FOLDER'], filename)
        file.save(original_file_path)
        
        # Read the Excel file with optimized settings
        df = pd.read_excel(
            original_file_path,
            dtype=str,  # Read all columns as strings
            na_filter=False,  # Don't convert empty cells to NaN
            engine='openpyxl'  # Use openpyxl engine for better performance
        )
        
        # Clean and normalize the data
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        
        # Save the processed DataFrame to a pickle file
        processed_file_name = f"{os.path.splitext(filename)[0]}.pkl"
        processed_file_path = os.path.join(app.config['EXCEL_FOLDER'], processed_file_name)
        df.to_pickle(processed_file_path)
        
        # Store the path to the processed file in the session
        session['excel_data_path'] = processed_file_path
        logger.info(f"Session excel_data_path set to: {session.get('excel_data_path')}")
        
        # Get some basic statistics
        total_rows = len(df)
        total_columns = len(df.columns)
        column_names = df.columns.tolist()
        
        logger.info(f"Excel file uploaded and processed successfully. Path: {processed_file_path}")
        
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
        logger.exception("Full traceback:")
        return jsonify({'error': f'Error processing Excel file: {str(e)}'})

@app.route('/check-number', methods=['POST'])
def check_number():
    data = request.get_json()
    number_input = data.get('number', '').strip()
    
    if not number_input:
        return jsonify({'error': 'No number provided'})
    
    # Check if excel_data_path is in session and file exists
    excel_data_path = session.get('excel_data_path')
    if not excel_data_path or not os.path.exists(excel_data_path):
        logger.error("No Excel data file found in session or on disk.")
        return jsonify({'error': 'Please upload an Excel file first'})
    
    try:
        # Load DataFrame from the pickle file
        excel_data = pd.read_pickle(excel_data_path).to_dict('records')

        logger.info(f"Processing number: {number_input}")
        logger.info(f"Excel data rows: {len(excel_data)}")
        
        found_matches = []
        
        # Generate search patterns
        search_patterns = set()
        normalized_input = normalize_input(number_input)
        if normalized_input:
            search_patterns.add(normalized_input)
            
            # Extract numeric part if present
            numeric_only = ''.join(filter(str.isdigit, normalized_input))
            if numeric_only:
                search_patterns.add(numeric_only)
            
            # Extract alphanumeric part
            alphanumeric = ''.join(filter(lambda x: x.isalnum(), normalized_input))
            if alphanumeric:
                search_patterns.add(alphanumeric)
        
        logger.info(f"Search patterns: {search_patterns}")
        
        if not search_patterns:
            save_search_history(number_input, 'not_found')
            return jsonify({'status': 'not_found', 'number': number_input})
            
        # Check each column in the Excel data
        for row_idx, row in enumerate(excel_data):
            for col_name, value in row.items():
                # Normalize the value from Excel
                normalized_value = normalize_input(str(value))
                
                if normalized_value in search_patterns:
                    # Get the row with the found number and up to 1 column before and after
                    columns = list(excel_data[0].keys())
                    col_idx = columns.index(col_name)
                    start_col = max(0, col_idx - 1)
                    end_col = min(len(columns), col_idx + 2)
                    relevant_columns = columns[start_col:end_col]
                    
                    # Create a preview with just the relevant data
                    preview_data = {
                        'headers': relevant_columns,
                        'row': {col: row[col] for col in relevant_columns},
                        'row_index': row_idx + 2  # Add 2 to match Excel's 1-based indexing
                    }
                    found_matches.append(preview_data)
        
        # Remove duplicates from found_matches
        unique_matches = []
        seen_rows = set()
        for match in found_matches:
            row_key = (match['row_index'], tuple(sorted(match['row'].items())))
            if row_key not in seen_rows:
                unique_matches.append(match)
                seen_rows.add(row_key)

        logger.info(f"Found matches: {len(unique_matches)}")

        if unique_matches:
            save_search_history(number_input, 'found', unique_matches) # Pass unique_matches
            return jsonify({
                'status': 'found',
                'number': number_input,
                'matches': unique_matches
            })
        else:
            save_search_history(number_input, 'not_found')
            return jsonify({
                'status': 'not_found',
                'number': number_input
            })
            
    except Exception as e:
        logger.error(f"Error checking number: {str(e)}")
        logger.exception("Full traceback:")
        return jsonify({'error': f'Error checking number: {str(e)}'})

@app.route('/get-history', methods=['GET'])
def get_history():
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
    return jsonify({'message': 'History cleared'})

@app.route('/export-search-history', methods=['GET'])
def export_search_history():
    try:
        history = session.get('search_history', [])
        
        if not history:
            return jsonify({'error': 'No search history to export'}), 400

        # Prepare data for DataFrame, flattening matches into separate rows
        flat_history_data = []
        for entry in history:
            if entry['status'] == 'found':
                for match in entry['matches']:
                    row_data = {
                        'Search Number': entry['number'],
                        'Status': entry['status'],
                        'Timestamp': entry['timestamp'],
                        'Row Index': match['row_index']
                    }
                    # Add the columns from the matched row
                    for header, value in match['row'].items():
                        row_data[header] = value
                    flat_history_data.append(row_data)
            else:
                flat_history_data.append({
                    'Search Number': entry['number'],
                    'Status': entry['status'],
                    'Timestamp': entry['timestamp'],
                    'Row Index': 'N/A' # No row index for not found items
                })
        
        df = pd.DataFrame(flat_history_data)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Search History')
        output.seek(0)

        file_name = f"search_history_{datetime.now().strftime('%dth_%%B_%%Y_%%I:%%M_%%p').replace(':','')}.xlsx"
        
        return send_file(output, as_attachment=True, download_name=file_name, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    except Exception as e:
        logger.error(f"Error exporting search history to Excel: {str(e)}")
        logger.exception("Full traceback:")
        return jsonify({'error': f'Error exporting search history to Excel: {str(e)}'})

# Cabinet Management Routes (for future use or if existing)
# These routes would handle adding, retrieving, and exporting cabinet data.
# Example: @app.route('/add-to-cabinet', methods=['POST'])
# def add_to_cabinet():
#     # ... logic to add item to cabinet ...
#     return jsonify({'message': 'Item added to cabinet'})

# Example: @app.route('/get-cabinet-contents', methods=['GET'])
# def get_cabinet_contents():
#     # ... logic to retrieve cabinet contents ...
#     return jsonify(session.get('cabinet_data', {}))

# Example: @app.route('/export-cabinet-to-excel', methods=['GET'])
# def export_cabinet_to_excel():
#     # ... logic to export cabinet data to Excel ...
#     return send_file(output, as_attachment=True, download_name=file_name)

@app.route('/select-folder', methods=['POST'])
def select_folder():
    data = request.get_json()
    folder_path = data.get('folder_path')

    if not folder_path:
        return jsonify({'error': 'No folder path provided'}), 400

    # Here you would typically save this path to a user's profile or configuration
    # For this example, we'll just acknowledge it.
    logger.info(f"Selected folder path: {folder_path}")
    return jsonify({'message': 'Folder path received', 'folder_path': folder_path})

@app.route('/download-excel', methods=['POST'])
def download_excel():
    try:
        data = request.get_json()
        folder_path = data.get('folder_path')
        cabinet_data = data.get('cabinetData') # Expecting client to send cabinetData

        if not folder_path:
            logger.error("No folder_path provided for download.")
            return jsonify({'error': 'Folder path not provided'}), 400

        if not cabinet_data:
            logger.error("No cabinet data found in session or request for download.")
            return jsonify({'error': 'No cabinet data to export'}), 400

        # Convert dict of lists to list of dicts for DataFrame
        flat_data = []
        for cabinet_num, entries in cabinet_data.items():
            for entry in entries:
                flat_data.append({
                    'Cabinet': cabinet_num,
                    'Number': entry.get('number'),
                    'Timestamp': datetime.fromtimestamp(entry.get('timestamp') / 1000).strftime('%Y-%m-%d %H:%M:%S') if entry.get('timestamp') else ''
                })
        
        if not flat_data:
            logger.warning("No data to export after flattening cabinet_data.")
            return jsonify({'message': 'No data to export'}), 200 # Or 400, depending on desired behavior

        df = pd.DataFrame(flat_data)

        output = io.BytesIO()
        # Use xlsxwriter engine, ensure it's installed: pip install XlsxWriter
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Cabinet Data')
        output.seek(0)

        file_name = f"cabinet_data_{datetime.now().strftime('%dth_%%B_%%Y_%%I:%%M_%%p').replace(':','')}.xlsx"
        
        # Construct the full file path for saving
        save_path = os.path.join(folder_path, file_name)
        
        # Save the file to the specified folder
        with open(save_path, 'wb') as f:
            f.write(output.getvalue())
            
        logger.info(f"Excel file saved to: {save_path}")

        return jsonify({'message': 'Excel file exported successfully', 'download_path': save_path})
    except Exception as e:
        logger.error(f"Error exporting cabinet data to Excel: {str(e)}")
        logger.exception("Full traceback:")
        return jsonify({'error': f'Error exporting cabinet data to Excel: {str(e)}'})

if __name__ == '__main__':
    # This part should be updated to consider a production-ready WSGI server
    app.run(host='0.0.0.0', port=5000, debug=True)
