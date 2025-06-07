# Number Checker App

A web application that allows users to take photos of numbers and check if they exist in an Excel database. The app features a modern, responsive UI with real-time camera capture and instant number verification.

## Features

- Real-time camera capture
- Automatic number detection using OCR
- Excel database integration
- Beautiful, responsive UI with animations
- Instant feedback on number verification

## Prerequisites

- Python 3.7 or higher
- Tesseract OCR installed on your system
- Modern web browser with camera access

## Installation

1. Install Tesseract OCR:
   - Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

2. Clone this repository:
   ```bash
   git clone <repository-url>
   cd number-checker-app
   ```

3. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Create an Excel file named `numbers.xlsx` in the root directory with your number database.

## Usage

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to `http://localhost:5000`

3. Click "Start Camera" to begin using the app

4. Point your camera at a number and click "Capture"

5. The app will automatically process the image and check if the number exists in your Excel database

## Project Structure

```
number-checker-app/
├── app.py              # Flask backend
├── requirements.txt    # Python dependencies
├── numbers.xlsx        # Number database
├── static/
│   └── index.html     # Frontend UI
└── uploads/           # Temporary image storage
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details. 