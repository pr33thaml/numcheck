#!/usr/bin/env bash
# exit on error
set -o errexit

# Install Tesseract and its dependencies
apt-get update
apt-get install -y tesseract-ocr
apt-get install -y libtesseract-dev
apt-get install -y tesseract-ocr-eng

# Verify Tesseract installation
tesseract --version

# Install Python dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p uploads
mkdir -p excel_files 