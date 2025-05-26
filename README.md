# Image Cartoonizer

A Streamlit web application that transforms regular images into cartoon-style images using OpenCV.

## Features

- Upload images in JPG, JPEG, or PNG format
- Real-time image cartoonization
- Side-by-side comparison of original and cartoonized images
- Download capability for cartoonized images

## Setup

1. Make sure you have Python installed on your system
2. Clone this repository
3. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # or
   .\venv\Scripts\activate  # On Windows
   ```
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Activate the virtual environment if you haven't already
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

## Usage

1. Upload an image using the file uploader
2. The original and cartoonized versions will be displayed side by side
3. Download the cartoonized image using the download button

## Technologies Used

- Python
- Streamlit
- OpenCV
- NumPy
- Pillow # Cartoonile
