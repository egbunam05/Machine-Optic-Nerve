```markdown
# Project 4: Building the Machine's Optic Nerve

An Optical Character Recognition (OCR) pipeline built for the DecodeLabs AI program.

## 🧠 Core Concepts
* **Pre-Processing:** Systematic Grayscale conversion, Gaussian Blur, and Adaptive Thresholding to isolate text from noise.
* **Confidence Filtering:** Only displays text strings where the AI's confidence score is strictly **80% or higher**.

## 🛠️ Tech Stack
* Python, OpenCV, pytesseract, Pandas

## 🚀 How to Run
1. Install the Tesseract Engine on your system.
2. Install requirements: `pip install opencv-python pytesseract`
3. Add an image named `sample_document.jpeg` to the folder.
4. Run the script: `python optic_nerve.py`
```