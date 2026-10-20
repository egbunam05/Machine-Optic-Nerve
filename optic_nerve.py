import cv2
import pytesseract
import os
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from pytesseract import Output
import numpy as np



image_path = 'sample_document.jpeg' 


print("--- INITIATING PROJECT 4: OPTIC NERVE ---")
print(f"Ingesting raw visual data from: {image_path}")


img = cv2.imread(image_path)
if img is None:
    print(f"ERROR: Could not load image at {image_path}. Please check the path.")
    exit()

print("\nExecuting Systematic Image Pre-Processing...")


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


blurred = cv2.GaussianBlur(gray, (5, 5), 0)


_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(" -> Pre-processing complete: Grayscale -> Blur -> Adaptive Threshold.")


print("\nAwakening the OCR Engine (pytesseract)...")


ocr_data = pytesseract.image_to_data(thresh, output_type=Output.DICT, config='--psm 3')

print("\nApplying the 80% Threshold: The Confidence Filter...")

n_boxes = len(ocr_data['text'])
validated_text = []
rejected_count = 0


output_img = img.copy()

for i in range(n_boxes):

    confidence = int(ocr_data['conf'][i])
    text = ocr_data['text'][i].strip()
    
   
    if len(text) == 0:
        continue

   
    if confidence >= 80:
        validated_text.append(text)
        
        
        (x, y, w, h) = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        rejected_count += 1
        
        (x, y, w, h) = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 1)



final_string = " ".join(validated_text)

print("\n--- MACHINE PERCEPTION RESULTS ---")
print(f"Total Text Blocks Detected: {n_boxes}")
print(f"Rejected (Below 80%): {rejected_count}")
print(f"Validated Strings Extracted: {len(validated_text)}")

print("\n--- EXTRACTED INTELLIGENCE ---")
print(final_string if final_string else "[No text met the 80% confidence threshold]")


cv2.imwrite('validated_output.png', output_img)
print("\nVisual Confirmation saved as 'validated_output.png'.")
