import easyocr
import numpy as np
import cv2

# Initialize OCR reader (downloads model on first run)
# Add more languages if needed e.g. ['en', 'hi'] for Hindi too
reader = easyocr.Reader(['en', 'hi'], gpu=True)

def read_text(image_bytes):
    # Convert bytes to numpy array
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Run OCR
    results = reader.readtext(frame)

    extracted_texts = []
    for (bbox, text, confidence) in results:
        if confidence > 0.4:  # Only include text with >40% confidence
            extracted_texts.append({
                'text': text.strip(),
                'confidence': round(confidence * 100, 1)
            })

    return extracted_texts