from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
import numpy as np
import cv2
from .preprocessing import preprocess_image

TESSERACT_LANGUAGES = 'mal+eng'

from pdf2image import convert_from_path
import tempfile
import pytesseract
from PIL import Image
import os
import cv2
import numpy as np

POPPLER_PATH = r"C:\poppler\Library\bin"  # Change if installed elsewhere

def extract_text_from_file(uploaded_file):
    file_type = uploaded_file.name.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    text = ""
    try:
        if file_type.endswith(".pdf"):
            pages = convert_from_path(tmp_path, dpi=300 )
            for page in pages:
                processed = preprocess_image(page)
                text += pytesseract.image_to_string(processed)
        else:
            image = Image.open(tmp_path)
            processed = preprocess_image(image)
            text += pytesseract.image_to_string(processed)
    finally:
        os.remove(tmp_path)

    return text.strip()
    finally:
        os.remove(tmp_path)

    return text.strip()
