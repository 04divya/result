import numpy as np
import cv2
from PIL import Image

from pdf2image import convert_from_path
import tempfile
import pytesseract
from PIL import Image
import os
import cv2
import numpy as np

def preprocess_image(image):
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    denoised = cv2.medianBlur(binary, 3)
    return Image.fromarray(denoised)
