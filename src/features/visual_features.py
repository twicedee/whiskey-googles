import cv2
import numpy as np

def detect_label_region(img):
    """Find rectangular whisky label using contour detection"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
    
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        
        if len(approx) == 4:  # Rectangular label
            x,y,w,h = cv2.boundingRect(approx)
            return img[y:y+h, x:x+w]
    
    return None

def extract_color_histogram(img, bins=(12,12,12)):
    """Enhanced color histogram with LAB color space"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [0,1,2], None, bins, [0,256,0,256,0,256])
    return cv2.normalize(hist, hist).flatten()

def preprocess_image(img):
    """Standard preprocessing with label detection fallback"""
    img = cv2.resize(img, (224, 224))  # Increased from 224x224
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # LAB better for color consistency