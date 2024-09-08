import os
import uuid
import tempfile
import requests
import fitz
import cv2
import numpy as np
import uvicorn
from io import BytesIO
from PIL import Image
from urllib.parse import urlparse
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO


app = FastAPI()


class QRCodeExtractor:
    def __init__(self, pdf_path, pdf_name):
        self.pdf_path = pdf_path
        self.pdf_name = pdf_name  
        self.model = YOLO("yolov8_qrcode_detector.pt")


    def generate_random_id(self):
        return str(uuid.uuid4())
    

    def pdf_to_images(self):
        pdf_document = fitz.open(self.pdf_path)
        return [
            Image.open(BytesIO(page.get_pixmap(matrix=fitz.Matrix(4.0, 4.0)).tobytes("png")))
            for page in pdf_document
        ]


    def detect_qrcode(self, image):
        results = self.model([np.array(image)])
        return [
            image.crop((x1, y1, x2, y2))
            for result in results
            for x1, y1, x2, y2 in result.boxes.xyxy.numpy()
        ]


    def read_qrcode(self, image_qrcode):
        image_qrcode = np.array(image_qrcode)
        detector = cv2.QRCodeDetector()
        data, points, _ = detector.detectAndDecode(image_qrcode)
        if data:
            return data
        else:
            return None  # Return None instead of "value not found"


    def extract_qr_codes(self):
        images = self.pdf_to_images()
        qr_codes = []
        for page_num, image in enumerate(images, start=1):
            image_qrcodes = self.detect_qrcode(image)
            for image_qrcode in image_qrcodes:
                qrcode_value = self.read_qrcode(image_qrcode)
                if qrcode_value:  # Only add if a QR code is actually detected and decoded
                    qr_codes.append({
                        "qrcode_id": self.generate_random_id(),
                        "qrcode_value": qrcode_value,
                        "page": page_num
                    })
        return {
            "file_name": self.pdf_name,
            "qr_codes": qr_codes
        }


async def save_temp_file(file_content, suffix=".pdf"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file_content)
        return tmp_file.name


async def handle_pdf_extraction(pdf_path, pdf_name):
    try:
        extractor = QRCodeExtractor(pdf_path, pdf_name)
        return extractor.extract_qr_codes()
    finally:
        os.remove(pdf_path)


@app.post("/extract_qr_codes_from_pdf_file")
async def extract_qr_codes_from_pdf_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    pdf_path = await save_temp_file(await file.read())
    return JSONResponse(content=await handle_pdf_extraction(pdf_path, file.filename))


@app.post("/extract_qr_codes_from_pdf_url")
async def extract_qr_codes_from_pdf_url(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()
        if 'pdf' not in response.headers.get('content-type', ''):
            raise HTTPException(status_code=400, detail="The URL does not point to a PDF file.")
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Could not retrieve PDF from URL: {str(e)}")
    
    pdf_path = await save_temp_file(response.content)
    pdf_name = os.path.basename(urlparse(url).path)
    return JSONResponse(content=await handle_pdf_extraction(pdf_path, pdf_name))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")
