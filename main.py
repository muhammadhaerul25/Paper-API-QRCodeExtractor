import os
import uuid
import tempfile
import requests
import fitz
import cv2
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
from urllib.parse import urlparse
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from qreader import QReader


app = FastAPI()


##--------------------------------------QRCodeExtractor----------------------------------------##
class QRCodeExtractor:
    def __init__(self, file_path, file_name):
        self.file_path = file_path
        self.file_name = file_name
        self.model = YOLO("yolov8_qrcode_detector.pt")

    def generate_random_id(self):
        return str(uuid.uuid4())

    def file_to_images(self):
        if self.file_name.lower().endswith('.pdf'):
            pdf_document = fitz.open(self.file_path)
            return [
                Image.open(BytesIO(page.get_pixmap(matrix=fitz.Matrix(4.0, 4.0)).tobytes("png"))).convert('RGB')
                for page in pdf_document
            ]
        else:
            return [Image.open(self.file_path).convert('RGB')]

    def detect_qrcode_yolov8(self, image):
        scale = 1.25
        image = image.convert('RGB')
        results = self.model([np.array(image)])
        img_width, img_height = image.size

        return [
            image.crop((
                int(max((x1 + x2) / 2 - (x2 - x1) * scale / 2, 0)),
                int(max((y1 + y2) / 2 - (y2 - y1) * scale / 2, 0)),
                int(min((x1 + x2) / 2 + (x2 - x1) * scale / 2, img_width)),
                int(min((y1 + y2) / 2 + (y2 - y1) * scale / 2, img_height))
            ))
            for result in results
            for x1, y1, x2, y2 in result.boxes.xyxy.numpy()
        ]

    def read_qrcode_qreader(self, image_qrcode):
        qreader = QReader()
        image_qrcode = np.array(image_qrcode)
        decoded_text = qreader.detect_and_decode(image=image_qrcode)
        if decoded_text:
            link = decoded_text[0]
            return link
        else:
            return None


    def extract_qr_codes(self):
        images = self.file_to_images()
        qr_codes = []
        for page_num, image in enumerate(images, start=1):
            image_qrcodes = self.detect_qrcode_yolov8(image)
            for image_qrcode in image_qrcodes:
                qrcode_value = self.read_qrcode_qreader(image_qrcode)
                if qrcode_value:
                    qr_codes.append({
                        "qrcode_id": self.generate_random_id(),
                        "qrcode_value": qrcode_value,
                        "page": page_num
                    })
        return {
            "file_name": self.file_name,
            "qr_codes": qr_codes
        }


##--------------------------------------Helper----------------------------------------##
async def save_temp_file(file_content, suffix):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file_content)
        return tmp_file.name


async def handle_extraction(file_path, file_name):
    try:
        extractor = QRCodeExtractor(file_path, file_name)
        return extractor.extract_qr_codes()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        os.remove(file_path)



##--------------------------------------Endpoints----------------------------------------##
@app.post("/extract_qr_codes_from_document_file")
async def extract_qr_codes_from_document_file(file: UploadFile = File(...)):
    if not (file.filename.lower().endswith('.pdf') or 
            file.filename.lower().endswith('.jpg') or 
            file.filename.lower().endswith('.jpeg') or 
            file.filename.lower().endswith('.png')):
        raise HTTPException(status_code=400, detail="Only PDF, JPG, JPEG, and PNG files are allowed.")
    
    suffix = os.path.splitext(file.filename)[1]
    file_path = await save_temp_file(await file.read(), suffix)
    return JSONResponse(content=await handle_extraction(file_path, file.filename))


@app.post("/extract_qr_codes_from_document_url")
async def extract_qr_codes_from_document_url(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '')
        
        if 'pdf' in content_type:
            suffix = ".pdf"
        elif 'image/jpeg' in content_type:
            suffix = ".jpeg"
        elif 'image/png' in content_type:
            suffix = ".png"
        else:
            raise HTTPException(status_code=400, detail="The URL does not point to a supported file format (PDF, JPG, JPEG, PNG).")
        
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Could not retrieve file from URL: {str(e)}")
    
    file_path = await save_temp_file(response.content, suffix)
    file_name = url
    return JSONResponse(content=await handle_extraction(file_path, file_name))



##--------------------------------------Main----------------------------------------##
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)