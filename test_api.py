import pytest
from fastapi.testclient import TestClient
from io import BytesIO
from main import app
from unittest.mock import patch

client = TestClient(app)

def create_file(file_name, file_content, content_type):
    return {'file': (file_name, BytesIO(file_content), content_type)}

def test_extract_qr_codes_for_pdf_file():
    with open("test_files/sample_pdf.pdf", "rb") as f: 
        file_bytes = f.read()
    files = create_file("sample.pdf", file_bytes, "application/pdf")
    response = client.post("/extract_qr_codes_from_document_file", files=files)
    assert response.status_code == 200
    assert "qr_codes" in response.json()

def test_extract_qr_codes_for_png_file():
    with open("test_files/sample_png.png", "rb") as f:  
        file_bytes = f.read()
    files = create_file("sample.png", file_bytes, "image/png")
    response = client.post("/extract_qr_codes_from_document_file", files=files)
    assert response.status_code == 200
    assert "qr_codes" in response.json()

def test_extract_qr_codes_for_jpg_file():
    with open("test_files/sample_jpg.jpg", "rb") as f:  
        file_bytes = f.read()
    files = create_file("sample.jpg", file_bytes, "image/jpeg")
    response = client.post("/extract_qr_codes_from_document_file", files=files)
    assert response.status_code == 200
    assert "qr_codes" in response.json()

def test_extract_qr_codes_for_jpeg_file():
    with open("test_files/sample_jpeg.jpeg", "rb") as f:
        file_bytes = f.read()
    files = create_file("sample.jpeg", file_bytes, "image/jpeg")
    response = client.post("/extract_qr_codes_from_document_file", files=files)
    assert response.status_code == 200
    assert "qr_codes" in response.json()

def test_extract_qr_codes_for_unsupported_file_type():
    files = create_file("sample.txt", b"dummy content", "text/plain")
    response = client.post("/extract_qr_codes_from_document_file", files=files)
    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF, JPG, JPEG, and PNG files are allowed."

def test_extract_qr_codes_from_pdf_url():
    response = client.post(
        "/extract_qr_codes_from_document_url",
        params={"url": "https://jdih.kemenkeu.go.id/download/e4c54c95-5a50-4864-8326-c46ae25f7c5e/134~PMK.03~2021Per.pdf"}
    )
    assert response.status_code == 200

def test_extract_qr_codes_from_image_url():
    response = client.post(
        "/extract_qr_codes_from_document_url",
        params={"url": "https://inforuangpublik.com/wp-content/uploads/2023/10/panduan-membeli-e-meterai-dan-cara-pemakaiannya-thumbnail-315-700x400.png"}
    )
    assert response.status_code == 200

def test_extract_qr_codes_from_invalid_url():
    response = client.post(
        "/extract_qr_codes_from_document_url",
        params={"url": "https://example.com/invalid.url"}
    )
    assert response.status_code == 400

def test_extract_qr_codes_from_unsupported_file_url():
    response = client.post(
        "/extract_qr_codes_from_document_url",
        params={"url": "https://example.com/sample.txt"}
    )
    assert response.status_code == 400

def test_extract_qr_codes_internal_server_error():
    with patch("main.QRCodeExtractor.extract_qr_codes", side_effect=Exception("Simulated internal error")):
        files = create_file("sample.pdf", b"dummy content", "application/pdf")
        response = client.post("/extract_qr_codes_from_document_file", files=files)
        assert response.status_code == 500
        assert response.json()["detail"] == "Internal Server Error"


if __name__ == "__main__":
    import subprocess
    subprocess.run(["pytest", "-v"])