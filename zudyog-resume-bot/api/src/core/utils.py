from reportlab.pdfgen import canvas
from PIL import Image
from reportlab.lib.pagesizes import letter
from fastapi import UploadFile


def sanitize_results(results):
    return [
        {
            "id": result["id"],
            "metadata": result["metadata"],
            "score": result["score"]
        }
        for result in results
    ]


def convert_to_pdf(file: UploadFile, pdf_path: str):
    """Convert non-PDF files to PDF format."""
    print(f"Converting {file.filename} to PDF")

    # Read the file content into memory
    file_content = file.file.read()

    if file.filename.lower().endswith(('.jpeg', '.jpg', '.png')):
        print("Converting image to PDF")
        # Use BytesIO to handle the file content in memory
        image = Image.open(io.BytesIO(file_content))
        image.save(pdf_path, "PDF", resolution=100.0)

    elif file.content_type.startswith('image/'):
        # Reset file pointer since we read it above
        image = Image.open(io.BytesIO(file_content))
        image.save(pdf_path, "PDF", resolution=100.0)

    elif file.content_type == 'text/plain':
        c = canvas.Canvas(pdf_path, pagesize=letter)
        # Decode the file content we read earlier
        text = file_content.decode('utf-8')
        c.drawString(100, 750, text)
        c.save()

    else:
        raise ValueError("Unsupported file type for conversion to PDF")

    # Reset the file pointer for any subsequent operations
    file.file.seek(0)
