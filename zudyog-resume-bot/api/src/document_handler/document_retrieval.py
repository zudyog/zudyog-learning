import os
from typing import List
import PyPDF2
from tenacity import retry, stop_after_attempt, wait_exponential
from src.core.config import get_openai_client, get_pinecone_index
import spacy
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from pdf2image import convert_from_path
import numpy as np
import easyocr
import pypdfium2
from src.document_handler.exceptions import EmbeddingGenerationError, MetadataValidationError, PDFProcessingError, PineconeUpsertError
from src.models.metadata import Metadata


class DocumentRetrieval:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(['en'])

    def index_texts(self, file_path: str, metadata: Metadata) -> int:
        try:
            self.validate_pdf(file_path)
            self.validate_metadata(metadata)
            paragraphs = self.extract_text_with_easyocr(file_path)
            if not paragraphs:
                return 0
            for paragraph in paragraphs:
                entities = self.perform_ner(paragraph)
                print(f"Entities in paragraph: {entities}")
            embeddings = self.generate_embeddings(paragraphs)
            self.upsert_to_pinecone(paragraphs, embeddings, metadata)
            return len(paragraphs)
        except (PDFProcessingError, MetadataValidationError,
                EmbeddingGenerationError, PineconeUpsertError) as e:
            raise e
        except Exception as e:
            raise Exception(f"Unexpected error during indexing: {str(e)}")

    def query_index(self, query, top_k=5):
        query_embedding = self.generate_embeddings([query])[0]
        pinecone_index = get_pinecone_index()
        results = pinecone_index.query(
            vector=query_embedding, top_k=top_k, include_metadata=True)
        return results["matches"]

    def validate_pdf(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            raise PDFProcessingError(f"File not found: {file_path}")

        if not file_path.lower().endswith('.pdf'):
            raise PDFProcessingError("File must be a PDF")

        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                if len(reader.pages) == 0:
                    return 0
        except PyPDF2.PdfReadError as e:
            raise PDFProcessingError(f"Invalid PDF file: {str(e)}")

    def preprocess_image(self, image: Image) -> Image:
        """Preprocess the image to improve OCR accuracy."""
        # Convert to grayscale
        image = image.convert('L')
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2)
        # Apply a slight blur to reduce noise
        image = image.filter(ImageFilter.GaussianBlur(radius=1))
        return image

    def extract_text(self, file_path: str) -> List[str]:
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                paragraphs = []

                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        page_paragraphs = [p.strip()
                                           for p in text.split('\n\n') if p.strip()]
                        paragraphs.extend(page_paragraphs)

                return paragraphs
        except Exception as e:
            raise PDFProcessingError(
                f"Error extracting text from PDF: {str(e)}")

    # def extract_text_with_ocr(self, file_path: str) -> List[str]:
    #     try:
    #         paragraphs = []
    #         # Convert PDF to images with high DPI
    #         images = convert_from_path(file_path, dpi=300)

    #         with concurrent.futures.ThreadPoolExecutor() as executor:
    #             futures = [executor.submit(self.process_image_with_ocr, image) for image in images]
    #             for future in concurrent.futures.as_completed(futures):
    #                 page_paragraphs = future.result()
    #                 paragraphs.extend(page_paragraphs)

    #         return paragraphs
    #     except Exception as e:
    #         raise PDFProcessingError(f"Error extracting text from PDF: {str(e)}")

    def extract_text_with_easyocr(self, file_path: str) -> List[str]:
        """Extract text from PDF using EasyOCR, replacing Poppler with pdfium + Pillow."""
        try:
            paragraphs = []
            pdf_document = pypdfium2.PdfDocument(file_path)  # Load PDF

            for page_number in range(len(pdf_document)):
                page = pdf_document[page_number]  # Get page
                # Convert to NumPy array
                bitmap = page.render(scale=3).to_numpy()

                # Convert to Pillow image (PIL) for further processing
                image = Image.fromarray(bitmap)

                # Convert to NumPy array for EasyOCR
                image_np = np.array(image)
                results = self.reader.readtext(image_np)
                page_text = " ".join([result[1] for result in results])

                if page_text:
                    page_paragraphs = [p.strip()
                                       for p in page_text.split('\n\n') if p.strip()]
                    paragraphs.extend(page_paragraphs)

            return paragraphs
        except Exception as e:
            raise Exception(
                f"Error extracting text from PDF using EasyOCR: {str(e)}")

    def process_image_with_ocr(self, image: Image) -> List[str]:
        """Process a single image with OCR."""
        try:
            # Preprocess the image
            image = self.preprocess_image(image)
            # Perform OCR
            text = pytesseract.image_to_string(image, lang=self.ocr_language)
            if text:
                return [p.strip() for p in text.split('\n\n') if p.strip()]
            return []
        except Exception as e:
            print(f"OCR processing error: {str(e)}")
            return []

    def convert_pdf_page_to_image(self, file_path: str, page_num: int) -> Image:
        images = convert_from_path(
            file_path, first_page=page_num+1, last_page=page_num+1, dpi=300)
        return images[0]

    def perform_ner(self, text: str) -> List[str]:
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = []
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                openai_client = get_openai_client()
                response = openai_client.embeddings.create(
                    input=batch,
                    model="text-embedding-ada-002"
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            return embeddings
        except Exception as e:
            raise EmbeddingGenerationError(
                f"Error generating embeddings: {str(e)}")

    def validate_metadata(self, metadata: Metadata) -> None:
        if not metadata['document_id']:
            raise MetadataValidationError("document_id is required")

        if not metadata['date_uploaded']:
            raise MetadataValidationError("date_uploaded is required")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def upsert_to_pinecone(self, texts: List[str], embeddings: List[List[float]], metadata: Metadata) -> None:
        try:
            vectors = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                vector_metadata = {
                    **metadata,
                    "text": text,
                    "paragraph_id": i
                }
                vectors.append({
                    "id": f"{metadata['document_id']}_p{i}",
                    "values": embedding,
                    "metadata": vector_metadata
                })
            pinecone_index = get_pinecone_index()
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                pinecone_index.upsert(vectors=batch)
        except Exception as e:
            raise PineconeUpsertError(f"Error upserting to Pinecone: {str(e)}")

    # def index_texts(self, file_path: str, metadata: Metadata) -> int:
    #     try:
    #         self.validate_pdf(file_path)
    #         self.validate_metadata(metadata)
    #         # Use EasyOCR for text extraction
    #         paragraphs = self.extract_text_with_easyocr(file_path)
    #         if not paragraphs:
    #             return 0
    #         for paragraph in paragraphs:
    #             entities = self.perform_ner(paragraph)
    #             print(f"Entities in paragraph: {entities}")
    #         embeddings = self.generate_embeddings(paragraphs)
    #         self.upsert_to_pinecone(paragraphs, embeddings, metadata)
    #         return len(paragraphs)
    #     except (PDFProcessingError, MetadataValidationError,
    #             EmbeddingGenerationError, PineconeUpsertError) as e:
    #         raise e
    #     except Exception as e:
    #         raise Exception(f"Unexpected error during indexing: {str(e)}")
