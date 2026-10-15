import os
from typing import List

# Import specific libraries for each modality
from pypdf import PdfReader
import docx
from PIL import Image
import pytesseract
import whisper

class DocumentLoader:
    """Standard interface for all document loaders."""
    def load(self, filepath: str) -> List[str]:
        """Load and extract text from file."""
        pass

class PDFLoader(DocumentLoader):
    def load(self, filepath: str) -> List[str]:
        """Extracts text from PDF files [cite: 517-520]."""
        text_content = []
        try:
            reader = PdfReader(filepath)
            for page in reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text_content.append(extracted_text)
        except Exception as e:
            raise RuntimeError(f"Failed to read PDF {filepath}: {e}")
        return text_content

class DOCXLoader(DocumentLoader):
    def load(self, filepath: str) -> List[str]:
        """Extracts paragraphs from Word documents [cite: 523-525]."""
        text_content = []
        try:
            doc = docx.Document(filepath)
            for para in doc.paragraphs:
                if para.text.strip():
                    text_content.append(para.text)
        except Exception as e:
            raise RuntimeError(f"Failed to read DOCX {filepath}: {e}")
        return text_content

class ImageLoader(DocumentLoader):
    def load(self, filepath: str) -> List[str]:
        """Applies Tesseract OCR to extract text from images [cite: 527-529]."""
        try:
            img = Image.open(filepath)
            # You may need to specify tesseract_cmd depending on your OS setup
            text = pytesseract.image_to_string(img)
            return [text] if text.strip() else []
        except Exception as e:
            raise RuntimeError(f"Failed to run OCR on {filepath}: {e}")

class AudioLoader(DocumentLoader):
    def __init__(self):
        # Loads the base Whisper model into memory
        print("Loading Whisper model for audio transcription...")
        self.model = whisper.load_model("base")
        
    def load(self, filepath: str) -> List[str]:
        """Uses OpenAI Whisper to transcribe audio files [cite: 531-534]."""
        try:
            result = self.model.transcribe(filepath)
            return [result["text"]]
        except Exception as e:
            raise RuntimeError(f"Failed to transcribe audio {filepath}: {e}")

# Factory function to route files easily
def get_loader(filepath: str) -> DocumentLoader:
    ext = filepath.split('.')[-1].lower()
    if ext == 'pdf':
        return PDFLoader()
    elif ext == 'docx':
        return DOCXLoader()
    elif ext in ['jpg', 'jpeg', 'png']:
        return ImageLoader()
    elif ext in ['mp3', 'wav']:
        return AudioLoader()
    else:
        raise ValueError(f"Unsupported file extension: {ext}")