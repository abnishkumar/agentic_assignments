from typing import List, Dict, Any
import os
from pypdf import PdfReader
import pandas as pd
from PIL import Image
import io
import fitz 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel

class PDFChunk(BaseModel):
    text: str
    page_number: int
    chunk_type: str  # 'text', 'table', or 'image'
    metadata: Dict[str, Any]

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def extract_text(self, pdf_path: str) -> List[PDFChunk]:
        """Extract text from PDF and perform semantic chunking."""
        reader = PdfReader(pdf_path)
        chunks = []
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                # Perform semantic chunking
                text_chunks = self.text_splitter.split_text(text)
                for chunk in text_chunks:
                    chunks.append(PDFChunk(
                        text=chunk,
                        page_number=page_num + 1,
                        chunk_type='text',
                        metadata={'source': os.path.basename(pdf_path)}
                    ))
        
        return chunks

    def extract_tables(self, pdf_path: str) -> List[PDFChunk]:
        """Extract tables from PDF."""
        doc = fitz.open(pdf_path)
        chunks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            tables = page.find_tables()
            
            for table in tables:
                df = table.to_pandas()
                table_text = df.to_string()
                chunks.append(PDFChunk(
                    text=table_text,
                    page_number=page_num + 1,
                    chunk_type='table',
                    metadata={
                        'source': os.path.basename(pdf_path),
                        'table_shape': df.shape
                    }
                ))
        
        return chunks

    def extract_images(self, pdf_path: str) -> List[PDFChunk]:
        """Extract images from PDF."""
        doc = fitz.open(pdf_path)
        chunks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Convert image to text description (placeholder)
                image_text = f"Image {img_index + 1} on page {page_num + 1}"
                chunks.append(PDFChunk(
                    text=image_text,
                    page_number=page_num + 1,
                    chunk_type='image',
                    metadata={
                        'source': os.path.basename(pdf_path),
                        'image_index': img_index
                    }
                ))
        
        return chunks

    def process_pdf(self, pdf_path: str) -> List[PDFChunk]:
        """Process PDF and extract all content types."""
        all_chunks = []
        
        # Extract text
        text_chunks = self.extract_text(pdf_path)
        all_chunks.extend(text_chunks)
        
        # Extract tables
        table_chunks = self.extract_tables(pdf_path)
        all_chunks.extend(table_chunks)
        
        # Extract images
        image_chunks = self.extract_images(pdf_path)
        all_chunks.extend(image_chunks)
        
        return all_chunks

    def process_multiple_pdfs(self, pdf_dir: str) -> List[PDFChunk]:
        """Process multiple PDFs in a directory."""
        all_chunks = []
        
        for filename in os.listdir(pdf_dir):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(pdf_dir, filename)
                chunks = self.process_pdf(pdf_path)
                all_chunks.extend(chunks)
        
        return all_chunks 