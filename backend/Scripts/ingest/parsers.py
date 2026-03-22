import os
import json
import fitz
from docling.document_converter import DocumentConverter

SUPPORTED_EXTENSIONS = [".txt", ".md", ".pdf", ".docx", ".json"]

doc_converter = DocumentConverter()

def load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_pdf(path):
    doc = fitz.open(path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text

def load_docx(path):
    result = doc_converter.convert(path)
    return result.document.export_to_markdown()

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return json.dumps(data, indent=2)

def load_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md"):
        return load_txt(path)
    if ext == ".pdf":
        return load_pdf(path)
    if ext == ".docx":
        return load_docx(path)
    if ext == ".json":
        return load_json(path)
    return None
