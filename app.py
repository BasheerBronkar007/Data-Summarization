import os
import re
import time
import heapq
import PyPDF2
import pdfplumber
import pytesseract
from PIL import Image
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from pdfminer.high_level import extract_text
from transformers import pipeline, AutoTokenizer
import torch
from urllib.parse import unquote
from rouge_score import rouge_scorer

app = Flask(__name__)

# Configuration
MAX_FILE_SIZE = 30 * 1024 * 1024  # 30MB
MAX_TEXT_LENGTH = 100000  # 100k characters
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Model Configuration
FAST_MODEL = "sshleifer/distilbart-cnn-6-6"  # Fast model
QUALITY_MODEL = "facebook/bart-large-cnn"     # High-quality model
TOKENIZER = AutoTokenizer.from_pretrained(FAST_MODEL)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize ROUGE scorer
rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Load models with improved error handling and memory management
def initialize_models():
    global fast_summarizer, quality_summarizer
    
    try:
        # Fast model (always loaded)
        fast_summarizer = pipeline(
            "summarization",
            model=FAST_MODEL,
            tokenizer=TOKENIZER,
            device=DEVICE,
            torch_dtype=torch.float16 if DEVICE == "cuda" else None,
            truncation=True
        )
        print(f"Fast model loaded successfully on {DEVICE}")
        
        # Quality model (load on demand)
        quality_summarizer = None
        
    except Exception as e:
        print(f"Model loading error: {e}")
        fast_summarizer = None
        quality_summarizer = None

def load_quality_model():
    global quality_summarizer
    if quality_summarizer is None:
        try:
            quality_summarizer = pipeline(
                "summarization",
                model=QUALITY_MODEL,
                tokenizer=QUALITY_MODEL,
                device=DEVICE,
                torch_dtype=torch.float16 if DEVICE == "cuda" else None,
                truncation=True
            )
            print("Quality model loaded on demand")
        except Exception as e:
            print(f"Failed to load quality model: {e}")
            return None
    return quality_summarizer

initialize_models()

# Text processing utilities
def clean_input_text(text):
    """Enhanced text cleaning with STEM paper support"""
    text = unquote(text)
    # Preserve important markers while removing citations
    text = re.sub(r'(\[\d+\]|\(\d+\))', '', text)  # Remove [1] or (1)
    # Handle equations and special characters
    text = re.sub(r'\s+', ' ', text)
    # Preserve section headers
    text = re.sub(r'(\\n\s*[A-Z][A-Za-z\s]+:)', r'\n\1', text)
    return text.strip()

def extractive_summary(text, reduction=0.3):
    """Improved extractive summarization for long documents"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) < 10:
        return text
    
    # Score sentences by length and position
    scored = []
    for i, sent in enumerate(sentences):
        score = len(sent.split()) * 0.6  # Weight by length
        score += (1 - i/len(sentences)) * 0.4  # Weight by position
        scored.append((score, sent))
    
    # Select top sentences
    scored.sort(reverse=True)
    keep = int(len(sentences) * (1 - reduction))
    selected = [s[1] for s in scored[:keep]]
    return " ".join(sorted(selected, key=lambda x: sentences.index(x)))

def chunk_text(text, max_tokens=1024):
    """Improved chunking that preserves paragraphs"""
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para_length = len(TOKENIZER.tokenize(para))
        if current_length + para_length > max_tokens and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(para)
        current_length += para_length
    
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
    return chunks

def needs_quality_summary(text):
    """Enhanced quality detection for research papers"""
    word_count = len(text.split())
    technical_terms = [
        'methodology', 'experiment', 'results', 'protein', 'algorithm',
        'hypothesis', 'analysis', 'conclusion', 'data', 'research'
    ]
    equations = re.findall(r'(\$[^$]+\$|\\\(.+?\\\)|\\\[.+?\\\])', text)
    
    return (word_count > 800 or 
            any(term in text.lower() for term in technical_terms) or
            len(equations) > 2)

# PDF Processing with multiple fallbacks
def extract_text_from_pdf(pdf_path):
    """Robust PDF extraction with OCR fallback"""
    try:
        # Try pdfminer first
        text = extract_text(pdf_path)
        if len(text) > 3000:
            return text
        
        # Try pdfplumber for better table handling
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
            if len(text) > 3000:
                return text
        
        # Try PyPDF2 as final fallback
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = " ".join([page.extract_text() or "" for page in reader.pages])
            if len(text) > 3000:
                return text
        
        # OCR fallback if all else fails
        if len(text) < 3000:
            images = convert_pdf_to_images(pdf_path)
            ocr_text = "\n".join([pytesseract.image_to_string(img) for img in images])
            if len(ocr_text) > 1000:  # Minimum viable text
                return ocr_text
        
        return text if len(text) > 1000 else "Extracted text too short or invalid"
    except Exception as e:
        return f"PDF Error: {str(e)}"

def convert_pdf_to_images(pdf_path):
    """Convert PDF pages to images for OCR"""
    try:
        images = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                im = page.to_image(resolution=300)
                img_bytes = BytesIO()
                im.save(img_bytes, format='PNG')
                images.append(Image.open(img_bytes))
        return images
    except Exception:
        return []

# Core summarization function
def summarize_with_metrics(text, max_length=300, high_quality=False):
    """Optimized two-stage summarization with enhanced metrics"""
    if not fast_summarizer:
        return "Model not ready. Please try again later.", None
    
    try:
        start_time = time.time()
        text = clean_input_text(text)
        
        # Extractive pre-summarization for long texts
        if len(text.split()) > 2000:
            text = extractive_summary(text, reduction=0.4)
        
        # Stage 1: Fast summary with DistilBART
        chunks = chunk_text(text)
        summaries = []
        
        for chunk in chunks:
            output = fast_summarizer(
                chunk,
                max_length=min(200, max_length//max(1, len(chunks))),
                min_length=50,
                do_sample=False,
                truncation=True,
                no_repeat_ngram_size=3
            )
            summaries.append(output[0]['summary_text'])
        
        combined = " ".join(summaries)
        
        # Stage 2: Quality refinement if needed
        model_used = "fast"
        if high_quality or needs_quality_summary(text):
            quality_model = load_quality_model()
            if quality_model:
                try:
                    combined = quality_model(
                        combined,
                        max_length=max_length,
                        min_length=max_length//2,
                        do_sample=False,
                        repetition_penalty=2.5,
                        no_repeat_ngram_size=3
                    )[0]['summary_text']
                    model_used = "quality"
                except torch.cuda.OutOfMemoryError:
                    print("GPU OOM, falling back to fast model")
        
        # Calculate enhanced metrics
        processing_time = time.time() - start_time
        rouge_scores = rouge.score(text, combined)
        
        metrics = {
            "processing_time": round(processing_time, 2),
            "rouge1": round(rouge_scores['rouge1'].fmeasure, 3),
            "rouge2": round(rouge_scores['rouge2'].fmeasure, 3),
            "rougeL": round(rouge_scores['rougeL'].fmeasure, 3),
            "compression_ratio": round(len(text)/len(combined), 2),
            "model_used": model_used,
            "word_count": len(text.split()),
            "summary_words": len(combined.split())
        }
        
        return combined, metrics
    except Exception as e:
        return f"Error: {str(e)}", None

# Flask Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/summarize_text", methods=["POST"])
def summarize_text_route():
    text = request.form.get("data", "").strip()
    if not text:
        return jsonify(error="Please enter some text"), 400
    if len(text) > MAX_TEXT_LENGTH:
        return jsonify(error=f"Text exceeds {MAX_TEXT_LENGTH//1000}KB limit"), 413
    
    try:
        max_length = min(int(request.form.get("maxL", 300)), 500)
        high_quality = request.form.get("high_quality", "").lower() == "true"
        summary, metrics = summarize_with_metrics(text, max_length, high_quality)
        
        if isinstance(summary, str) and metrics:
            return jsonify({
                "summary": summary,
                "metrics": metrics,
                "performance": (
                    f"Processed {metrics['word_count']} words → {metrics['summary_words']} words "
                    f"in {metrics['processing_time']}s (ROUGE-L: {metrics['rougeL']})"
                )
            })
        return jsonify(summary=summary)
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route("/summarize_pdf", methods=["POST"])
def summarize_pdf_route():
    if "file" not in request.files:
        return jsonify(error="No file uploaded"), 400
    
    file = request.files["file"]
    if not file.filename:
        return jsonify(error="No file selected"), 400
    if file.content_length > MAX_FILE_SIZE:
        return jsonify(error="File too large (max 30MB)"), 413
    
    filename = secure_filename(file.filename)
    pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(pdf_path)
    
    try:
        text = extract_text_from_pdf(pdf_path)
        if "Error" in text or "short" in text:
            return jsonify(error=text), 400
        
        high_quality = request.form.get("high_quality", "").lower() == "true"
        summary, metrics = summarize_with_metrics(text, 400, high_quality)
        
        if isinstance(summary, str) and metrics:
            return jsonify({
                "summary": summary,
                "metrics": metrics,
                "performance": (
                    f"Processed PDF ({metrics['word_count']} words) → {metrics['summary_words']} words "
                    f"in {metrics['processing_time']}s (Quality: {metrics['model_used']})"
                )
            })
        return jsonify(summary=summary)
    except Exception as e:
        return jsonify(error=str(e)), 500
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)