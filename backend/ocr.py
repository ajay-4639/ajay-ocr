import os
import base64
import litellm
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from time import time
from pdf2image import convert_from_bytes
from PIL import Image
import io
import tempfile
import hashlib
from pathlib import Path
import numpy as np
import cv2
from litellm import completion
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from fastapi.middleware.cors import CORSMiddleware

# Configure minimal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ocr_process.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")

if not gemini_key:
    logger.error("GEMINI_API_KEY not found")
    raise ValueError("GEMINI_API_KEY not found")

os.environ["GEMINI_API_KEY"] = gemini_key

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper function to get a short unique identifier for an image
def get_image_hash(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()[:8]

class BoundingRegion(BaseModel):
    x: float
    y: float

class FormElement(BaseModel):
    label: str
    canonicalId: str
    labelName: str 
    formname: str
    encompassValue: str
    value: str
    boundingRegions: List[BoundingRegion]
    page: int
    order: int

class OCRResult(BaseModel):
    status: int = 200
    message: str = "Encompass data read successfully."
    body: Dict = {}
    formData: List[FormElement]

def upload_gemini(image: str):
    """Perform OCR using Gemini with bounding boxes"""
    try:
        response = completion(
            model="gemini/gemini-2.0-flash",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": """Analyze the uploaded image and transcribe all visible text, focusing especially on tables, ICD-10 codes, forms, and handwritten notes. Follow these detailed instructions to ensure accurate extraction, correction, and formatting of all content:

Table Formatting

Detect and preserve all table structures using proper markdown table format.

Maintain correct alignment of columns (left, center, right as originally shown).

Distinctly identify header rows.

Keep consistent spacing and formatting in all cells.

Format using this structure:


Column1	Column2	Column3
Data1	Data2	Data3
Accurately reflect any merged cells, cell spans, or empty cells using dash symbols.

Text Formatting

Make all handwritten text bold using markdown.

Keep all printed text in regular formatting.

Retain original line breaks and spacing.

Maintain paragraph structures and document flow as in the image.

Forms and Fields

Transcribe form elements with clear label and value layout.

Preserve checkboxes and display them as: [✓] for checked [×] for crossed [ ] for empty

Lists and Bullets

Preserve bullet points and numbered lists with their original formatting.

ICD-10 Code Handling

Recognize and extract all valid ICD-10 codes using the format: A single uppercase letter followed by 2 digits, optionally followed by a dot and 1 or 2 digits.

Example valid format: Z10.24, I17.24, M54.50, K21.9

If a code starts with the digit 2, replace it with the letter Z.

If a code starts with the digit 1, replace it with the letter I.

Always retain the decimal point and digits as shown.

Apply these correction rules: 210.24 becomes Z10.24
217.24 becomes Z17.24
110.24 becomes I10.24
117.24 becomes I17.24
154.50 becomes I54.50

Code Validation and Filtering

Exclude any diagnosis or ICD-10 codes that:

Are missing the initial letter (e.g. 10.24)

Have more than two digits after the decimal (e.g. Z10.245)

Have malformed patterns (e.g. Z.10.24)

Data Type Preservation

Do NOT apply the above letter substitutions to phone numbers, fax numbers, zip codes, ID numbers, NPI numbers, or any numerically formatted data.

Treat these values as pure numbers and do not alter them.

Spelling and Geographic Correction

Intelligently correct any spelling mistakes, especially for names, city names, street names, state abbreviations, and common medical terms.

Use contextual understanding to replace misspelled place names with the closest valid real-world equivalent.

For any city or place mentioned, include the ZIP code corresponding to that city (but do not convert the ZIP code into a city name).

Ensure factual geographic consistency: for example, if a state or city is incorrectly paired, correct it to the most likely valid combination.

Use best judgment and contextual intelligence to infer and correct any ambiguities in the text related to names, places, and factual content.

Output Requirements

Output the final result in plain text only, no special characters or formatting outside of markdown standards.

Clearly separate different sections like text blocks, tables, and codes.

Maintain document structure and readability.

Special Judgment Handling

Use intelligent pattern recognition to improve text accuracy.

Match partial or incorrectly written terms to the most realistic, valid equivalent using general knowledge (e.g. ‘New Yrok’ becomes ‘New York’, ‘San Joes’ becomes ‘San Jose’).

Respect the intended meaning even if handwritten or printed characters are ambiguous.
If a word is partially obscured or unclear, use contextual clues to infer the most likely intended word."""},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image}"}
                ]}
            ]
        )
        
        content = response.choices[0].message.content
        logger.info(f"Raw LLM response: {content}")
        
        # Return simple result with raw text
        result = {
            "elements": [{
                "value": content,
                "label": "Extracted Text"
            }],
            "raw_response": content,
            "error": None,
            "token_usage": {
                "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                "total_tokens": getattr(response.usage, 'total_tokens', 0)
            },
            "cost": 0
        }

        # Comment out JSON parsing section
        '''
        import re
        json_match = re.search(r'```json[\s]*(\{[\s\S]*?\})\s*```|(\{[\s\S]*?\})', content)
        if json_match:
            json_str = json_match.group(1) or json_match.group(2)
            json_str = json_str.strip()
            
            try:
                parsed_result = json.loads(json_str)
                if isinstance(parsed_result, dict) and "formData" in parsed_result:
                    result["elements"] = parsed_result["formData"]
                else:
                    logger.error("Invalid JSON structure: missing formData")
                    result["error"] = "Invalid response format"
            except json.JSONDecodeError as je:
                logger.error(f"JSON parsing error: {str(je)}")
                result["error"] = f"Failed to parse response: {str(je)}"
        else:
            logger.error("No JSON content found in response")
            result["error"] = "No valid JSON found in response"
        '''
            
        return result

    except Exception as e:
        logger.error(f"Gemini error: {str(e)}")
        return {
            "elements": [],
            "raw_response": None,
            "error": f"Processing error: {str(e)}",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "cost": 0
        }

# Update SUPPORTED_FORMATS to include PDF mime types
SUPPORTED_FORMATS = {
    # Images
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/gif': '.gif',
    'image/bmp': '.bmp',
    'image/tiff': '.tiff',
    'image/webp': '.webp',
    # Documents
    'application/pdf': '.pdf',
    'application/x-pdf': '.pdf',  # Alternative PDF mime type
    'image/heic': '.heic',
    'image/heif': '.heif'
}

def convert_to_jpg(file_content: bytes, content_type: str) -> Image.Image:
    """Synchronous version of convert_to_jpg"""
    try:
        if content_type in ['application/pdf', 'application/x-pdf']:
            try:
                images = convert_from_bytes(
                    file_content, 
                    dpi=300,
                    fmt='jpeg',
                    thread_count=2,
                    size=(2000, None)
                )
                if not images:
                    raise ValueError("No pages found in PDF")
                return images[0].convert('RGB')
            except Exception as pdf_error:
                logger.error(f"PDF conversion error: {str(pdf_error)}")
                raise ValueError(f"PDF conversion failed: {str(pdf_error)}")
        elif content_type.startswith('image/'):
            image = Image.open(io.BytesIO(file_content))
            if image.mode in ('RGBA', 'P', 'LA'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                if 'A' in image.getbands():
                    background.paste(image, mask=image.getchannel('A'))
                else:
                    background.paste(image)
                image = background
            return image
        else:
            raise ValueError(f"Unsupported format: {content_type}")
    except Exception as e:
        logger.error(f"Conversion error: {str(e)[:100]}")
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to convert file: {str(e)[:100]}"
        )

def normalize_image(image: Image.Image) -> Image.Image:
    """Normalize intensity for images with CLAHE"""
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale if image is RGB
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    normalized = clahe.apply(img_array)
    
    # Convert back to RGB mode for compatibility
    normalized_rgb = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
    
    # Convert back to PIL Image
    return Image.fromarray(normalized_rgb)

def preprocess_image(image: Image.Image, page_num: int, total_pages: int) -> str:
    """Synchronous version of preprocess_image"""
    try:
        rgb_image = image.convert('RGB')
        normalized_image = normalize_image(rgb_image)
        buffered = io.BytesIO()
        normalized_image.save(
            buffered, 
            format="JPEG", 
            quality=100,
            optimize=True,
            dpi=(64000, 64000)
        )
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        logger.info(f"Preprocessed page {page_num}/{total_pages}")
        return img_base64
    except Exception as e:
        logger.error(f"Preprocessing error on page {page_num}: {str(e)}")
        raise
 
def process_file(file: UploadFile) -> list[str]:
    """Process uploaded file and return list of base64 encoded images"""
    try:
        content_type = file.content_type
        file_bytes = file.file.read()
        file_hash = get_image_hash(file_bytes)
        
        logger.info(f"Processing file: {file.filename} ({content_type}, hash: {file_hash})")
        
        if content_type not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type. Supported types: {', '.join(SUPPORTED_FORMATS.keys())}"
            )
        
        if content_type in ['application/pdf', 'application/x-pdf']:
            try:
                images = convert_from_bytes(
                    file_bytes,
                    dpi=3200,
                    fmt='jpeg',
                    thread_count=4,
                    size=(8000, None)
                )
                
                if not images:
                    raise ValueError("No pages found in PDF")
                
                logger.info(f"PDF {file_hash}: Found {len(images)} pages")
                
                # Process pages sequentially
                base64_images = []
                for i, image in enumerate(images, 1):
                    img_base64 = preprocess_image(
                        image=image,
                        page_num=i,
                        total_pages=len(images)
                    )
                    base64_images.append(img_base64)
                
                return base64_images
                
            except Exception as pdf_error:
                logger.error(f"PDF processing error: {str(pdf_error)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to process PDF: {str(pdf_error)}"
                )
        else:
            image = convert_to_jpg(file_bytes, content_type)
            base64_image = preprocess_image(image, 1, 1)
            return [base64_image]
            
    except Exception as e:
        logger.error(f"Processing error: {str(e)[:100]}")
        raise HTTPException(status_code=400, detail=f"Processing error: {str(e)[:100]}")

@app.get("/test-keys")
async def test_keys():
    """Test if API keys are loaded correctly"""
    return {
        "gemini_key_present": bool(os.getenv("GEMINI_API_KEY"))
    }

@app.post("/convert-preview")
async def convert_preview(file: UploadFile = File(...)):
    """Convert uploaded file to base64 images for preview"""
    try:
        base64_images = process_file(file)
        return {
            "status": "success",
            "pages": base64_images
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/upload-ocr")
async def upload_ocr(file: UploadFile = File(...)):
    """Process file with OCR"""
    try:
        # Get base64 images
        base64_images = process_file(file)
        
        # Process each page with OCR
        start_time = time()
        results = []
        
        for i, image in enumerate(base64_images):
            ocr_result = upload_gemini(image)
            ocr_result["page"] = i + 1
            results.append(ocr_result)
            
        processing_time = time() - start_time
        
        return {
            "status": "success",
            "total_pages": len(base64_images),
            "processing_time_seconds": processing_time,
            "total_cost": 0,  # Update if you implement cost tracking
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))