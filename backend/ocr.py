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

# Helper function to get a short unique identifier for an image
def get_image_hash(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()[:8]

def upload_gemini(image: str):
    """Synchronous version of Gemini upload"""
    img_hash = get_image_hash(image.encode('utf-8'))
    logger.info(f"Gemini OCR: {img_hash}")
    try:
        response = completion(
            model="gemini/gemini-2.0-flash",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": """Analyze this image and transcribe all text, with special attention to tables and ICD-10 codes:

Guidelines:
1. Table Formatting:
   - Preserve table structure using markdown table format
   - Align columns properly (left, center, right as in original)
   - Keep header rows distinct
   - Maintain cell spacing and formatting
   - Format as:
     | Column1 | Column2 | Column3 |
     |---------|---------|---------|
     | Data1   | Data2   | Data3   |

2. Text Formatting:
   - Make all handwritten text **bold** using markdown
   - Keep printed text in regular format
   - Preserve original line breaks and spacing
   - Maintain paragraph structure

3. Special Elements:
   - Tables: Use markdown table format
   - Lists: Preserve bullets/numbers
   - Checkboxes: Mark as [✓], [×], or [ ]
   - Forms: Show labels and fields clearly

4. Table Content Rules:
   - Keep numerical data aligned properly
   - Preserve any column headers
   - Maintain any table titles or captions
   - Keep cell content formatting (bold, regular)
   - Note any merged cells or spans
   - Indicate empty cells with '-'

5. Special Instructions:
   - Clearly distinguish between printed and handwritten text
   - Use **bold** for handwritten content
   - Keep original case and punctuation
   - Preserve numerical formats
   - Skip any struck-through text
6. ICD-10 Code Recognition:
   - Look for codes matching pattern: [A-Z][0-9][0-9](\.[0-9]{1,2})?
   - For any code starting with '2', replace it with 'Z'
   - For any code starting with '1', replace it with 'I'
   - Example corrections:
     • 210.24 should be Z10.24
     • 217.24 should be Z17.24
     • 254.50 should be Z54.50
     • 110.24 should be I10.24
     • 117.24 should be I17.24
     • 154.50 should be I54.50
   - Valid examples:
     • Z10.24 - Primary diagnosis
     • I17.24 - Secondary diagnosis
     • M54.50 - Low back pain
     • K21.9 - GERD
   - Invalid patterns to ignore:
     • 10.24 (missing letter)
     • Z10.245 (extra digits)
     • Z.10.24 (wrong format)

7. Code Correction Rules:
   - Always check first character of codes
   - If first character is '2', replace with 'Z'
   - If first character is '1', replace with 'I'
   - Apply this rule to both diagnosis codes and ICD-10 codes
   - Examples:
     | Original | Corrected |
     |----------|-----------|
     | 210.24   | Z10.24    |
     | 217.24   | Z17.24    |
     | 110.24   | I10.24    |
     | 117.24   | I17.24    |

8. Special Code Instructions:
   - Always preserve the letter prefix
   - Keep decimal points exactly as shown
   - Match codes with their descriptions
   - Bold any handwritten annotations
   - Preserve code order from document
Guidelines:
1. Text Types:
   - "printed": Regular printed text
   - "handwritten": Handwritten annotations
   - "icd-code": Medical/ICD-10 codes

If there are any strike throughs or any 
if there is a fax number or phone number it should not  convert 2 to z and 1 to I. all of it should be numbers in fax, mobile number and member id, NPI number etc
Important: When encountering any medical or diagnosis code:
- If it starts with '2', automatically convert to 'Z'
- If it starts with '1', automatically convert to 'I'
- Keep the rest of the code unchanged
- Ensure all codes begin with a letter, not a number"""},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image}"}
                ]}
            ]
        )
        
        # Get actual response cost
        response_cost = response._hidden_params.get("response_cost", 0)
        logger.info(f"Response cost for {img_hash}: ${response_cost:.6f}")
        
        # Extract token usage
        token_usage = {
            "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
            "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
            "total_tokens": getattr(response.usage, 'total_tokens', 0)
        }
        
        return {
            "text": response.choices[0].message.content,
            "token_usage": token_usage,
            "cost": response_cost
        }
    except Exception as e:
        logger.error(f"Gemini error: {str(e)}")
        return {
            "text": f"Error with Gemini: {str(e)}",
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
            dpi=(3200, 3200)
        )
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        logger.info(f"Preprocessed page {page_num}/{total_pages}")
        return img_base64
    except Exception as e:
        logger.error(f"Preprocessing error on page {page_num}: {str(e)}")
        raise

def process_file(file: UploadFile) -> list[str]:
    """Synchronous version of process_file"""
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

@app.post("/upload-ocr")
def upload_all(file: UploadFile = File(...)):
    """Synchronous version of upload_all"""
    try:
        start_time = time()
        logger.info(f"Starting OCR: {file.filename}")
        
        base64_images = process_file(file)
        logger.info(f"File processed: {len(base64_images)} image(s)")
        
        all_results = []
        total_token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        total_cost = 0
        
        # Process pages sequentially
        for page_num, image_base64 in enumerate(base64_images, 1):
            try:
                result = upload_gemini(image=image_base64)
                logger.info(f"Gemini complete for page {page_num}")
                
                # Update token usage and cost
                for key in total_token_usage:
                    total_token_usage[key] += result["token_usage"][key]
                total_cost += result["cost"]
                
                all_results.append({
                    "page": page_num,
                    "text": result["text"],
                    "cost": result["cost"]
                })
            except Exception as e:
                logger.error(f"Gemini error on page {page_num}: {str(e)[:100]}")
                all_results.append({
                    "page": page_num,
                    "text": f"Error processing page {page_num}: {str(e)[:100]}",
                    "cost": 0
                })
        
        total_time = time() - start_time
        
        return {
            "total_pages": len(base64_images),
            "results": all_results,
            "processing_time_seconds": total_time,
            "total_token_usage": total_token_usage,
            "total_cost": total_cost
        }

    except Exception as e:
        logger.error(f"Error in upload_all: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add a helper function to calculate estimated cost
def calculate_cost(total_tokens: int) -> float:
    """Calculate estimated cost based on Gemini's pricing"""
    # Current Gemini pricing (subject to change)
    COST_PER_1K_TOKENS = 0.00025  # $0.00025 per 1K tokens
    return (total_tokens / 1000) * COST_PER_1K_TOKENS

@app.post("/convert-preview")
async def convert_preview(file: UploadFile = File(...)):
    """Convert PDF pages to JPG previews"""
    try:
        content_type = file.content_type
        file_bytes = await file.read()
        file_hash = get_image_hash(file_bytes)
        
        logger.info(f"Preview request: {file.filename} ({content_type}) - {file_hash}")
        
        if content_type == "application/pdf":
            try:
                # Convert all PDF pages
                images = convert_from_bytes(
                    file_bytes,
                    dpi=200,
                    fmt='jpeg',
                    thread_count=2,
                    size=(1000, None),
                    grayscale=False,
                    use_cropbox=True,
                    strict=False
                )
                
                if not images:
                    raise HTTPException(status_code=400, detail="No pages found in PDF")
                
                # Convert all pages to base64
                previews = []
                for i, image in enumerate(images):
                    img_byte_arr = io.BytesIO()
                    image.convert('RGB').save(
                        img_byte_arr,
                        format='JPEG',
                        quality=85,
                        optimize=True
                    )
                    img_byte_arr.seek(0)
                    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                    previews.append(img_base64)
                
                return {
                    "pages": previews,
                    "total_pages": len(previews)
                }
                
            except Exception as pdf_error:
                logger.error(f"PDF preview error: {str(pdf_error)}")
                raise HTTPException(status_code=400, detail=str(pdf_error))
        
        elif content_type.startswith('image/'):
            # Handle single image
            image = Image.open(io.BytesIO(file_bytes))
            # ... existing image processing ...
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
            img_byte_arr.seek(0)
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            
            return {
                "pages": [img_base64],
                "total_pages": 1
            }
            
        else:
            raise HTTPException(status_code=415, detail="Unsupported file type")
            
    except Exception as e:
        logger.error(f"Preview error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-keys")
async def test_keys():
    """Test if API keys are loaded correctly"""
    return {
        "gemini_key_present": bool(os.getenv("GEMINI_API_KEY"))
    }