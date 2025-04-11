import os
import base64
import litellm
import logging
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
openai_key = os.getenv("OPENAI_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

if not openai_key:
    logger.error("OPENAI_API_KEY not found")
    raise ValueError("OPENAI_API_KEY not found")
if not gemini_key:
    logger.error("GEMINI_API_KEY not found")
    raise ValueError("GEMINI_API_KEY not found")

os.environ["OPENAI_API_KEY"] = openai_key
os.environ["GEMINI_API_KEY"] = gemini_key

app = FastAPI()

# Helper function to get a short unique identifier for an image
def get_image_hash(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()[:8]

# Use different models or different prompts to ensure variety in results
async def upload_openai(image: str):
    img_hash = get_image_hash(image.encode('utf-8'))
    logger.info(f"OpenAI OCR: {img_hash}")
    try:
        response = litellm.completion(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": """Please extract and transcribe all text and form elements from this image exactly as it appears:

Instructions:
1. Maintain the exact layout and positioning of text
2. Preserve all:
   - Line breaks
   - Paragraph spacing
   - Indentation
   - Text alignment
   - Column formats

3. For form elements:
   - Mark checkboxes as: [✓] for checked, [×] for unchecked, [ ] for empty
   - Note checkbox labels and their state
   - Preserve checkbox groupings if any
   - Indicate any related form fields

4. Keep exact table structures and maintain:
   - Bullet points or numbering
   - Section headings
   - Font variations (if visible)

Output Format:
# Document Title/Header (if any)

## Main Content
[Preserve exact layout as in image]

## Form Elements (if any)
Checkboxes:
- [ ] Unchecked option
- [✓] Checked option
- [×] Cross marked option

## Tables (if any)
[Maintain column alignment and spacing]

## Lists (if any)
[Keep original bullet style/numbering]

Note: Pay special attention to form elements and ensure checkbox states are clearly indicated."""},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image}"}
                ]}
            ]
        )
        logger.info(f"OpenAI complete: {img_hash}")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI error: {str(e)[:100]}")
        return f"Error with OpenAI: {str(e)[:100]}"

async def upload_gemini(image: str):
    img_hash = get_image_hash(image.encode('utf-8'))
    logger.info(f"Gemini OCR: {img_hash}")
    try:
        response = litellm.completion(
            model="gemini/gemini-2.0-flash",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": """Analyze this image and transcribe all text and form elements exactly as they appear:

Guidelines:
1. Preserve visual layout:
   - Keep exact line breaks
   - Match paragraph spacing
   - Maintain text alignment
   - Keep column structure

2. For form elements and checkboxes:
   - Use [✓] for checked boxes
   - Use [×] for unchecked boxes
   - Use [ ] for empty boxes
   - Include checkbox labels and context
   - Group related checkboxes together

3. For text elements:
   - Copy headings as positioned
   - Keep original indentation
   - Maintain list formatting
   - Preserve table layouts

4. Format tables and lists:
   - Keep original markers
   - Maintain indentation levels
   - Preserve spacing between items

Output Structure:
# Document Content
[Main text with preserved formatting]

# Form Elements
Checkboxes and Form Fields:
- [ ] Option 1: [Context/Label]
- [✓] Option 2: [Context/Label]
- [×] Option 3: [Context/Label]

Remember: Accurately represent all form elements and their states."""},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image}"}
                ]}
            ]
        )
        logger.info(f"Gemini complete: {img_hash}")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Gemini error: {str(e)[:100]}")
        return f"Error with Gemini: {str(e)[:100]}"

# Add supported file types constant
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
    'image/heic': '.heic',
    'image/heif': '.heif'
}

async def convert_to_jpg(file_content: bytes, content_type: str) -> Image.Image:
    """Convert various file formats to JPG"""
    try:
        # For standard image formats
        if content_type.startswith('image/'):
            image = Image.open(io.BytesIO(file_content))
            
            # Convert to RGB if necessary
            if image.mode in ('RGBA', 'P'):
                image = image.convert('RGB')
                
            # Create a new image with white background for transparency
            if image.mode == 'LA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[1])
                image = background
            
            return image
            
        elif content_type == 'application/pdf':
            # Handle PDF conversion
            images = convert_from_bytes(file_content, 500)
            if not images:
                raise ValueError("Could not convert PDF")
                
            return images[0].convert('RGB')
                
        else:
            raise ValueError(f"Unsupported format: {content_type}")
            
    except Exception as e:
        logger.error(f"Conversion error: {str(e)[:100]}")
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to convert file: {str(e)[:100]}"
        )

async def process_file(file: UploadFile) -> list[str]:
    """Process uploaded file and return list of base64 encoded images"""
    try:
        content_type = file.content_type
        file_bytes = await file.read()
        file_hash = get_image_hash(file_bytes)
        
        logger.info(f"Processing file: {file.filename} ({content_type}, hash: {file_hash})")
        
        if content_type not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type. Supported types: {', '.join(SUPPORTED_FORMATS.keys())}"
            )
        
        if content_type == 'application/pdf':
            # Convert PDF pages to images
            images = convert_from_bytes(file_bytes, 500)
            logger.info(f"PDF {file_hash}: {len(images)} pages")
            
            # Convert each page to JPG and then base64
            base64_images = []
            for i, image in enumerate(images, 1):
                # Convert to RGB
                rgb_image = image.convert('RGB')
                
                # Save as JPEG (use 'JPEG' for PIL format string)
                buffered = io.BytesIO()
                rgb_image.save(buffered, format="JPEG", quality=95)
                img_bytes = buffered.getvalue()
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                base64_images.append(img_base64)
                logger.info(f"Processed PDF page {i}/{len(images)}")
                
            return base64_images
            
        else:
            # Convert single file to JPG
            image = await convert_to_jpg(file_bytes, content_type)
            
            # Save as JPEG (use 'JPEG' for PIL format string)
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=95)
            img_bytes = buffered.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            logger.info(f"Processed image {file_hash}")
            
            return [img_base64]
            
    except Exception as e:
        logger.error(f"Processing error: {str(e)[:100]}")
        raise HTTPException(status_code=400, detail=f"Processing error: {str(e)[:100]}")

@app.post("/upload-ocr")
async def upload_all(file: UploadFile = File(...)):
    try:
        start_time = time()
        logger.info(f"Starting OCR: {file.filename}")
        
        base64_images = await process_file(file)
        logger.info(f"File processed: {len(base64_images)} image(s)")
        
        all_results = []
        for page_num, image_base64 in enumerate(base64_images, 1):
            page_start = time()
            img_hash = get_image_hash(image_base64.encode('utf-8'))
            logger.info(f"Processing page {page_num}/{len(base64_images)} - {img_hash}")
            
            # Process sequentially with OpenAI first, then Gemini
            try:
                openai_text = await upload_openai(image_base64)
                logger.info(f"OpenAI processing complete for page {page_num}")
            except Exception as e:
                logger.error(f"OpenAI processing failed: {str(e)[:100]}")
                openai_text = "Error processing with OpenAI"

            try:
                gemini_text = await upload_gemini(image_base64)
                logger.info(f"Gemini processing complete for page {page_num}")
            except Exception as e:
                logger.error(f"Gemini processing failed: {str(e)[:100]}")
                gemini_text = "Error processing with Gemini"
            
            # Store results
            all_results.append({
                "page": page_num,
                "openai_output": openai_text,
                "gemini_output": gemini_text
            })
            
            logger.info(f"Page {page_num} done in {time() - page_start:.2f}s")

        total_time = time() - start_time
        logger.info(f"All processing done in {total_time:.2f}s")
        
        return {
            "total_pages": len(base64_images),
            "results": all_results,
            "processing_time_seconds": total_time
        }

    except Exception as e:
        logger.error(f"Error in upload_all: {str(e)[:100]}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/convert-preview")
async def convert_preview(file: UploadFile = File(...)):
    """Convert first page of PDF to JPG for preview"""
    try:
        file_hash = get_image_hash(await file.read())
        await file.seek(0)
        
        logger.info(f"PDF preview: {file.filename} - {file_hash}")
        
        if not file.content_type == "application/pdf":
            logger.error(f"Invalid file type: {file.content_type}")
            raise HTTPException(status_code=400, detail="Only PDF files supported")
            
        file_bytes = await file.read()
        
        # Convert first page of PDF to image
        images = convert_from_bytes(file_bytes, 500)
        if not images:
            logger.error(f"Failed to convert PDF {file_hash}")
            raise HTTPException(status_code=400, detail="Failed to convert PDF")
            
        # Get first page and convert to JPG
        first_page = images[0]
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPG')  # Changed from JPEG to JPG
        img_byte_arr.seek(0)
        
        logger.info(f"PDF preview generated: {file_hash}")
        
        return Response(
            content=img_byte_arr.getvalue(), 
            media_type="image/jpeg"  # Keep this as image/jpeg for MIME type compatibility
        )
        
    except Exception as e:
        logger.error(f"Preview error: {str(e)[:100]}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-keys")
async def test_keys():
    """Test if API keys are loaded correctly"""
    return {
        "openai_key_present": bool(os.getenv("OPENAI_API_KEY")),
        "gemini_key_present": bool(os.getenv("GEMINI_API_KEY"))
    }