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

# Configure minimal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('ocr_process.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
logger.info("Loading environment variables")
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY not found in environment variables")
logger.info("API key loaded successfully")

os.environ["OPENAI_API_KEY"] = api_key

app = FastAPI()

async def upload_openai(image: str):
    start_time = time()
    logger.info("Starting OpenAI OCR processing")
    try:
        logger.info("Making OpenAI API call")
        response = litellm.completion(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract text from this image:"},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image}"}
                ]}
            ]
        )
        logger.info(f"OpenAI API call completed in {time() - start_time:.2f} seconds")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in OpenAI processing: {str(e)}", exc_info=True)
        return f"Error with OpenAI: {str(e)}"

async def upload_gemma(image: str):
    start_time = time()
    logger.info("Starting Gemma OCR processing")
    try:
        logger.info("Making Gemma API call")
        response = litellm.completion(
            model="ollama/gemma3",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract text from this image:"},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image}"}
                ]}
            ]
        )
        logger.info(f"Gemma API call completed in {time() - start_time:.2f} seconds")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in Gemma processing: {str(e)}", exc_info=True)
        return f"Error with Gemma: {str(e)}"

async def upload_llama(image: str):
    start_time = time()
    logger.info("Starting LLaMA OCR processing")
    try:
        logger.info("Making LLaMA API call")
        response = litellm.completion(
            model="ollama/llama3.2-vision",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract text from this image:"},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image}"}
                ]}
            ]
        )
        logger.info(f"LLaMA API call completed in {time() - start_time:.2f} seconds")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in LLaMA processing: {str(e)}", exc_info=True)
        return f"Error with LLaMA: {str(e)}"

async def upload_llava(image: str):
    start_time = time()
    logger.info("Starting LLaVA OCR processing")
    try:
        logger.info("Making LLaVA API call")
        response = litellm.completion(
            model="ollama/llava",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract text from this image:"},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image}"}
                ]}
            ]
        )
        logger.info(f"LLaVA API call completed in {time() - start_time:.2f} seconds")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in LLaVA processing: {str(e)}", exc_info=True)
        return f"Error with LLaVA: {str(e)}"

async def process_file(file: UploadFile) -> list[str]:
    """Process uploaded file (PDF or image) and return list of base64 encoded images"""
    content_type = file.content_type
    file_bytes = await file.read()
    
    logger.info(f"Processing file: {file.filename} ({content_type})")
    
    if content_type == 'application/pdf':
        try:
            logger.info("Converting PDF to images")
            # Convert PDF to images
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
                temp_pdf.write(file_bytes)
                temp_pdf.flush()
                
                # Convert PDF pages to images
                images = convert_from_bytes(file_bytes, 500)  # 500 DPI for quality
                logger.info(f"Converted PDF: {len(images)} pages")
                
                # Convert each page to base64
                base64_images = []
                for i, image in enumerate(images, 1):
                    buffered = io.BytesIO()
                    image.save(buffered, format="JPEG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    base64_images.append(img_base64)
                    logger.info(f"Processed PDF page {i}")
                    
                return base64_images
                
        except Exception as e:
            logger.error(f"Error converting PDF: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")
            
    elif content_type.startswith('image/'):
        try:
            # Process single image
            image = Image.open(io.BytesIO(file_bytes))
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            logger.info("Processed single image")
            return [img_base64]
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
    else:
        raise HTTPException(status_code=415, detail="Unsupported file type")

@app.post("/upload-ocr")
async def upload_all(file: UploadFile = File(...)):
    try:
        # Process file (PDF or image)
        base64_images = await process_file(file)
        logger.info(f"File processed successfully: {len(base64_images)} image(s)")
        
        all_results = []
        for page_num, image_base64 in enumerate(base64_images, 1):
            logger.info(f"Processing page/image {page_num}/{len(base64_images)}")
            
            # Process with different models
            openai_text = await upload_openai(image_base64)
            gemma_text = await upload_gemma(image_base64)
            llama_text = await upload_llama(image_base64)
            llava_text = await upload_llava(image_base64)

            # Get GPT-4o reference text
            gpt4o_response = litellm.completion(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": "Extract the text from this image accurately."},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
                    ]}
                ],
                max_tokens=1000
            )
            
            gpt4o_text = gpt4o_response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Generate ranking for this page
            ranking_prompt = f"""
            Rank the following OCR results based on their similarity to the extracted text from GPT-4o.
            The ranking should be from 1 (most similar) to 4 (least similar).
            Calculate accuracy percentage compared to the best performing model.
            
            **GPT-4o Extracted Text:** 
            {gpt4o_text}

            **OCR Outputs:**
            OpenAI: {openai_text}
            Gemma: {gemma_text}
            LLaMA: {llama_text}
            LLaVA: {llava_text}

            Provide ranking order in this format:
            1. [Best Model]: [Output Text] (100% - Reference)
            2. [Second Best Model] (Accuracy: X% compared to best)
            3. [Third Best Model] (Accuracy: Y% compared to best)
            4. [Worst Model] (Accuracy: Z% compared to best)

            Include a 1-2 lines brief explanation of why each model performed better or worse.
            """

            ranking_response = litellm.completion(
                model="gpt-4o",
                messages=[{"content": ranking_prompt, "role": "user"}],
                max_tokens=1000
            )
            
            all_results.append({
                "page": page_num,
                "response": ranking_response.choices[0].message.content
            })
            
            logger.info(f"Completed processing page/image {page_num}")

        return {
            "total_pages": len(base64_images),
            "results": all_results
        }

    except Exception as e:
        logger.error(f"Error in upload_all: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/convert-preview")
async def convert_preview(file: UploadFile = File(...)):
    """Convert first page of PDF to JPEG for preview"""
    try:
        if not file.content_type == "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
            
        file_bytes = await file.read()
        
        # Convert first page of PDF to image
        images = convert_from_bytes(file_bytes, 500)  # 500 DPI for quality
        if not images:
            raise HTTPException(status_code=400, detail="Failed to convert PDF")
            
        # Get first page and convert to JPEG
        first_page = images[0]
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        return Response(
            content=img_byte_arr.getvalue(), 
            media_type="image/jpeg"
        )
        
    except Exception as e:
        logger.error(f"Error generating PDF preview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))