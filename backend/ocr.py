import os
import base64
import litellm
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from time import time

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('ocr_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
logger.debug("Loading environment variables")
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY not found in environment variables")
logger.debug("API key loaded successfully")

os.environ["OPENAI_API_KEY"] = api_key

app = FastAPI()

async def upload_openai(image: str):
    start_time = time()
    logger.debug("Starting OpenAI OCR processing")
    try:
        logger.debug("Making OpenAI API call")
        response = litellm.completion(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract text from this image:"},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image}"}
                ]}
            ]
        )
        logger.debug(f"OpenAI API call completed in {time() - start_time:.2f} seconds")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in OpenAI processing: {str(e)}", exc_info=True)
        return f"Error with OpenAI: {str(e)}"

async def upload_gemma(image: str):
    start_time = time()
    logger.debug("Starting Gemma OCR processing")
    try:
        logger.debug("Making Gemma API call")
        response = litellm.completion(
            model="ollama/gemma3",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract text from this image:"},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image}"}
                ]}
            ]
        )
        logger.debug(f"Gemma API call completed in {time() - start_time:.2f} seconds")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in Gemma processing: {str(e)}", exc_info=True)
        return f"Error with Gemma: {str(e)}"

async def upload_llama(image: str):
    start_time = time()
    logger.debug("Starting LLaMA OCR processing")
    try:
        logger.debug("Making LLaMA API call")
        response = litellm.completion(
            model="ollama/llama3.2-vision",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract text from this image:"},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image}"}
                ]}
            ]
        )
        logger.debug(f"LLaMA API call completed in {time() - start_time:.2f} seconds")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in LLaMA processing: {str(e)}", exc_info=True)
        return f"Error with LLaMA: {str(e)}"

async def upload_llava(image: str):
    start_time = time()
    logger.debug("Starting LLaVA OCR processing")
    try:
        logger.debug("Making LLaVA API call")
        response = litellm.completion(
            model="ollama/llava",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract text from this image:"},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image}"}
                ]}
            ]
        )
        logger.debug(f"LLaVA API call completed in {time() - start_time:.2f} seconds")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in LLaVA processing: {str(e)}", exc_info=True)
        return f"Error with LLaVA: {str(e)}"

@app.post("/upload-ocr")
async def upload_all(image: UploadFile = File(...)):
    total_start_time = time()
    logger.info(f"Starting OCR process for file: {image.filename}")
    try:
        # Read and encode image
        logger.debug("Reading image file")
        image_bytes = await image.read()
        image_size = len(image_bytes)
        logger.debug(f"Image size: {image_size / 1024:.2f}KB")
        
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        logger.debug("Image successfully encoded to base64")

        # Process with different models
        logger.debug("Starting model processing")
        openai_start = time()
        openai_text = await upload_openai(image_base64)
        logger.debug(f"OpenAI processing completed in {time() - openai_start:.2f}s")

        gemma_start = time()
        gemma_text = await upload_gemma(image_base64)
        logger.debug(f"Gemma processing completed in {time() - gemma_start:.2f}s")

        llama_start = time()
        llama_text = await upload_llama(image_base64)
        logger.debug(f"LLaMA processing completed in {time() - llama_start:.2f}s")

        llava_start = time()
        llava_text = await upload_llava(image_base64)
        logger.debug(f"LLaVA processing completed in {time() - llava_start:.2f}s")

        # Get GPT-4o reference text
        logger.debug("Getting GPT-4o reference text")
        gpt4o_start = time()
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
        logger.debug(f"GPT-4o reference text obtained in {time() - gpt4o_start:.2f}s")

        gpt4o_text = gpt4o_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        logger.debug(f"GPT-4o text length: {len(gpt4o_text)} characters")

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

        # Generate ranking
        logger.debug("Starting ranking process")
        ranking_start = time()
        ranking_response = litellm.completion(
            model="gpt-4o",
            messages=[{"content": ranking_prompt, "role": "user"}],
            max_tokens=1000
        )
        logger.debug(f"Ranking completed in {time() - ranking_start:.2f}s")

        total_time = time() - total_start_time
        logger.info(f"Total OCR process completed in {total_time:.2f} seconds")
        
        return {
            "response": ranking_response.choices[0].message.content,
            "processing_time": f"{total_time:.2f}s"
        }

    except Exception as e:
        logger.error(f"Error in upload_all: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))