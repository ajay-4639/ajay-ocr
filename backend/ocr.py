import os
import base64
import litellm
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set API key
os.environ["OPENAI_API_KEY"] = "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"

app = FastAPI()

# Helper functions for OCR processing
async def upload_openai(image: str):
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
        logger.debug("OpenAI API call successful")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in upload_openai: {str(e)}", exc_info=True)
        return f"Error with OpenAI: {str(e)}"

async def upload_gemma(image: str):
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
        logger.debug("Gemma API call successful")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in upload_gemma: {str(e)}", exc_info=True)
        return f"Error with Gemma: {str(e)}"

async def upload_llama(image: str):
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
        logger.debug("LLaMA API call successful")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in upload_llama: {str(e)}", exc_info=True)
        return f"Error with LLaMA: {str(e)}"

async def upload_llava(image: str):
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
        logger.debug("LLaVA API call successful")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in upload_llava: {str(e)}", exc_info=True)
        return f"Error with LLaVA: {str(e)}"

# Main endpoint
@app.post("/upload-ocr")
async def upload_all(image: UploadFile = File(...)):
    logger.info("Starting OCR process for uploaded image")
    try:
        image_bytes = await image.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        logger.debug(f"Image successfully converted to base64. Length: {len(image_base64)}")
        logger.debug("Starting OpenAI OCR...")
        openai_text = await upload_openai(image_base64)
        logger.debug("Starting Gemma OCR...")
        gemma_text = await upload_gemma(image_base64)
        logger.debug("Starting LLaMA OCR...")
        llama_text = await upload_llama(image_base64)
        logger.debug("Starting LLaVA OCR...")
        llava_text = await upload_llava(image_base64)

        logger.debug("Starting GPT-4o extraction...")
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
        logger.debug(f"GPT-4o extraction completed. Text length: {len(gpt4o_text)}")

        ranking_prompt = f"""
        Rank the following OCR results based on their similarity to the extracted text from GPT-4o.
        The ranking should be from 1 (most similar) to 4 (least similar).
        
        **GPT-4o Extracted Text:** 
        {gpt4o_text}

        **OCR Outputs:**
        OpenAI: {openai_text}
        Gemma: {gemma_text}
        LLaMA: {llama_text}
        LLaVA: {llava_text}

        Provide ranking order and output text in this format and the output text should be for the best model:
        1. [Best Model]: [Output Text]
        2. [Second Best]
        3. [Third Best]
        4. [Worst Model]
        """

        logger.debug("Starting ranking process...")
        ranking_response = litellm.completion(
            model="gpt-4o",
            messages=[{"content": ranking_prompt, "role": "user"}],
            max_tokens=1000
        )
        logger.info("OCR process completed successfully")

        return {
            "response": ranking_response.choices[0].message.content,
        }

    except Exception as e:
        logger.error(f"Error in upload_all: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))