from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import traceback
import logging
import os
import io
import tempfile
from typing import Optional
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Nova.AI Backend", version="1.0.0")

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Global variable to store Groq client
groq_client = None

# Request models
class TextRequest(BaseModel):
    text: str

class HealthResponse(BaseModel):
    status: str
    message: str
    models_loaded: dict

# Initialize Groq client
async def initialize_groq_client():
    global groq_client
    
    try:
        from groq import Groq
        
        # Try different ways to get the API key
        api_key = None
        
        # Method 1: Environment variable
        api_key = os.getenv("GROQ_API_KEY")
        
        # Method 2: Google Colab userdata (if available)
        if not api_key:
            try:
                from google.colab import userdata
                api_key = userdata.get("GROQ_API_KEY")
            except:
                pass
        
        if api_key:
            groq_client = Groq(api_key=api_key)
            logger.info("Groq client initialized successfully")
        else:
            logger.error("GROQ_API_KEY not found in environment variables")
            raise Exception("GROQ_API_KEY not found")
            
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        groq_client = None
        raise e

def validate_audio_file(content_type: str, file_size: int) -> tuple[bool, str]:
    """
    Validate audio file type and size according to Groq API requirements
    """
    # Groq supported formats
    supported_formats = [
        "audio/flac", "audio/mp3", "audio/mp4", "audio/mpeg", 
        "audio/mpga", "audio/m4a", "audio/ogg", "audio/wav", "audio/webm"
    ]
    
    # Also check for common MIME type variations
    supported_extensions = [
        "flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"
    ]
    
    # Check file size (19.5MB max for Groq)
    max_size = 19.5 * 1024 * 1024  # 19.5MB in bytes
    if file_size > max_size:
        return False, f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds maximum limit (19.5MB)"
    
    # Check content type
    if content_type in supported_formats:
        return True, "Valid format"
    
    # Check if content type contains supported extension
    for ext in supported_extensions:
        if ext in content_type.lower():
            return True, "Valid format"
    
    return False, f"Unsupported format: {content_type}. Supported formats: {', '.join(supported_extensions)}"

def convert_audio_if_needed(audio_bytes: bytes, content_type: str, filename: str) -> tuple[bytes, str]:
    """
    Convert audio to a supported format if needed using pydub
    """
    try:
        from pydub import AudioSegment
        
        # If it's already a supported format, return as-is
        supported_extensions = ["flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"]
        
        # Try to determine format from content type or filename
        format_hint = None
        if content_type:
            for ext in supported_extensions:
                if ext in content_type.lower():
                    format_hint = ext
                    break
        
        if not format_hint and filename:
            file_ext = filename.split('.')[-1].lower() if '.' in filename else None
            if file_ext in supported_extensions:
                format_hint = file_ext
        
        # If we have a supported format hint, try to use the audio as-is
        if format_hint in supported_extensions:
            return audio_bytes, format_hint
        
        # Otherwise, convert to WAV (widely supported)
        logger.info(f"Converting audio from {content_type} to WAV")
        
        # Load audio
        if format_hint:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=format_hint)
        else:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        
        # Convert to WAV
        output_buffer = io.BytesIO()
        audio.export(output_buffer, format="wav")
        converted_bytes = output_buffer.getvalue()
        
        logger.info(f"Successfully converted audio to WAV. Size: {len(converted_bytes)} bytes")
        return converted_bytes, "wav"
        
    except Exception as e:
        logger.error(f"Audio conversion failed: {e}")
        # Return original if conversion fails
        return audio_bytes, "webm"

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Nova.AI Backend...")
    try:
        await initialize_groq_client()
        logger.info("Backend startup completed successfully")
    except Exception as e:
        logger.error(f"Backend startup failed: {e}")
        # Don't raise here to allow health checks to show the error

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if groq_client is not None else "degraded",
        message="Nova.AI Backend is running" if groq_client is not None else "Groq client not initialized",
        models_loaded={
            "groq_client": groq_client is not None,
            "whisper_api": groq_client is not None,
            "text_generation": groq_client is not None
        }
    )

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Nova.AI Backend API", 
        "status": "running",
        "version": "2.0.0",
        "features": ["transcription", "summarization", "response_suggestions"]
    }

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if groq_client is None:
        raise HTTPException(
            status_code=503,
            detail="Groq client not available. Please check GROQ_API_KEY environment variable."
        )
    
    try:
        logger.info(f"Processing audio file: {audio.filename}, type: {audio.content_type}")
        
        # Read audio data
        audio_bytes = await audio.read()
        file_size = len(audio_bytes)
        
        logger.info(f"Read {file_size} bytes from audio file")
        
        # Validate file
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        is_valid, validation_message = validate_audio_file(audio.content_type or "", file_size)
        if not is_valid:
            raise HTTPException(status_code=400, detail=validation_message)
        
        # Convert audio if needed
        try:
            processed_audio_bytes, audio_format = convert_audio_if_needed(
                audio_bytes, 
                audio.content_type or "", 
                audio.filename or ""
            )
            logger.info(f"Audio processed successfully. Format: {audio_format}")
        except Exception as conversion_error:
            logger.warning(f"Audio conversion failed, using original: {conversion_error}")
            processed_audio_bytes = audio_bytes
            # Try to guess format from filename or default to webm
            if audio.filename and '.' in audio.filename:
                audio_format = audio.filename.split('.')[-1].lower()
            else:
                audio_format = "webm"
        
        # Create a temporary file for Groq API
        with tempfile.NamedTemporaryFile(suffix=f'.{audio_format}', delete=False) as temp_file:
            temp_file.write(processed_audio_bytes)
            temp_file_path = temp_file.name
        
        try:
            # Transcribe using Groq Whisper API
            logger.info("Starting Groq Whisper transcription...")
            
            with open(temp_file_path, "rb") as file:
                transcription_response = groq_client.audio.transcriptions.create(
                    file=(audio.filename or f"audio.{audio_format}", file.read()),
                    model="whisper-large-v3",
                    response_format="verbose_json",
                    temperature=0.0,  # For more consistent results
                )
            
            transcription_text = transcription_response.text.strip() if transcription_response.text else ""
            
            # Filter out common Whisper artifacts
            artifacts = [
                "[Music]", "[Applause]", "[Laughter]", "MBC 뉴스", 
                "ご視聴ありがとうございました", "(Music)", "(Applause)",
                "♪ Music ♪", "♪♪♪", "Thanks for watching!", "Thank you for watching!"
            ]
            
            for artifact in artifacts:
                transcription_text = transcription_text.replace(artifact, "").strip()
            
            # Remove multiple spaces
            transcription_text = " ".join(transcription_text.split())
            
            logger.info(f"Transcription completed: '{transcription_text[:100]}{'...' if len(transcription_text) > 100 else ''}'")
            
            return {
                "text": transcription_text,
                "duration": getattr(transcription_response, 'duration', None),
                "language": getattr(transcription_response, 'language', None)
            }
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temporary file: {cleanup_error}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        logger.error(traceback.format_exc())
        
        # Provide more helpful error messages
        error_detail = str(e)
        if "rate_limit" in error_detail.lower():
            error_detail = "API rate limit exceeded. Please try again in a moment."
        elif "file_size" in error_detail.lower():
            error_detail = "Audio file is too large. Maximum size is 19.5MB."
        elif "invalid_request_error" in error_detail.lower():
            error_detail = "Invalid audio format or corrupted file."
        else:
            error_detail = f"Transcription failed: {error_detail}"
            
        raise HTTPException(status_code=500, detail=error_detail)

@app.post("/summarize")
async def summarize(request: TextRequest):
    if groq_client is None:
        raise HTTPException(
            status_code=503,
            detail="Groq client not available. Please check GROQ_API_KEY environment variable."
        )
    
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="No text provided for summarization")
    
    try:
        logger.info(f"Summarizing text of length: {len(request.text)}")
        
        # Truncate text if too long (Groq has token limits)
        max_chars = 8000  # Conservative limit for llama models
        text_to_summarize = request.text[:max_chars]
        if len(request.text) > max_chars:
            text_to_summarize += "... [text truncated]"
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that creates concise, actionable summaries of meeting transcripts. Focus on key decisions, action items, and important discussion points. Structure your summary with clear sections when appropriate."
            },
            {
                "role": "user",
                "content": f"Summarize this meeting transcript in 2-3 paragraphs, highlighting key decisions and action items:\n\n{text_to_summarize}"
            }
        ]
        
        response = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.1-70b-versatile",
            max_tokens=400,
            temperature=0.3,
            top_p=0.9
        )
        
        summary = response.choices[0].message.content.strip()
        logger.info("Summary generated successfully")
        
        return {"summary": summary}
        
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        logger.error(traceback.format_exc())
        
        error_detail = str(e)
        if "rate_limit" in error_detail.lower():
            error_detail = "API rate limit exceeded. Please try again in a moment."
        else:
            error_detail = f"Summarization failed: {error_detail}"
            
        raise HTTPException(status_code=500, detail=error_detail)

@app.post("/suggest_response")
async def suggest_response(request: TextRequest):
    if groq_client is None:
        raise HTTPException(
            status_code=503,
            detail="Groq client not available. Please check GROQ_API_KEY environment variable."
        )
    
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="No text provided for response suggestion")
    
    try:
        logger.info(f"Generating response suggestion for text of length: {len(request.text)}")
        
        # Get last portion of transcript for context
        max_chars = 6000
        text_for_response = request.text[-max_chars:] if len(request.text) > max_chars else request.text
        
        messages = [
            {
                "role": "system",
                "content": "You are a professional meeting assistant. Analyze the transcript to identify the most recent question, request, or discussion point that requires a response. Provide a brief, professional, and contextually appropriate response suggestion that directly addresses the query."
            },
            {
                "role": "user",
                "content": f"Based on this meeting transcript, suggest a professional response to the most recent query or discussion point that needs addressing:\n\n{text_for_response}"
            }
        ]
        
        response = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.1-70b-versatile",
            max_tokens=250,
            temperature=0.4,
            top_p=0.9
        )
        
        suggestion = response.choices[0].message.content.strip()
        logger.info("Response suggestion generated successfully")
        
        return {"suggestion": suggestion}
        
    except Exception as e:
        logger.error(f"Response suggestion error: {e}")
        logger.error(traceback.format_exc())
        
        error_detail = str(e)
        if "rate_limit" in error_detail.lower():
            error_detail = "API rate limit exceeded. Please try again in a moment."
        else:
            error_detail = f"Response suggestion failed: {error_detail}"
            
        raise HTTPException(status_code=500, detail=error_detail)

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred. Please try again."}
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
