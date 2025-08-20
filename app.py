from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import traceback
import logging
import os
import io
import asyncio
import tempfile
from typing import Optional

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
        logger.info("Initializing Groq client...")
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
            logger.error("GROQ_API_KEY not found! Please set the environment variable.")
            groq_client = None
            
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        logger.error(traceback.format_exc())
        groq_client = None

def prepare_audio_for_groq(audio_bytes: bytes, filename: str = None) -> tuple:
    """
    Prepare audio data for Groq API - returns tuple format (filename, bytes)
    """
    try:
        # Groq expects a tuple of (filename, file_content)
        # Extract file extension from original filename or default to webm
        if filename:
            # Keep original extension
            file_extension = os.path.splitext(filename)[1] or ".webm"
            clean_filename = f"audio{file_extension}"
        else:
            clean_filename = "audio.webm"
        
        logger.info(f"Prepared audio tuple: {len(audio_bytes)} bytes, filename: {clean_filename}")
        return (clean_filename, audio_bytes)
        
    except Exception as e:
        logger.error(f"Error preparing audio for Groq: {e}")
        raise

def convert_audio_format(audio_bytes: bytes, target_format: str = "mp3") -> tuple:
    """
    Convert audio to a more compatible format using pydub if the original format fails
    """
    try:
        from pydub import AudioSegment
        
        logger.info(f"Converting audio to {target_format} format...")
        
        # Try to load the audio with pydub
        audio_segment = None
        
        # Try different format hints
        formats_to_try = ["webm", "ogg", "mp4", "wav", None]  # None = auto-detect
        
        for fmt in formats_to_try:
            try:
                if fmt is None:
                    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
                else:
                    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
                logger.info(f"Successfully loaded audio with format: {fmt or 'auto-detect'}")
                break
            except Exception as fmt_error:
                logger.debug(f"Format {fmt} failed: {fmt_error}")
                continue
        
        if audio_segment is None:
            raise Exception("Could not load audio with any format")
        
        # Export to target format
        output_buffer = io.BytesIO()
        audio_segment.export(output_buffer, format=target_format)
        converted_bytes = output_buffer.getvalue()
        filename = f"converted_audio.{target_format}"
        
        logger.info(f"Audio converted to {target_format}: {len(converted_bytes)} bytes")
        return (filename, converted_bytes)
        
    except Exception as e:
        logger.error(f"Audio conversion failed: {e}")
        raise

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Nova.AI Backend...")
    await initialize_groq_client()
    logger.info("Backend startup completed")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        message="Nova.AI Backend is running",
        models_loaded={
            "groq_client": groq_client is not None,
            "whisper_via_groq": groq_client is not None
        }
    )

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Nova.AI Backend API", "status": "running"}

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if groq_client is None:
        raise HTTPException(
            status_code=503, 
            detail="Groq client not available. Please check GROQ_API_KEY."
        )
    
    try:
        logger.info(f"Processing audio file: {audio.filename}, type: {audio.content_type}")
        
        # Read audio data
        audio_bytes = await audio.read()
        logger.info(f"Read {len(audio_bytes)} bytes from audio file")
        
        # Validate audio data
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        if len(audio_bytes) > 25 * 1024 * 1024:  # 25MB limit (Groq's limit)
            raise HTTPException(status_code=413, detail="Audio file too large (max 25MB)")
        
        # Prepare audio for Groq
        audio_file = None
        
        try:
            # First, try to use the audio as-is
            logger.info("Attempting to use original audio format...")
            audio_file = prepare_audio_for_groq(audio_bytes, audio.filename)
            
            # Test transcription with original format
            logger.info("Starting transcription with Groq Whisper...")
            result = groq_client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                response_format="verbose_json"  # Get detailed response
            )
            
            transcription_text = result.text if hasattr(result, 'text') else str(result)
            
        except Exception as original_error:
            logger.warning(f"Original format failed: {original_error}")
            
            # Try converting to MP3 format
            try:
                logger.info("Converting audio to MP3 format...")
                audio_file = convert_audio_format(audio_bytes, "mp3")
                
                logger.info("Retrying transcription with converted audio...")
                result = groq_client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3",
                    response_format="verbose_json"
                )
                
                transcription_text = result.text if hasattr(result, 'text') else str(result)
                
            except Exception as conversion_error:
                logger.warning(f"MP3 conversion failed: {conversion_error}")
                
                # Try WAV format as final fallback
                try:
                    logger.info("Converting audio to WAV format...")
                    audio_file = convert_audio_format(audio_bytes, "wav")
                    
                    logger.info("Final attempt with WAV format...")
                    result = groq_client.audio.transcriptions.create(
                        file=audio_file,
                        model="whisper-large-v3",
                        response_format="text"  # Fallback to simple text
                    )
                    
                    transcription_text = result if isinstance(result, str) else result.text
                    
                except Exception as wav_error:
                    logger.error(f"All audio formats failed. WAV error: {wav_error}")
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Could not process audio format. Original error: {str(original_error)}"
                    )
        
        # Clean up transcription text
        transcription_text = transcription_text.strip() if transcription_text else ""
        
        # Filter out common Whisper artifacts and noise
        artifacts = [
            "[Music]", "[Applause]", "[Laughter]", "[Noise]", 
            "MBC 뉴스", "ご視聴ありがとうございました", "Thank you for watching",
            "(music)", "(applause)", "(laughter)", "(noise)"
        ]
        
        for artifact in artifacts:
            transcription_text = transcription_text.replace(artifact, "").strip()
        
        # Remove extra whitespace
        transcription_text = " ".join(transcription_text.split())
        
        logger.info(f"Transcription completed successfully: '{transcription_text[:100]}{'...' if len(transcription_text) > 100 else ''}'")
        
        return {"text": transcription_text}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected transcription error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/summarize")
async def summarize(request: TextRequest):
    if groq_client is None:
        raise HTTPException(
            status_code=503, 
            detail="Groq client not available. Please check GROQ_API_KEY."
        )
    
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="No text provided for summarization")
    
    try:
        logger.info(f"Summarizing text of length: {len(request.text)}")
        
        # Truncate text if too long (Groq has token limits)
        max_chars = 6000  # More generous limit for Groq
        text_to_summarize = request.text[:max_chars]
        if len(request.text) > max_chars:
            text_to_summarize += "... [text truncated]"
        
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful assistant that creates concise, actionable summaries of meeting transcripts. Focus on key decisions, action items, and important discussion points. Provide a well-structured summary with clear bullet points when appropriate."
            },
            {
                "role": "user", 
                "content": f"Please summarize this meeting transcript, highlighting key decisions, action items, and important discussion points:\n\n{text_to_summarize}"
            }
        ]
        
        result = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.1-70b-versatile",
            max_tokens=400,
            temperature=0.3
        )
        
        summary = result.choices[0].message.content.strip()
        logger.info("Summary generated successfully")
        
        return {"summary": summary}
        
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@app.post("/suggest_response")
async def suggest_response(request: TextRequest):
    if groq_client is None:
        raise HTTPException(
            status_code=503, 
            detail="Groq client not available. Please check GROQ_API_KEY."
        )
    
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="No text provided for response suggestion")
    
    try:
        logger.info(f"Generating response suggestion for text of length: {len(request.text)}")
        
        # Get last portion of transcript for context
        max_chars = 4000
        text_for_response = request.text[-max_chars:] if len(request.text) > max_chars else request.text
        
        messages = [
            {
                "role": "system", 
                "content": "You are a professional meeting assistant. Analyze the transcript to identify the most recent question, request, or discussion point that requires a response. Provide a brief, professional, and contextually appropriate response suggestion. If no clear question is present, suggest a relevant follow-up or clarifying question."
            },
            {
                "role": "user", 
                "content": f"Based on this meeting transcript, suggest a professional response to the most recent query or discussion point:\n\n{text_for_response}"
            }
        ]
        
        result = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.1-70b-versatile",
            max_tokens=250,
            temperature=0.4
        )
        
        suggestion = result.choices[0].message.content.strip()
        logger.info("Response suggestion generated successfully")
        
        return {"suggestion": suggestion}
        
    except Exception as e:
        logger.error(f"Response suggestion error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Response suggestion failed: {str(e)}")

@app.post("/transcribe_with_options")
async def transcribe_with_options(
    audio: UploadFile = File(...),
    language: str = None,
    temperature: float = 0.0
):
    """
    Enhanced transcription endpoint with additional options
    """
    if groq_client is None:
        raise HTTPException(
            status_code=503, 
            detail="Groq client not available. Please check GROQ_API_KEY."
        )
    
    try:
        logger.info(f"Processing audio with options - Language: {language}, Temperature: {temperature}")
        
        # Read and validate audio
        audio_bytes = await audio.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        if len(audio_bytes) > 25 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Audio file too large (max 25MB)")
        
        # Prepare transcription parameters
        transcription_params = {
            "model": "whisper-large-v3",
            "response_format": "json",  # Get more detailed response
            "temperature": max(0.0, min(1.0, temperature))  # Ensure valid range
        }
        
        if language:
            transcription_params["language"] = language
        
        # Prepare audio
        audio_file = prepare_audio_for_groq(audio_bytes, audio.filename)
        transcription_params["file"] = audio_file
        
        try:
            # Attempt transcription
            result = groq_client.audio.transcriptions.create(**transcription_params)
            
            # Extract text and additional info
            if isinstance(result, str):
                transcription_text = result.strip()
                response_data = {"text": transcription_text}
            else:
                # Handle verbose_json response with additional metadata
                transcription_text = result.text.strip() if hasattr(result, 'text') else str(result).strip()
                response_data = {
                    "text": transcription_text,
                    "language": getattr(result, 'language', language),
                    "duration": getattr(result, 'duration', None),
                    "segments": getattr(result, 'segments', None)  # Word-level timestamps if available
                }
            
            # Clean up transcription
            artifacts = [
                "[Music]", "[Applause]", "[Laughter]", "[Noise]", 
                "(music)", "(applause)", "(laughter)", "(noise)"
            ]
            
            for artifact in artifacts:
                transcription_text = transcription_text.replace(artifact, "").strip()
            
            response_data["text"] = " ".join(transcription_text.split())
            
            logger.info("Enhanced transcription completed successfully")
            return response_data
            
        except Exception as e:
            # Fallback to format conversion
            logger.warning(f"Direct transcription failed, trying format conversion: {e}")
            
            audio_file = convert_audio_format(audio_bytes, "mp3")
            transcription_params["file"] = audio_file
            
            result = groq_client.audio.transcriptions.create(**transcription_params)
            transcription_text = result.text.strip() if hasattr(result, 'text') else str(result).strip()
            
            return {"text": " ".join(transcription_text.split())}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced transcription error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
