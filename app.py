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
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging for production
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment detection
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = ENVIRONMENT == "development"

app = FastAPI(
    title="Nova.AI Backend", 
    version="2.0.1",
    debug=DEBUG,
    docs_url="/docs" if DEBUG else None,  # Disable docs in production for security
    redoc_url="/redoc" if DEBUG else None
)

# Production-ready CORS middleware
allowed_origins = [
    "https://your-domain.com",  # Replace with your actual domain
    "https://www.your-domain.com",
    "http://localhost:3000",  # For development
    "http://127.0.0.1:3000",
]

if DEBUG:
    allowed_origins.append("*")  # Allow all in development

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if not DEBUG else ["*"],
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
    environment: str
    models_loaded: dict

# Rate limiting (simple in-memory) - Enhanced for production
request_timestamps = {}
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds
MAX_REQUESTS_PER_WINDOW = int(os.getenv("MAX_REQUESTS_PER_WINDOW", "30"))

def get_client_id(request: Request) -> str:
    """Get client identifier for rate limiting"""
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def check_rate_limit(client_id: str) -> bool:
    """Enhanced rate limiting with cleanup"""
    now = time.time()
    
    if client_id not in request_timestamps:
        request_timestamps[client_id] = []
    
    # Clean old timestamps
    request_timestamps[client_id] = [
        ts for ts in request_timestamps[client_id] 
        if now - ts < RATE_LIMIT_WINDOW
    ]
    
    # Check limit
    if len(request_timestamps[client_id]) >= MAX_REQUESTS_PER_WINDOW:
        return False
    
    # Add current request
    request_timestamps[client_id].append(now)
    return True

# Initialize Groq client with retry logic
async def initialize_groq_client():
    global groq_client
    
    try:
        from groq import Groq
        
        # Get API key from environment
        api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            logger.error("GROQ_API_KEY not found in environment variables")
            raise Exception("GROQ_API_KEY not found")
        
        groq_client = Groq(api_key=api_key)
        
        # Test the client with a simple request
        test_response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": "Test"}],
            model="llama-3.1-70b-versatile",
            max_tokens=5
        )
        
        logger.info("Groq client initialized and tested successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        groq_client = None
        raise e

def validate_audio_file(content_type: str, file_size: int) -> tuple[bool, str]:
    """Validate audio file type and size according to Groq API requirements"""
    supported_formats = [
        "audio/flac", "audio/mp3", "audio/mp4", "audio/mpeg", 
        "audio/mpga", "audio/m4a", "audio/ogg", "audio/wav", "audio/webm"
    ]
    
    supported_extensions = [
        "flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"
    ]
    
    # Check file size (25MB max - updated for better reliability)
    max_size = 25 * 1024 * 1024
    if file_size > max_size:
        return False, f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds maximum limit (25MB)"
    
    if file_size < 1024:  # Less than 1KB
        return False, "File too small (less than 1KB)"
    
    # Check content type
    if content_type in supported_formats:
        return True, "Valid format"
    
    for ext in supported_extensions:
        if ext in content_type.lower():
            return True, "Valid format"
    
    return False, f"Unsupported format: {content_type}. Supported formats: {', '.join(supported_extensions)}"

def convert_audio_if_needed(audio_bytes: bytes, content_type: str, filename: str) -> tuple[bytes, str]:
    """Convert audio to a supported format if needed using pydub"""
    try:
        from pydub import AudioSegment
        
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
        
        # Otherwise, convert to WAV
        logger.info(f"Converting audio from {content_type} to WAV")
        
        try:
            if format_hint:
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=format_hint)
            else:
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        except Exception as load_error:
            logger.warning(f"Pydub conversion failed: {load_error}, trying with webm format")
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
        
        # Convert to WAV with reasonable settings
        output_buffer = io.BytesIO()
        audio.export(
            output_buffer, 
            format="wav",
            parameters=["-ar", "16000", "-ac", "1"]  # 16kHz, mono for better transcription
        )
        converted_bytes = output_buffer.getvalue()
        
        logger.info(f"Successfully converted audio to WAV. Size: {len(converted_bytes)} bytes")
        return converted_bytes, "wav"
        
    except Exception as e:
        logger.error(f"Audio conversion failed: {e}")
        return audio_bytes, "webm"

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting Nova.AI Backend in {ENVIRONMENT} mode...")
    try:
        await initialize_groq_client()
        logger.info("Backend startup completed successfully")
    except Exception as e:
        logger.error(f"Backend startup failed: {e}")
        # Don't raise here to allow health checks to show the error

# Health check endpoint - Enhanced for production monitoring
@app.get("/health", response_model=HealthResponse)
async def health_check():
    groq_available = groq_client is not None
    
    return HealthResponse(
        status="healthy" if groq_available else "degraded",
        message="All services operational" if groq_available else "Groq API unavailable - check GROQ_API_KEY",
        environment=ENVIRONMENT,
        models_loaded={
            "groq_client": groq_available,
            "whisper_api": groq_available,
            "text_generation": groq_available,
            "asr_model": groq_available,
            "summarization": groq_available,
            "response_generation": groq_available
        }
    )

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Nova.AI Backend API", 
        "status": "running",
        "version": "2.0.1",
        "environment": ENVIRONMENT,
        "features": ["transcription", "summarization", "response_suggestions"],
        "health_endpoint": "/health"
    }

@app.post("/transcribe")
async def transcribe(request: Request, audio: UploadFile = File(...)):
    if groq_client is None:
        raise HTTPException(
            status_code=503,
            detail="Groq client not available. Please check GROQ_API_KEY environment variable."
        )
    
    # Rate limiting
    client_id = get_client_id(request)
    if not check_rate_limit(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again in a minute."
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
            logger.info(f"Audio processed successfully. Format: {audio_format}, Size: {len(processed_audio_bytes)} bytes")
        except Exception as conversion_error:
            logger.warning(f"Audio conversion failed, using original: {conversion_error}")
            processed_audio_bytes = audio_bytes
            if audio.filename and '.' in audio.filename:
                audio_format = audio.filename.split('.')[-1].lower()
            else:
                audio_format = "webm"
        
        # Create temporary file for Groq API
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
                    temperature=0.2,  # Slightly higher for natural speech
                    prompt="This is a meeting recording. Please transcribe accurately including all speakers."
                )
            
            transcription_text = transcription_response.text.strip() if transcription_response.text else ""
            
            # Clean up common artifacts but be less aggressive
            artifacts = [
                "[Music]", "[Applause]", "[Laughter]", 
                "(Music)", "(Applause)", "(Laughter)",
                "♪ Music ♪", "♪♪♪", 
                "Thanks for watching!", "Thank you for watching!",
                "MBC 뉴스", "ご視聴ありがとうございました"
            ]
            
            for artifact in artifacts:
                transcription_text = transcription_text.replace(artifact, "").strip()
            
            # Clean up multiple spaces and empty lines
            transcription_text = " ".join(transcription_text.split())
            
            # If text is very short, it might be silence or noise
            if len(transcription_text.strip()) < 3:
                transcription_text = ""
            
            logger.info(f"Transcription completed: '{transcription_text[:100]}{'...' if len(transcription_text) > 100 else ''}'")
            
            return {
                "text": transcription_text,
                "duration": getattr(transcription_response, 'duration', None),
                "language": getattr(transcription_response, 'language', None),
                "confidence": "high" if len(transcription_text) > 20 else "low"
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
            error_detail = "Audio file is too large. Maximum size is 25MB."
        elif "invalid_request_error" in error_detail.lower():
            error_detail = "Invalid audio format or corrupted file."
        elif "timeout" in error_detail.lower():
            error_detail = "Request timeout. Please try with a shorter audio file."
        else:
            error_detail = f"Transcription failed: {error_detail}"
            
        raise HTTPException(status_code=500, detail=error_detail)

@app.post("/summarize")
async def summarize(request: Request, text_request: TextRequest):
    if groq_client is None:
        raise HTTPException(
            status_code=503,
            detail="Groq client not available. Please check GROQ_API_KEY environment variable."
        )
    
    # Rate limiting
    client_id = get_client_id(request)
    if not check_rate_limit(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again in a minute."
        )
    
    if not text_request.text or not text_request.text.strip():
        raise HTTPException(status_code=400, detail="No text provided for summarization")
    
    try:
        logger.info(f"Summarizing text of length: {len(text_request.text)}")
        
        # Handle very short text
        if len(text_request.text.strip()) < 50:
            return {"summary": "Text too short to summarize meaningfully."}
        
        # Truncate text if too long (conservative limit for token constraints)
        max_chars = 12000  # Increased limit
        text_to_summarize = text_request.text[:max_chars]
        if len(text_request.text) > max_chars:
            text_to_summarize += "... [text truncated for processing]"
        
        # Enhanced prompt for better summaries
        messages = [
            {
                "role": "system",
                "content": """You are an expert meeting summarizer. Create concise, actionable summaries that capture:
1. Key decisions made
2. Action items and who owns them
3. Important discussion points
4. Next steps
5. Unresolved issues

Structure your summary with clear sections. Be specific about outcomes and avoid generic statements."""
            },
            {
                "role": "user",
                "content": f"""Summarize this meeting transcript. Focus on actionable outcomes and key decisions:

{text_to_summarize}

Provide a structured summary in 2-4 paragraphs highlighting the most important points."""
            }
        ]
        
        response = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.1-70b-versatile",
            max_tokens=500,  # Increased for more detailed summaries
            temperature=0.3,
            top_p=0.9
        )
        
        summary = response.choices[0].message.content.strip()
        
        # Validate response
        if not summary or len(summary) < 20:
            summary = "Unable to generate meaningful summary from the provided text."
        
        logger.info("Summary generated successfully")
        
        return {
            "summary": summary,
            "word_count": len(text_to_summarize.split()),
            "original_length": len(text_request.text)
        }
        
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        logger.error(traceback.format_exc())
        
        error_detail = str(e)
        if "rate_limit" in error_detail.lower():
            error_detail = "API rate limit exceeded. Please try again in a moment."
        elif "context_length" in error_detail.lower():
            error_detail = "Text too long for processing. Please try with shorter text."
        else:
            error_detail = f"Summarization failed: {error_detail}"
            
        raise HTTPException(status_code=500, detail=error_detail)

@app.post("/suggest_response")
async def suggest_response(request: Request, text_request: TextRequest):
    if groq_client is None:
        raise HTTPException(
            status_code=503,
            detail="Groq client not available. Please check GROQ_API_KEY environment variable."
        )
    
    # Rate limiting
    client_id = get_client_id(request)
    if not check_rate_limit(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again in a minute."
        )
    
    if not text_request.text or not text_request.text.strip():
        raise HTTPException(status_code=400, detail="No text provided for response suggestion")
    
    try:
        logger.info(f"Generating response suggestion for text of length: {len(text_request.text)}")
        
        # Handle very short text
        if len(text_request.text.strip()) < 30:
            return {"suggestion": "Need more context to suggest an appropriate response."}
        
        # Focus on the most recent portion of the transcript
        max_chars = 8000  # Focus on recent context
        text_for_response = text_request.text[-max_chars:] if len(text_request.text) > max_chars else text_request.text
        
        # Enhanced prompt for better response suggestions
        messages = [
            {
                "role": "system",
                "content": """You are a professional meeting assistant. Analyze the conversation to:
1. Identify the most recent question, request, or discussion point that needs a response
2. Understand the context and tone of the meeting
3. Provide a brief, professional, and contextually appropriate response

Your suggestions should be:
- Professional and diplomatic
- Directly address the most recent query or discussion point
- Brief but substantive (2-3 sentences max)
- Actionable when appropriate"""
            },
            {
                "role": "user",
                "content": f"""Based on this meeting transcript, suggest a professional response to the most recent question or discussion point that requires input:

{text_for_response}

Provide a concise, professional response suggestion that directly addresses what was just discussed."""
            }
        ]
        
        response = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.1-70b-versatile",
            max_tokens=200,  # Keep responses concise
            temperature=0.4,
            top_p=0.9
        )
        
        suggestion = response.choices[0].message.content.strip()
        
        # Validate response
        if not suggestion or len(suggestion) < 10:
            suggestion = "Unable to identify a clear question or discussion point that requires a response."
        
        # Clean up any formatting artifacts
        suggestion = suggestion.replace('"', '').strip()
        
        logger.info("Response suggestion generated successfully")
        
        return {
            "suggestion": suggestion,
            "context_length": len(text_for_response),
            "confidence": "high" if len(text_request.text) > 200 else "medium"
        }
        
    except Exception as e:
        logger.error(f"Response suggestion error: {e}")
        logger.error(traceback.format_exc())
        
        error_detail = str(e)
        if "rate_limit" in error_detail.lower():
            error_detail = "API rate limit exceeded. Please try again in a moment."
        elif "context_length" in error_detail.lower():
            error_detail = "Text too long for processing. Please try with shorter text."
        else:
            error_detail = f"Response suggestion failed: {error_detail}"
            
        raise HTTPException(status_code=500, detail=error_detail)

# Additional utility endpoint for testing
@app.post("/test_groq")
async def test_groq_connection():
    """Test endpoint to verify Groq API connectivity"""
    if groq_client is None:
        raise HTTPException(status_code=503, detail="Groq client not initialized")
    
    try:
        test_response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": "Say 'API test successful'"}],
            model="llama-3.1-70b-versatile",
            max_tokens=10,
            temperature=0
        )
        
        return {
            "status": "success",
            "response": test_response.choices[0].message.content,
            "model": "llama-3.1-70b-versatile"
        }
    except Exception as e:
        logger.error(f"Groq test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Groq API test failed: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "type": "http_error"}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error occurred. Please try again.",
            "type": "internal_error"
        }
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=DEBUG)
