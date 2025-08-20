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
import numpy as np

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

# Global variables to store loaded models
asr_pipe = None
groq_client = None

# Request models
class TextRequest(BaseModel):
    text: str

class HealthResponse(BaseModel):
    status: str
    message: str
    models_loaded: dict

# Initialize models with proper error handling
async def initialize_models():
    global asr_pipe, groq_client
    
    try:
        logger.info("Initializing ASR model...")
        from transformers import pipeline
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load Whisper model with error handling
        try:
            asr_pipe = pipeline(
                "automatic-speech-recognition", 
                model="openai/whisper-large-v3-turbo", 
                device=device,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            logger.info("ASR model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ASR model: {e}")
            # Fallback to smaller model
            try:
                asr_pipe = pipeline(
                    "automatic-speech-recognition", 
                    model="openai/whisper-base", 
                    device=device
                )
                logger.info("Fallback ASR model loaded")
            except Exception as e2:
                logger.error(f"Failed to load fallback ASR model: {e2}")
                asr_pipe = None

        # Initialize Groq client
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
                logger.warning("GROQ_API_KEY not found, text generation features disabled")
                groq_client = None
                
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            groq_client = None
            
    except Exception as e:
        logger.error(f"Error during model initialization: {e}")
        logger.error(traceback.format_exc())

def process_audio_with_ffmpeg(audio_bytes: bytes, input_format: str = "webm") -> tuple:
    """
    Process audio using FFmpeg via pydub (most reliable for WebM)
    """
    try:
        from pydub import AudioSegment
        
        # Try different approaches for problematic chunks
        audio_segment = None
        
        # Method 1: Try direct format processing
        try:
            audio_segment = AudioSegment.from_file(
                io.BytesIO(audio_bytes), 
                format=input_format
            )
        except Exception as e1:
            logger.warning(f"Direct format processing failed: {e1}")
            
            # Method 2: Try without format specification (auto-detect)
            try:
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            except Exception as e2:
                logger.warning(f"Auto-detect failed: {e2}")
                
                # Method 3: Try as raw audio if it's a chunk continuation
                try:
                    # This is likely a raw audio chunk, try to process as raw PCM
                    # Assume 48kHz, 16-bit, mono (common WebM parameters)
                    audio_segment = AudioSegment(
                        audio_bytes,
                        frame_rate=48000,
                        sample_width=2,
                        channels=1
                    )
                    logger.info("Processed as raw PCM audio")
                except Exception as e3:
                    logger.error(f"Raw PCM processing failed: {e3}")
                    raise e1  # Re-raise the original error
        
        if audio_segment is None:
            raise Exception("All processing methods failed")
        
        # Convert to mono if stereo
        if audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)
        
        # Get sample rate
        samplerate = audio_segment.frame_rate
        
        # Convert to numpy array
        audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        
        # Normalize based on original bit depth
        if audio_segment.sample_width == 2:  # 16-bit
            audio_data = audio_data / 32768.0
        elif audio_segment.sample_width == 3:  # 24-bit
            audio_data = audio_data / 8388608.0
        elif audio_segment.sample_width == 4:  # 32-bit
            audio_data = audio_data / 2147483648.0
        else:  # Assume already normalized
            max_val = max(abs(audio_data.max()) if len(audio_data) > 0 else 1.0, 
                         abs(audio_data.min()) if len(audio_data) > 0 else 1.0, 
                         1.0)
            audio_data = audio_data / max_val
        
        logger.info(f"FFmpeg processed: shape={audio_data.shape}, sr={samplerate}, channels={audio_segment.channels}")
        return audio_data, samplerate
        
    except Exception as e:
        logger.error(f"FFmpeg processing failed: {e}")
        raise

def process_audio_with_temp_file(audio_bytes: bytes) -> tuple:
    """
    Process audio by writing to temp file (fallback method)
    """
    try:
        # Write to temporary file
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        try:
            # Try with librosa
            import librosa
            audio_data, samplerate = librosa.load(tmp_file_path, sr=None)
            logger.info(f"Librosa processed: shape={audio_data.shape}, sr={samplerate}")
            return audio_data, samplerate
            
        except Exception as e1:
            logger.warning(f"Librosa failed: {e1}")
            
            # Try with soundfile
            try:
                import soundfile as sf
                audio_data, samplerate = sf.read(tmp_file_path)
                logger.info(f"Soundfile processed: shape={audio_data.shape}, sr={samplerate}")
                return audio_data, samplerate
            except Exception as e2:
                logger.error(f"Soundfile also failed: {e2}")
                raise e2
                
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_file_path)
        except:
            pass

def handle_audio_chunk(audio_bytes: bytes, filename: str = None) -> tuple:
    """
    Handle audio chunks that might be incomplete WebM containers
    """
    try:
        # First, try standard processing
        return process_audio_with_ffmpeg(audio_bytes, "webm")
        
    except Exception as primary_error:
        logger.warning(f"Standard processing failed: {primary_error}")
        
        # Check if this might be a raw audio chunk
        if len(audio_bytes) > 0:
            try:
                # Try to process as raw Opus data (common in WebM chunks)
                # We'll use a more robust approach with temp files
                with tempfile.NamedTemporaryFile(suffix='.opus', delete=False) as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_file_path = tmp_file.name

                try:
                    import librosa
                    audio_data, samplerate = librosa.load(tmp_file_path, sr=None)

                    if len(audio_data) > 0:
                        logger.info(f"Raw Opus processed: shape={audio_data.shape}, sr={samplerate}")
                        return audio_data, samplerate

                except Exception as opus_error:
                    logger.warning(f"Opus processing failed: {opus_error}")

                    # Try as raw PCM with common WebM settings
                    try:
                        # Assume 16-bit PCM at 48kHz (common for WebM/Opus)
                        samples = np.frombuffer(audio_bytes, dtype=np.int16)
                        audio_data = samples.astype(np.float32) / 32768.0
                        samplerate = 48000

                        if len(audio_data) > 100:  # At least some samples
                            logger.info(f"Raw PCM processed: shape={audio_data.shape}, assumed sr={samplerate}")
                            return audio_data, samplerate

                    except Exception as pcm_error:
                        logger.warning(f"Raw PCM processing failed: {pcm_error}")

                finally:
                    # Clean up temp file
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
            except Exception:
                pass
                        
        # If all methods fail, re-raise the original error
        raise primary_error

def validate_audio_data(audio_data: np.ndarray, samplerate: int) -> bool:
    """
    Validate processed audio data
    """
    if audio_data is None or len(audio_data) == 0:
        logger.warning("Audio data is empty")
        return False
    
    # Check for silent audio
    if np.abs(audio_data).max() < 1e-6:
        logger.warning("Audio appears to be silent")
        return False
    
    # Check minimum length (100ms)
    min_samples = int(samplerate * 0.1)
    if len(audio_data) < min_samples:
        logger.warning(f"Audio too short: {len(audio_data)} samples < {min_samples} minimum")
        return False
    
    # Check for reasonable sample rate
    if samplerate < 8000 or samplerate > 96000:
        logger.warning(f"Unusual sample rate: {samplerate}")
        return False
    
    return True

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Nova.AI Backend...")
    await initialize_models()
    logger.info("Backend startup completed")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        message="Nova.AI Backend is running",
        models_loaded={
            "asr_model": asr_pipe is not None,
            "groq_client": groq_client is not None
        }
    )

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Nova.AI Backend API", "status": "running"}

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if asr_pipe is None:
        raise HTTPException(
            status_code=503, 
            detail="ASR model not available. Please check server logs."
        )
    
    try:
        logger.info(f"Processing audio file: {audio.filename}, type: {audio.content_type}")
        
        # Read audio data first
        audio_bytes = await audio.read()
        logger.info(f"Read {len(audio_bytes)} bytes from audio file")
        
        # Validate size and emptiness after reading
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        if len(audio_bytes) > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=413, detail="Audio file too large (max 50MB)")
        
        audio_data = None
        samplerate = None
        last_error = None
        
        # Method 1: Try with enhanced chunk handling
        try:
            logger.info("Trying enhanced chunk handling...")
            audio_data, samplerate = handle_audio_chunk(audio_bytes, audio.filename)
        except Exception as e:
            logger.warning(f"FFmpeg method failed: {e}")
            last_error = e
            
            # Method 2: Try temp file approach
            try:
                logger.info("Trying temp file method...")
                audio_data, samplerate = process_audio_with_temp_file(audio_bytes)
            except Exception as e2:
                logger.warning(f"Temp file method failed: {e2}")
                last_error = e2
                
                # Method 3: Try direct memory processing with different formats
                try:
                    logger.info("Trying direct memory processing...")
                    from pydub import AudioSegment
                    
                    # Try different format hints
                    formats_to_try = ["webm", "ogg", "mp4", "wav"]
                    
                    for fmt in formats_to_try:
                        try:
                            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
                            
                            # Convert to mono
                            if audio_segment.channels > 1:
                                audio_segment = audio_segment.set_channels(1)
                            
                            samplerate = audio_segment.frame_rate
                            audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                            
                            # Normalize
                            if audio_segment.sample_width == 2:
                                audio_data = audio_data / 32768.0
                            else:
                                max_val = max(abs(audio_data.max()), abs(audio_data.min()), 1.0)
                                audio_data = audio_data / max_val
                            
                            logger.info(f"Success with format {fmt}")
                            break
                            
                        except Exception as fmt_error:
                            logger.debug(f"Format {fmt} failed: {fmt_error}")
                            continue
                    
                    if audio_data is None:
                        raise Exception("All format attempts failed")
                        
                except Exception as e3:
                    logger.error(f"All audio processing methods failed: {e3}")
                    # Provide more specific error message
                    error_msg = "Unable to process audio format. "
                    if "webm" in str(last_error).lower():
                        error_msg += "WebM format issue detected. Please ensure FFmpeg is installed with WebM support."
                    else:
                        error_msg += f"Last error: {str(last_error)}"
                    
                    raise HTTPException(status_code=400, detail=error_msg)
        
        # Validate processed audio
        if not validate_audio_data(audio_data, samplerate):
            return {"text": ""}
        
        # Handle stereo to mono conversion if needed
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Ensure float32 format
        audio_data = audio_data.astype(np.float32)
        
        # Prepare audio for Whisper
        audio_input = {
            "sampling_rate": samplerate,
            "raw": audio_data
        }
        
        # Transcribe with error handling
        logger.info("Starting transcription...")
        try:
            result = asr_pipe(audio_input)
            transcription = result.get("text", "").strip()
            
            # Filter out common Whisper artifacts
            artifacts = ["[Music]", "[Applause]", "[Laughter]", "MBC 뉴스", "ご視聴ありがとうございました"]
            for artifact in artifacts:
                transcription = transcription.replace(artifact, "").strip()
            
            logger.info(f"Transcription completed: '{transcription[:100]}{'...' if len(transcription) > 100 else ''}'")
            return {"text": transcription}
            
        except Exception as transcribe_error:
            logger.error(f"Whisper transcription failed: {transcribe_error}")
            # Try with different parameters
            try:
                logger.info("Retrying transcription with different parameters...")
                result = asr_pipe(audio_input, return_timestamps=False, chunk_length_s=30)
                transcription = result.get("text", "").strip()
                logger.info("Retry transcription successful")
                return {"text": transcription}
            except Exception as retry_error:
                logger.error(f"Retry transcription also failed: {retry_error}")
                raise HTTPException(status_code=500, detail=f"Transcription failed: {str(transcribe_error)}")
        
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
            detail="Text generation service not available. Please check GROQ_API_KEY."
        )
    
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="No text provided for summarization")
    
    try:
        logger.info(f"Summarizing text of length: {len(request.text)}")
        
        # Truncate text if too long (Groq has token limits)
        max_chars = 4000  # Conservative limit
        text_to_summarize = request.text[:max_chars]
        if len(request.text) > max_chars:
            text_to_summarize += "... [text truncated]"
        
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful assistant that creates concise, actionable summaries of meeting transcripts. Focus on key decisions, action items, and important discussion points."
            },
            {
                "role": "user", 
                "content": f"Summarize this meeting transcript in 2-3 paragraphs:\n\n{text_to_summarize}"
            }
        ]
        
        result = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.1-70b-versatile",  # More reliable model
            max_tokens=300,
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
            detail="Text generation service not available. Please check GROQ_API_KEY."
        )
    
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="No text provided for response suggestion")
    
    try:
        logger.info(f"Generating response suggestion for text of length: {len(request.text)}")
        
        # Get last portion of transcript for context
        max_chars = 3000
        text_for_response = request.text[-max_chars:] if len(request.text) > max_chars else request.text
        
        messages = [
            {
                "role": "system", 
                "content": "You are a professional meeting assistant. Analyze the transcript to identify the most recent question, request, or discussion point that requires a response. Provide a brief, professional, and contextually appropriate response suggestion."
            },
            {
                "role": "user", 
                "content": f"Based on this meeting transcript, suggest a professional response to the most recent query or discussion point:\n\n{text_for_response}"
            }
        ]
        
        result = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.1-70b-versatile",
            max_tokens=200,
            temperature=0.4
        )
        
        suggestion = result.choices[0].message.content.strip()
        logger.info("Response suggestion generated successfully")
        
        return {"suggestion": suggestion}
        
    except Exception as e:
        logger.error(f"Response suggestion error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Response suggestion failed: {str(e)}")

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