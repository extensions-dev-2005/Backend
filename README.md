# Nova.AI Meeting Transcriber

A Chrome extension that provides real-time meeting transcription, summarization, and response suggestions.

## 🚀 Features

- **Live Transcription**: Real-time transcription of both your voice and other participants
- **Non-Disruptive**: Captures audio without muting your meeting
- **AI Summarization**: Generate concise meeting summaries with key decisions and action items
- **Response Suggestions**: Get AI-powered suggestions for meeting responses
- **Multiple Audio Sources**: Captures both tab audio (other participants) and microphone (your voice)

## 🛠 Setup Instructions

### Backend Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**
   ```bash
   export GROQ_API_KEY="your_groq_api_key_here"
   ```
   
   Or create a `.env` file:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Run the Backend**
   ```bash
   python app.py
   ```
   
   Or with uvicorn:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Test Backend Health**
   Visit `http://localhost:8000/health` to verify the backend is running correctly.

### Chrome Extension Setup

1. **Load Extension in Developer Mode**
   - Open Chrome and go to `chrome://extensions/`
   - Enable "Developer mode" (top right toggle)
   - Click "Load unpacked" and select your project folder

2. **Update Backend URL**
   - If using a different backend URL, update the `backendUrl` variable in `popup.js`

3. **Grant Permissions**
   - The extension will request microphone and tab capture permissions
   - Accept these for full functionality

## 🎯 How to Use

### Starting Transcription

1. **Navigate to a Meeting**
   - Open your meeting platform (Google Meet, Zoom, Teams, etc.)
   - Join your meeting as usual

2. **Start Nova.AI**
   - Click the Nova.AI extension icon
   - Click "Start" to begin transcription
   - **The meeting audio will continue normally** - you won't be muted

3. **Monitor Progress**
   - View live transcription in the extension popup
   - Word count and connection status are displayed
   - Transcription happens in real-time with ~10-second chunks

### Using AI Features

1. **Generate Summary**
   - After capturing sufficient audio, click "Summarize"
   - Get structured summaries with key decisions and action items

2. **Get Response Suggestions**
   - Click "Suggest" to get AI-powered response recommendations
   - Suggestions are based on the most recent conversation context

3. **Stop Transcription**
   - Click "Stop" when you're done
   - All transcribed text remains available for summary/suggestions

## ⚙️ Technical Architecture

### Audio Capture Strategy
```
Meeting Audio → Audio Context → Channel Splitter
                    ↓               ↓
            Recording Stream    Playback Stream
                    ↓               ↓
            Nova.AI Backend    User's Speakers
```

This approach ensures:
- ✅ User continues hearing meeting audio
- ✅ Nova.AI captures audio for transcription  
- ✅ Both microphone and tab audio are captured
- ✅ No disruption to the meeting experience

### Backend Processing
1. **Audio Reception**: Receives audio chunks from extension
2. **Format Conversion**: Converts to Whisper-compatible format
3. **Transcription**: Uses Groq's Whisper API for speech-to-text
4. **AI Processing**: Leverages Llama models for summaries and suggestions

## 🔧 Troubleshooting

### Common Issues

#### "Backend unavailable" Error
- Check that your backend is running on the correct port
- Verify GROQ_API_KEY environment variable is set
- Test with `curl http://localhost:8000/health`

#### Permission Denied Errors
- Grant microphone permissions to Chrome
- Allow tab capture permissions for the extension
- Reload the extension if permissions were recently granted

#### Audio Not Being Transcribed
- Check browser console for error messages
- Ensure you're on a supported website (not chrome:// pages)
- Verify backend logs for transcription errors

#### Summarization/Suggestions Not Working
- Ensure GROQ_API_KEY is valid and has credits
- Check that transcription text exists before requesting summaries
- Monitor backend logs for API errors

### Debug Mode

Enable detailed logging:
```javascript
// In popup.js, set this flag
const DEBUG_MODE = true;
```

### Backend Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

Expected response:
```json
{
  "status": "healthy",
  "message": "All services operational",
  "models_loaded": {
    "groq_client": true,
    "whisper_api": true,
    "text_generation": true
  }
}
```

## 🔒 Privacy & Security

- **No Data Storage**: Transcriptions are processed in real-time and not permanently stored
- **Local Processing**: Audio processing happens locally in your browser
- **API Communication**: Only text (not audio) is sent to AI APIs for summaries/suggestions
- **Secure Connections**: All API communications use HTTPS

## 📝 API Endpoints

- `GET /health` - Backend health check
- `POST /transcribe` - Audio transcription
- `POST /summarize` - Generate meeting summary  
- `POST /suggest_response` - Generate response suggestions
- `POST /test_groq` - Test Groq API connectivity

## 🚧 Known Limitations

1. **Chrome Only**: Currently supports Chrome browser exclusively
2. **Internet Required**: Requires internet connection for AI features
3. **Groq API Dependency**: Requires valid Groq API key and credits
4. **Tab Capture Limitations**: Some websites may block tab capture
5. **Audio Format Support**: Best results with standard meeting platforms

## 🛡 Error Handling

The application includes comprehensive error handling:
- **Rate Limiting**: Prevents API abuse
- **Retry Logic**: Automatic retries for failed requests
- **Graceful Degradation**: Core features work even if some fail
- **User Feedback**: Clear error messages and status updates

## 📈 Performance Tips

1. **Stable Internet**: Ensure good internet connection for real-time processing
2. **Close Unused Tabs**: Reduces browser resource usage
3. **Regular Restarts**: Restart extension if experiencing issues
4. **Monitor API Usage**: Keep track of Groq API credits

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and support:
1. Check this README troubleshooting section
2. Review browser console logs
3. Check backend server logs
4. Create an issue on GitHub with detailed logs

---

**Version**: 2.0  
**Last Updated**: December 2024
