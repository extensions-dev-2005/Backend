# Nova.AI Backend Deployment on Render

1. Create a Render account at render.com.
2. Create a new Web Service.
3. Connect to a GitHub repo: push this backend directory to a new GitHub repo.
4. Select Python as runtime.
5. Set build command: `pip install -r requirements.txt`
6. Set start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
7. For large models, upgrade to a paid plan with GPU/ sufficient RAM.
8. Once deployed, copy the .onrender.com URL.
9. Update `backendUrl` in extension/popup.js with this URL.
10. Note: Models load on startup, which may take time/memory.

Test locally: `uvicorn app:app --reload`