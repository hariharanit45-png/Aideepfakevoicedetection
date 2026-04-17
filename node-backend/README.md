# Node Backend (Express)

This service is the frontend-facing backend:

- Accepts audio uploads from the React UI
- Proxies the audio to the FastAPI ML service (`ml-service/app.py`)
- Stores a simple analysis history (JSON) and serves uploaded audio back for playback

## Run

```powershell
cd node-backend
npm install
copy .env.example .env
npm run dev
```

Environment variables are documented in `.env.example`.

