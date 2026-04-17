# Full Stack Setup (React + Node + FastAPI)

This repo now contains a complete interactive UI + backend + ML API flow:

**React UI (`Front/`) → Node Backend (`node-backend/`) → FastAPI ML Service (`ml-service/`)**

## 1) Start the FastAPI ML service

```powershell
cd ml-service
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Notes:
- Model file default: `ml-service/models/voice_model.pkl`
- Override with: `VOICESHIELD_MODEL_PATH=...`
- File size limit (MB): `VOICESHIELD_MAX_FILE_SIZE_MB` (default 50)

## 2) Start the Node backend (history + proxy)

```powershell
cd node-backend
npm install
Copy-Item .env.example .env
npm run dev
```

Default URLs:
- Node API: `http://localhost:5001`
- ML API: `http://localhost:8000`

## 3) Start the React frontend

```powershell
cd Front
npm install
Copy-Item .env.example .env
npm run dev
```

Open:
- `http://localhost:5173`

## Endpoints

- React → Node
  - `POST /api/analyze` (multipart field: `file`)
  - `GET /api/history`
  - `DELETE /api/history`
  - `GET /api/history/:id/audio`
- Node → FastAPI
  - `POST http://localhost:8000/predict`

