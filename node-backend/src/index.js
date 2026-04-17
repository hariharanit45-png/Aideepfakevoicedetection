const path = require("path");
const fs = require("fs/promises");
const crypto = require("crypto");
const express = require("express");
const cors = require("cors");
const multer = require("multer");
const axios = require("axios");
const FormData = require("form-data");
require("dotenv").config();

const { createHistoryStore } = require("./historyStore");

const PORT = parseInt(process.env.PORT || "5001", 10);
const ML_API_URL = process.env.ML_API_URL || "http://localhost:8000";
const FRONTEND_ORIGIN = process.env.FRONTEND_ORIGIN || "http://localhost:5173";

const DATA_DIR = path.join(__dirname, "..", "data");
const UPLOADS_DIR = path.join(__dirname, "..", "uploads");

async function ensureDirs() {
  await fs.mkdir(DATA_DIR, { recursive: true });
  await fs.mkdir(UPLOADS_DIR, { recursive: true });
}

function safeExt(filename) {
  const ext = path.extname(filename || "").toLowerCase();
  if (!ext) return ".wav";
  if ([".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"].includes(ext)) return ext;
  return ".wav";
}

function createApp() {
  const app = express();

  const corsOptions =
    FRONTEND_ORIGIN === "*"
      ? { origin: true }
      : { origin: [FRONTEND_ORIGIN], credentials: true };

  app.use(cors(corsOptions));

  const upload = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: 50 * 1024 * 1024 },
  });

  const historyStore = createHistoryStore({ dataDir: DATA_DIR });

  app.get("/api/health", async (_req, res) => {
    let ml = { ok: false };
    try {
      const mlResp = await axios.get(`${ML_API_URL}/health`, {
        timeout: 5000,
        validateStatus: () => true,
      });
      ml = {
        ok: mlResp.status < 400,
        status: mlResp.status,
        data: mlResp.data,
      };
    } catch (e) {
      ml = { ok: false, error: e?.message || "ML health check failed" };
    }

    res.json({
      status: "ok",
      ml_api_url: ML_API_URL,
      ml,
    });
  });

  app.get("/api/history", async (_req, res) => {
    const items = await historyStore.list();
    res.json({ items });
  });

  app.delete("/api/history", async (_req, res) => {
    const items = await historyStore.list();
    await historyStore.clear();
    await Promise.all(
      items
        .map((x) => x.audioPath)
        .filter(Boolean)
        .map(async (p) => {
          try {
            await fs.unlink(p);
          } catch {
            // ignore
          }
        })
    );
    res.json({ success: true });
  });

  app.get("/api/history/:id/audio", async (req, res) => {
    const entry = await historyStore.get(req.params.id);
    if (!entry) return res.status(404).json({ error: "Not found" });
    if (!entry.audioPath) return res.status(404).json({ error: "Audio not stored" });
    return res.sendFile(entry.audioPath);
  });

  app.post("/api/analyze", upload.single("file"), async (req, res) => {
    try {
      if (!req.file) return res.status(400).json({ error: "No file uploaded" });

      const id = crypto.randomUUID();

      const ext = safeExt(req.file.originalname);
      const savedPath = path.join(UPLOADS_DIR, `${id}${ext}`);

      const form = new FormData();
      form.append("file", req.file.buffer, {
        filename: req.file.originalname || `audio${ext}`,
        contentType: req.file.mimetype || "application/octet-stream",
      });

      const mlResp = await axios.post(`${ML_API_URL}/predict`, form, {
        headers: form.getHeaders(),
        maxBodyLength: Infinity,
        maxContentLength: Infinity,
        validateStatus: () => true,
      });

      if (mlResp.status >= 400) {
        const detail =
          mlResp.data?.detail || mlResp.data?.message || mlResp.data?.error || "ML service error";
        return res.status(502).json({
          error: detail,
          ml_status: mlResp.status,
        });
      }

      const prediction = mlResp.data?.prediction;
      const confidence = mlResp.data?.confidence;
      if (!prediction || typeof confidence !== "number") {
        return res.status(502).json({
          error: "Invalid ML response",
          ml_response: mlResp.data,
        });
      }

      await fs.writeFile(savedPath, req.file.buffer);

      const entry = {
        id,
        fileName: req.file.originalname || path.basename(savedPath),
        uploadedAt: new Date().toISOString(),
        prediction,
        confidence,
        confidence0To1: mlResp.data?.confidence_0_to_1,
        modelType: mlResp.data?.model_type,
        rawScore: mlResp.data?.raw_score,
        audioPath: savedPath,
        audioUrl: `/api/history/${id}/audio`,
      };

      await historyStore.add(entry);

      res.json({ success: true, item: entry });
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error(err);
      res.status(500).json({ error: err?.message || "Server error" });
    }
  });

  return app;
}

async function main() {
  await ensureDirs();
  const app = createApp();
  app.listen(PORT, () => {
    // eslint-disable-next-line no-console
    console.log(`[node-backend] listening on http://localhost:${PORT}`);
    // eslint-disable-next-line no-console
    console.log(`[node-backend] ML_API_URL=${ML_API_URL}`);
  });
}

main().catch((err) => {
  // eslint-disable-next-line no-console
  console.error(err);
  process.exit(1);
});
