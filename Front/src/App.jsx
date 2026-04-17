import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Home,
  Info,
  Mic2,
  History as HistoryIcon,
  UploadCloud,
  Loader2,
  CheckCircle2,
  XCircle,
  Trash2,
  Volume2,
} from "lucide-react";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:5001";

function cx(...parts) {
  return parts.filter(Boolean).join(" ");
}

function formatDateTime(iso) {
  try {
    const d = new Date(iso);
    return d.toLocaleString();
  } catch {
    return iso || "—";
  }
}

function PredictionPill({ prediction, confidence }) {
  const isReal = prediction === "REAL";
  return (
    <div
      className={cx(
        "inline-flex items-center gap-2 rounded-full px-3 py-1 text-sm font-semibold",
        isReal
          ? "bg-emerald-50 text-emerald-800 border border-emerald-200"
          : "bg-rose-50 text-rose-800 border border-rose-200"
      )}
    >
      {isReal ? <CheckCircle2 size={16} /> : <XCircle size={16} />}
      <span>{prediction || "—"}</span>
      {typeof confidence === "number" && (
        <span className="font-medium opacity-80">{confidence}%</span>
      )}
    </div>
  );
}

function TopNav({ page, setPage }) {
  const nav = [
    { key: "home", label: "Home", icon: Home },
    { key: "audio", label: "Audio", icon: Mic2 },
    { key: "history", label: "History", icon: HistoryIcon },
  ];

  return (
    <header className="sticky top-0 z-10 border-b border-[var(--border)] bg-[rgba(234,246,234,0.65)] backdrop-blur">
      <div className="mx-auto max-w-6xl px-6 py-4 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="h-11 w-11 rounded-xl bg-gradient-to-br from-[var(--olive)] to-[var(--olive-2)] text-white flex items-center justify-center shadow-soft">
            <Volume2 size={18} />
          </div>
          <div>
            <div className="text-lg font-bold tracking-tight">VoiceShield</div>
            <div className="text-xs text-slate-600">
              Real vs Fake voice detection
            </div>
          </div>
        </div>

        <nav className="flex items-center gap-2">
          {nav.map((item) => {
            const Icon = item.icon;
            const active = page === item.key;
            return (
              <button
                key={item.key}
                onClick={() => setPage(item.key)}
                className={cx(
                  "px-3 py-2 rounded-lg text-sm font-semibold flex items-center gap-2 transition",
                  active
                    ? "bg-white border border-[var(--border)] shadow-soft text-[var(--olive)]"
                    : "text-slate-700 hover:bg-white/70 hover:border hover:border-[var(--border)]"
                )}
              >
                <Icon size={16} />
                {item.label}
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
}

function SectionCard({ title, icon: Icon, children, right }) {
  return (
    <section className="glass shadow-soft rounded-2xl p-6">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          {Icon ? (
            <div className="h-10 w-10 rounded-xl bg-[rgba(85,107,47,0.10)] border border-[var(--border)] flex items-center justify-center text-[var(--olive)]">
              <Icon size={18} />
            </div>
          ) : null}
          <h2 className="text-lg font-bold">{title}</h2>
        </div>
        {right}
      </div>
      <div className="mt-4">{children}</div>
    </section>
  );
}

function HomePage() {
  return (
    <div className="grid gap-6">
      <SectionCard title="About The Project" icon={Info}>
        <p className="text-slate-700 leading-relaxed">
          This web app detects whether an uploaded voice recording is{" "}
          <span className="font-semibold">REAL</span> or{" "}
          <span className="font-semibold">FAKE</span> (AI-generated) using your
          trained ML model.
        </p>
        <div className="mt-4 grid md:grid-cols-3 gap-4">
          <div className="rounded-xl border border-[var(--border)] bg-white/60 p-4">
            <div className="text-sm font-bold text-olive">Upload</div>
            <div className="mt-1 text-sm text-slate-700">
              Drop an audio file in the Audio tab (WAV/MP3/FLAC/etc).
            </div>
          </div>
          <div className="rounded-xl border border-[var(--border)] bg-white/60 p-4">
            <div className="text-sm font-bold text-olive">Predict</div>
            <div className="mt-1 text-sm text-slate-700">
              The backend forwards the audio to the FastAPI ML service for
              inference.
            </div>
          </div>
          <div className="rounded-xl border border-[var(--border)] bg-white/60 p-4">
            <div className="text-sm font-bold text-olive">History</div>
            <div className="mt-1 text-sm text-slate-700">
              Every analyzed file is stored and shown in the History tab.
            </div>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="How The System Connects">
        <div className="rounded-xl border border-[var(--border)] bg-white/60 p-4 text-sm text-slate-700 leading-relaxed">
          <div className="font-semibold text-slate-900">
            React UI → Node Backend → FastAPI ML Service
          </div>
          <ul className="mt-2 list-disc pl-5">
            <li>
              React sends the uploaded audio to{" "}
              <span className="font-mono">{API_BASE_URL}/api/analyze</span>
            </li>
            <li>
              Node forwards it to the ML service{" "}
              <span className="font-mono">/predict</span> and stores the result
            </li>
            <li>React shows the result and updates History</li>
          </ul>
        </div>
      </SectionCard>
    </div>
  );
}

function AudioPage({ onAnalyzeComplete, apiStatus }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const inputRef = useRef(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  async function analyze() {
    if (!file || busy) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const resp = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: "POST",
        body: form,
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        throw new Error(data.error || data.detail || `API error: ${resp.status}`);
      }
      if (!data?.item) {
        throw new Error("Unexpected server response");
      }
      setResult(data.item);
      onAnalyzeComplete?.(data.item);
    } catch (e) {
      setError(e?.message || "Failed to analyze audio");
    } finally {
      setBusy(false);
    }
  }

  function onDrop(e) {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (!f) return;
    if (!f.type.startsWith("audio/") && !f.name.match(/\.(wav|mp3|flac|ogg|m4a|webm)$/i)) {
      setError("Please drop an audio file");
      return;
    }
    setFile(f);
  }

  return (
    <div className="grid gap-6">
      <SectionCard
        title="Upload Audio"
        icon={UploadCloud}
        right={
          <div className="text-xs text-slate-600">
            Backend:{" "}
            <span className={cx(apiStatus.ok ? "text-emerald-700" : "text-rose-700")}>
              {apiStatus.ok ? "Connected" : "Not connected"}
            </span>
          </div>
        }
      >
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={onDrop}
          className={cx(
            "rounded-2xl border-2 border-dashed p-6 md:p-10 text-center transition",
            "border-[var(--border)] bg-white/55 hover:bg-white/70"
          )}
        >
          <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-[rgba(85,107,47,0.10)] border border-[var(--border)] text-[var(--olive)]">
            <UploadCloud />
          </div>
          <div className="mt-4 text-base font-bold">
            Drag & drop an audio file
          </div>
          <div className="mt-1 text-sm text-slate-600">
            or choose a file from your device
          </div>
          <div className="mt-4 flex items-center justify-center gap-3">
            <input
              ref={inputRef}
              type="file"
              accept="audio/*"
              className="hidden"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
            <button
              onClick={() => inputRef.current?.click()}
              className="px-4 py-2 rounded-lg bg-[var(--olive)] text-white font-semibold hover:opacity-95 active:opacity-90"
              disabled={busy}
            >
              Choose file
            </button>
            <button
              onClick={() => {
                setFile(null);
                setResult(null);
                setError(null);
              }}
              className="px-4 py-2 rounded-lg border border-[var(--border)] bg-white/60 font-semibold text-slate-700 hover:bg-white"
              disabled={busy}
            >
              Reset
            </button>
          </div>
          <div className="mt-3 text-xs text-slate-500">
            Supported: WAV, MP3, FLAC, OGG, M4A, WebM (max 50MB)
          </div>
        </div>

        {file ? (
          <div className="mt-4 rounded-2xl border border-[var(--border)] bg-white/60 p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <div className="text-sm font-bold text-slate-900">Selected</div>
                <div className="text-sm text-slate-700 break-all">{file.name}</div>
              </div>
              <button
                onClick={analyze}
                disabled={busy || !apiStatus.ok}
                className={cx(
                  "px-4 py-2 rounded-lg font-semibold text-white flex items-center gap-2",
                  busy || !apiStatus.ok ? "bg-slate-400" : "bg-[var(--olive-2)] hover:opacity-95"
                )}
              >
                {busy ? <Loader2 className="animate-spin" size={18} /> : null}
                {busy ? "Analyzing..." : "Analyze"}
              </button>
            </div>
            {previewUrl ? (
              <audio className="mt-3 w-full" controls src={previewUrl} />
            ) : null}
          </div>
        ) : null}

        {error ? (
          <div className="mt-4 rounded-2xl border border-rose-200 bg-rose-50 p-4 text-rose-800">
            <div className="font-semibold">Error</div>
            <div className="text-sm mt-1">{error}</div>
          </div>
        ) : null}

        {result ? (
          <div className="mt-4 rounded-2xl border border-[var(--border)] bg-white/70 p-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="text-sm text-slate-600">
                Result for <span className="font-semibold">{result.fileName}</span>
              </div>
              <PredictionPill
                prediction={result.prediction}
                confidence={result.confidence}
              />
            </div>
            <div className="mt-3">
              <div className="h-2 w-full rounded-full bg-slate-200 overflow-hidden">
                <div
                  className={cx(
                    "h-full",
                    result.prediction === "REAL" ? "bg-emerald-500" : "bg-rose-500"
                  )}
                  style={{ width: `${Math.max(0, Math.min(100, result.confidence))}%` }}
                />
              </div>
              <div className="mt-2 text-xs text-slate-600">
                Analyzed at {formatDateTime(result.uploadedAt)}
              </div>
            </div>
          </div>
        ) : null}
      </SectionCard>
    </div>
  );
}

function HistoryPage({
  history,
  selectedId,
  setSelectedId,
  onClear,
  refreshing,
  onRefresh,
}) {
  const selected = useMemo(
    () => history.find((x) => x.id === selectedId) || null,
    [history, selectedId]
  );

  return (
    <div className="grid md:grid-cols-5 gap-6">
      <div className="md:col-span-2">
        <SectionCard
          title="History"
          icon={HistoryIcon}
          right={
            <div className="flex items-center gap-2">
              <button
                onClick={onRefresh}
                disabled={refreshing}
                className="px-3 py-2 rounded-lg border border-[var(--border)] bg-white/60 text-sm font-semibold hover:bg-white disabled:opacity-70"
              >
                {refreshing ? "Refreshing..." : "Refresh"}
              </button>
              <button
                onClick={onClear}
                disabled={history.length === 0}
                className="px-3 py-2 rounded-lg bg-rose-600 text-white text-sm font-semibold hover:opacity-95 disabled:opacity-60 flex items-center gap-2"
              >
                <Trash2 size={16} />
                Clear
              </button>
            </div>
          }
        >
          {history.length === 0 ? (
            <div className="rounded-xl border border-[var(--border)] bg-white/60 p-4 text-sm text-slate-700">
              No history yet. Analyze an audio file to see it here.
            </div>
          ) : (
            <div className="grid gap-3">
              {history.map((item) => (
                <button
                  key={item.id}
                  onClick={() => setSelectedId(item.id)}
                  className={cx(
                    "text-left rounded-2xl border p-4 transition",
                    selectedId === item.id
                      ? "border-[var(--olive-2)] bg-white shadow-soft"
                      : "border-[var(--border)] bg-white/55 hover:bg-white/75"
                  )}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="text-sm font-bold text-slate-900 truncate">
                      {item.fileName}
                    </div>
                    <span className="text-xs text-slate-500">
                      {new Date(item.uploadedAt).toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </span>
                  </div>
                  <div className="mt-2 flex items-center justify-between gap-2">
                    <PredictionPill
                      prediction={item.prediction}
                      confidence={item.confidence}
                    />
                    <span className="text-xs text-slate-500">
                      {new Date(item.uploadedAt).toLocaleDateString()}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          )}
        </SectionCard>
      </div>

      <div className="md:col-span-3">
        <SectionCard title="Details" icon={Mic2}>
          {!selected ? (
            <div className="rounded-xl border border-[var(--border)] bg-white/60 p-4 text-sm text-slate-700">
              Select a history item to see details and play the audio.
            </div>
          ) : (
            <div className="rounded-2xl border border-[var(--border)] bg-white/70 p-5">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <div className="text-sm text-slate-600">File</div>
                  <div className="text-base font-bold break-all">
                    {selected.fileName}
                  </div>
                  <div className="mt-1 text-xs text-slate-600">
                    {formatDateTime(selected.uploadedAt)}
                  </div>
                </div>
                <PredictionPill
                  prediction={selected.prediction}
                  confidence={selected.confidence}
                />
              </div>
              <audio
                className="mt-4 w-full"
                controls
                src={`${API_BASE_URL}${selected.audioUrl}`}
              />
              <div className="mt-4 text-sm text-slate-700">
                {selected.prediction === "REAL" ? (
                  <div>
                    <span className="font-semibold text-emerald-800">
                      Looks authentic.
                    </span>{" "}
                    The model considers this audio similar to real-voice patterns.
                  </div>
                ) : (
                  <div>
                    <span className="font-semibold text-rose-800">
                      Potentially AI-generated.
                    </span>{" "}
                    The model flagged this audio as an outlier compared to real-voice patterns.
                  </div>
                )}
              </div>
            </div>
          )}
        </SectionCard>
      </div>
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState("home");
  const [history, setHistory] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [apiStatus, setApiStatus] = useState({ ok: false, checked: false });
  const [refreshing, setRefreshing] = useState(false);

  async function refreshHistory() {
    setRefreshing(true);
    try {
      const resp = await fetch(`${API_BASE_URL}/api/history`);
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(data.error || `API error: ${resp.status}`);
      const items = Array.isArray(data.items) ? data.items : [];
      setHistory(items);
      if (!selectedId && items[0]?.id) setSelectedId(items[0].id);
    } catch {
      // ignore; UI already indicates connection state
    } finally {
      setRefreshing(false);
    }
  }

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const resp = await fetch(`${API_BASE_URL}/api/health`);
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error("health failed");
        if (!mounted) return;
        setApiStatus({ ok: true, checked: true, info: data });
        await refreshHistory();
      } catch {
        if (!mounted) return;
        setApiStatus({ ok: false, checked: true });
      }
    })();
    return () => {
      mounted = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function clearHistory() {
    try {
      const resp = await fetch(`${API_BASE_URL}/api/history`, { method: "DELETE" });
      if (!resp.ok) return;
      setHistory([]);
      setSelectedId(null);
    } catch {
      // ignore
    }
  }

  function onAnalyzeComplete(item) {
    setHistory((prev) => [item, ...prev]);
    setSelectedId(item.id);
  }

  return (
    <div className="min-h-full">
      <TopNav page={page} setPage={setPage} />

      <main className="mx-auto max-w-6xl px-6 py-10">
        {!apiStatus.ok && apiStatus.checked ? (
          <div className="mb-6 rounded-2xl border border-amber-200 bg-amber-50 p-4 text-amber-900">
            <div className="font-bold">Backend not connected</div>
            <div className="mt-1 text-sm">
              Start the Node backend on{" "}
              <span className="font-mono">http://localhost:5001</span> and the ML
              service on <span className="font-mono">http://localhost:8000</span>.
            </div>
          </div>
        ) : null}

        {page === "home" ? <HomePage /> : null}
        {page === "audio" ? (
          <AudioPage onAnalyzeComplete={onAnalyzeComplete} apiStatus={apiStatus} />
        ) : null}
        {page === "history" ? (
          <HistoryPage
            history={history}
            selectedId={selectedId}
            setSelectedId={setSelectedId}
            onClear={clearHistory}
            refreshing={refreshing}
            onRefresh={refreshHistory}
          />
        ) : null}
      </main>

      <footer className="pb-10 text-center text-xs text-slate-600">
        <div>
          VoiceShield • UI palette: light green / white / olive •{" "}
          <span className="font-mono">{API_BASE_URL}</span>
        </div>
      </footer>
    </div>
  );
}

