// ─────────────────────────────────────────────────────────────
// Multimodal AI Detector — Express Backend
// Proxies image, audio, and video-frame analysis to Hugging Face
// Serverless Inference API, keeping API tokens server-side.
// ─────────────────────────────────────────────────────────────

import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;
const HF_API_KEY = process.env.HF_API_KEY;

// ── Multer (in-memory) ──────────────────────────────────────
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 }, // 50 MB ceiling
});

// ── Static files ─────────────────────────────────────────────
app.use(express.static(join(__dirname, 'public')));

// ── HF model endpoints ──────────────────────────────────────
const HF_BASE = 'https://router.huggingface.co/hf-inference/models';

// Image ensemble: 3 models for majority-vote accuracy
const IMAGE_MODELS = [
  { url: `${HF_BASE}/dima806/ai_vs_real_image_detection`,  name: 'ViT (dima806)' },
  { url: `${HF_BASE}/umm-maybe/AI-image-detector`,         name: 'ViT (umm-maybe)' },
  { url: `${HF_BASE}/Organika/sdxl-detector`,              name: 'SDXL Detector' },
];
const HF_AUDIO_MODEL = `${HF_BASE}/mo-thecreator/Deepfake-audio-detection`;

// ─────────────────────────────────────────────────────────────
// Helper: call Hugging Face Inference API and handle errors
// ─────────────────────────────────────────────────────────────
async function callHuggingFace(url, buffer, contentType = 'application/octet-stream') {
  // Guard: no API key configured
  if (!HF_API_KEY || HF_API_KEY === 'your_huggingface_api_token_here') {
    return {
      ok: false,
      retryable: false,
      error: 'HF_API_KEY is not configured on the server. Set it in .env and restart.',
    };
  }

  let response;
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${HF_API_KEY}`,
        'Content-Type': contentType,
      },
      body: buffer,
      signal: AbortSignal.timeout(60_000), // 60-second timeout
    });
  } catch (err) {
    // Network / DNS / timeout errors
    const code = err?.cause?.code || err.code || '';
    if (code === 'ENOTFOUND') {
      return {
        ok: false,
        retryable: true,
        error: 'Cannot reach Hugging Face API — DNS resolution failed. Check your internet connection or try again shortly.',
      };
    }
    if (code === 'ECONNREFUSED' || code === 'ECONNRESET') {
      return {
        ok: false,
        retryable: true,
        error: 'Connection to Hugging Face API was refused/reset. Try again shortly.',
      };
    }
    if (err.name === 'TimeoutError' || err.name === 'AbortError') {
      return {
        ok: false,
        retryable: true,
        error: 'Request to Hugging Face timed out (60s). The model may be cold-starting — retry in ~20s.',
      };
    }
    return {
      ok: false,
      retryable: true,
      error: `Network error contacting Hugging Face: ${err.message}`,
    };
  }

  // Model still loading (cold-start)
  if (response.status === 503) {
    const body = await response.json().catch(() => ({}));
    const wait = body.estimated_time ? Math.ceil(body.estimated_time) : 20;
    return {
      ok: false,
      retryable: true,
      error: `Model is warming up — please retry in ~${wait}s.`,
      estimatedWait: wait,
    };
  }

  // Auth errors
  if (response.status === 401 || response.status === 403) {
    return {
      ok: false,
      retryable: false,
      error: 'Hugging Face API key is invalid or expired. Update HF_API_KEY in .env and restart the server.',
    };
  }

  if (!response.ok) {
    const text = await response.text().catch(() => 'Unknown error');
    return { ok: false, retryable: false, error: `Hugging Face API error (${response.status}): ${text}` };
  }

  const data = await response.json();
  return { ok: true, data };
}

// ─────────────────────────────────────────────────────────────
// Helper: extract fake/real scores from any single-model HF response.
// Different models use different label names; we normalise them all.
// ─────────────────────────────────────────────────────────────
function extractScores(data) {
  const flat = Array.isArray(data[0]) ? data[0] : data;
  let fakeScore = 0;
  let realScore = 0;
  for (const item of flat) {
    const lbl = item.label.toLowerCase();
    if (lbl.includes('ai') || lbl.includes('fake') || lbl.includes('generated') ||
        lbl.includes('artificial') || lbl.includes('sdxl') || lbl.includes('synthetic')) {
      fakeScore = item.score;
    } else {
      realScore = item.score;
    }
  }
  return { fakeScore, realScore, raw: flat };
}

// ─────────────────────────────────────────────────────────────
// Ensemble: query all IMAGE_MODELS, average scores, majority-vote.
// Returns { verdict, confidence, fakeScore, realScore, models, analysisSource }
// ─────────────────────────────────────────────────────────────
async function ensembleImageAnalysis(buffer) {
  const results = await Promise.all(
    IMAGE_MODELS.map(async (m) => {
      const r = await callHuggingFace(m.url, buffer);
      return { ...r, modelName: m.name };
    })
  );

  // If ALL models are retryable-errored, bubble the first one up
  const retryable = results.find((r) => !r.ok && r.retryable);
  const successes = results.filter((r) => r.ok);
  if (successes.length === 0) {
    if (retryable) return { ok: false, retryable: true, error: retryable.error };
    return { ok: false, retryable: false, error: results[0].error };
  }

  // Parse each successful model's scores
  const parsed = successes.map((r) => ({
    ...extractScores(r.data),
    modelName: r.modelName,
  }));

  // Weighted average of scores
  const avgFake = parsed.reduce((s, p) => s + p.fakeScore, 0) / parsed.length;
  const avgReal = parsed.reduce((s, p) => s + p.realScore, 0) / parsed.length;

  // Majority vote: count how many models lean "fake"
  const fakeVotes = parsed.filter((p) => p.fakeScore > p.realScore).length;
  const totalVotes = parsed.length;

  // Decision logic — higher thresholds + majority vote to reduce false positives:
  //   AI verdict: majority of models say fake AND average fake score >= 0.60
  //   Real verdict: majority say real OR average real score >= 0.55
  //   Otherwise: inconclusive
  let verdict = 'inconclusive';
  if (fakeVotes > totalVotes / 2 && avgFake >= 0.60) {
    verdict = 'ai';
  } else if (fakeVotes <= totalVotes / 2 && avgReal >= 0.55) {
    verdict = 'real';
  } else if (avgReal > avgFake) {
    verdict = 'real';
  }

  const confidence = Math.max(avgFake, avgReal);
  const modelNames = parsed.map((p) => p.modelName).join(', ');

  return {
    ok: true,
    verdict,
    confidence,
    fakeScore: avgFake,
    realScore: avgReal,
    modelsUsed: parsed.length,
    modelNames,
    fakeVotes,
    totalVotes,
    perModel: parsed,
    analysisSource: `Ensemble: ${parsed.length} models (${modelNames})`,
  };
}

// ─────────────────────────────────────────────────────────────
// POST /api/detect/image — 3-model ensemble
// ─────────────────────────────────────────────────────────────
app.post('/api/detect/image', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No image file uploaded.' });

    const result = await ensembleImageAnalysis(req.file.buffer);
    if (!result.ok) {
      const status = result.retryable ? 503 : 502;
      return res.status(status).json({ error: result.error, retryable: result.retryable });
    }

    return res.json({
      modality: 'image',
      ...result,
    });
  } catch (err) {
    console.error('[/api/detect/image]', err);
    return res.status(500).json({ error: 'Internal server error during image analysis.' });
  }
});

// ─────────────────────────────────────────────────────────────
// POST /api/detect/audio
// ─────────────────────────────────────────────────────────────
app.post('/api/detect/audio', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No audio file uploaded.' });

    const result = await callHuggingFace(HF_AUDIO_MODEL, req.file.buffer);
    if (!result.ok) {
      const status = result.retryable ? 503 : 502;
      return res.status(status).json({ error: result.error, retryable: result.retryable });
    }

    // Normalise audio labels → fake / real
    const flat = Array.isArray(result.data[0]) ? result.data[0] : result.data;
    let fakeScore = 0;
    let realScore = 0;
    for (const item of flat) {
      const lbl = item.label.toLowerCase();
      if (lbl.includes('fake') || lbl.includes('spoof') || lbl.includes('deepfake') || lbl.includes('synthetic')) {
        fakeScore = item.score;
      } else {
        realScore = item.score;
      }
    }
    const confidence = Math.max(fakeScore, realScore);
    let verdict = 'inconclusive';
    if (fakeScore >= 0.5) verdict = 'ai';
    else if (realScore >= 0.5) verdict = 'real';

    return res.json({
      modality: 'audio',
      model: 'mo-thecreator/Deepfake-audio-detection',
      analysisSource: 'Wav2Vec2 Acoustic Model',
      verdict,
      confidence,
      fakeScore,
      realScore,
      raw: flat,
    });
  } catch (err) {
    console.error('[/api/detect/audio]', err);
    return res.status(500).json({ error: 'Internal server error during audio analysis.' });
  }
});

// ─────────────────────────────────────────────────────────────
// POST /api/detect/video-frames
// Receives up to 3 extracted JPEG keyframes and runs each
// through the image model concurrently.
// ─────────────────────────────────────────────────────────────
app.post('/api/detect/video-frames', upload.array('frames', 3), async (req, res) => {
  try {
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ error: 'No video frames uploaded.' });
    }

    // Run each frame through the full 3-model ensemble
    const frameResults = await Promise.all(
      req.files.map((f) => ensembleImageAnalysis(f.buffer))
    );

    // If any frame ensemble got a retryable error, bubble it up
    const retryable = frameResults.find((r) => !r.ok && r.retryable);
    if (retryable) {
      return res.status(503).json({ error: retryable.error, retryable: true });
    }

    const successes = frameResults.filter((r) => r.ok);
    if (successes.length === 0) {
      return res.status(502).json({ error: frameResults[0].error, retryable: false });
    }

    const avgFake = successes.reduce((s, r) => s + r.fakeScore, 0) / successes.length;
    const avgReal = successes.reduce((s, r) => s + r.realScore, 0) / successes.length;
    const avgConf = Math.max(avgFake, avgReal);

    // Use same majority-vote logic across frames
    const framesVotingAI = successes.filter((r) => r.verdict === 'ai').length;
    let verdict = 'inconclusive';
    if (framesVotingAI > successes.length / 2 && avgFake >= 0.60) verdict = 'ai';
    else if (avgReal > avgFake) verdict = 'real';

    return res.json({
      modality: 'video',
      analysisSource: `${successes.length} Keyframes × 3-Model Ensemble`,
      verdict,
      confidence: avgConf,
      fakeScore: avgFake,
      realScore: avgReal,
      framesAnalyzed: successes.length,
      perFrame: successes,
    });
  } catch (err) {
    console.error('[/api/detect/video-frames]', err);
    return res.status(500).json({ error: 'Internal server error during video analysis.' });
  }
});

// ── Health-check ─────────────────────────────────────────────
app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// ── Catch-all: serve index.html for SPA-like routing ─────────
app.get('*', (_req, res) => {
  res.sendFile(join(__dirname, 'public', 'index.html'));
});

// ── Global error handler (catches multer/busboy crashes) ─────
app.use((err, _req, res, _next) => {
  console.error('[Global Error Handler]', err.message || err);
  if (!res.headersSent) {
    res.status(400).json({ error: 'Bad request: ' + (err.message || 'malformed upload') });
  }
});

// ── Process-level guards — never crash on stray errors ───────
process.on('uncaughtException', (err) => {
  console.error('[Uncaught Exception]', err);
});
process.on('unhandledRejection', (reason) => {
  console.error('[Unhandled Rejection]', reason);
});

// ── Start ────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n  🛡️  Multimodal AI Detector running → http://localhost:${PORT}\n`);
  if (!HF_API_KEY || HF_API_KEY === 'your_huggingface_api_token_here') {
    console.warn('  ⚠️  WARNING: HF_API_KEY is not set. API calls will fail.\n');
  }
});
