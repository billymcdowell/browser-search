// Runs the embedding model and all searches off the main thread, so typing never stalls the page.
import { env, pipeline } from '@huggingface/transformers';
import {
  FuzzyIndex,
  VectorIndex,
  reciprocalRankFusion,
  type Hit,
  type PriceRecord,
} from './search-core';

export type SearchMode = 'fuzzy' | 'semantic' | 'hybrid';

export type WorkerRequest =
  | { type: 'init'; baseUrl: string }
  | { type: 'search'; id: number; query: string; mode: SearchMode; limit: number };

export type WorkerResponse =
  | { type: 'data-ready'; records: PriceRecord[]; dim: number; modelId: string }
  | { type: 'model-progress'; file: string; progress: number }
  | { type: 'model-ready'; ms: number }
  | { type: 'model-error'; message: string }
  | { type: 'results'; id: number; mode: SearchMode; hits: Hit[]; embedMs: number; searchMs: number }
  | { type: 'search-error'; id: number; message: string };

const MODEL_ID = 'Xenova/all-MiniLM-L6-v2';
// Pool of candidates kept from each ranker before fusing in hybrid mode.
const HYBRID_POOL = 200;

type Extractor = (
  text: string,
  options: { pooling: 'mean'; normalize: boolean },
) => Promise<{ data: Float32Array; dims: number[] }>;

let records: PriceRecord[] = [];
let fuzzy: FuzzyIndex | null = null;
let vectors: VectorIndex | null = null;
let extractorPromise: Promise<Extractor> | null = null;

const post = (msg: WorkerResponse) => self.postMessage(msg);

async function loadData(baseUrl: string): Promise<void> {
  const [pricesRes, metaRes, embRes] = await Promise.all([
    fetch(new URL('data/prices.json', baseUrl)),
    fetch(new URL('data/meta.json', baseUrl)),
    fetch(new URL('data/embeddings.bin', baseUrl)),
  ]);
  for (const r of [pricesRes, metaRes, embRes]) {
    if (!r.ok) throw new Error(`Could not load ${r.url} (HTTP ${r.status}). Run "npm run build" first.`);
  }
  records = (await pricesRes.json()) as PriceRecord[];
  const meta = (await metaRes.json()) as { dim: number; count: number; model: string };
  const buffer = await embRes.arrayBuffer();

  fuzzy = new FuzzyIndex(records);
  vectors = new VectorIndex(buffer, meta.count, meta.dim);
  post({ type: 'data-ready', records, dim: meta.dim, modelId: meta.model });
}

function loadModel(baseUrl: string): Promise<Extractor> {
  // Everything is served from our own origin. No CDN and no Hugging Face calls at runtime.
  env.allowLocalModels = true;
  env.allowRemoteModels = false;
  // Must be a path, not a full URL: transformers.js treats full URLs as remote and skips the local lookup.
  env.localModelPath = new URL('models/', baseUrl).pathname;
  const onnx = (env as unknown as { backends: { onnx: { wasm: Record<string, unknown> } } }).backends.onnx;
  onnx.wasm.wasmPaths = {
    mjs: new URL('wasm/ort-wasm-simd-threaded.mjs', baseUrl).href,
    wasm: new URL('wasm/ort-wasm-simd-threaded.wasm', baseUrl).href,
  };
  onnx.wasm.numThreads = 1; // avoids needing cross-origin isolation headers on the server

  const started = performance.now();
  return pipeline('feature-extraction', MODEL_ID, {
    dtype: 'q8',
    progress_callback: (p: { status?: string; file?: string; progress?: number }) => {
      if (p.status === 'progress' && p.file) {
        post({ type: 'model-progress', file: p.file, progress: p.progress ?? 0 });
      }
    },
  }).then((ext) => {
    post({ type: 'model-ready', ms: performance.now() - started });
    return ext as unknown as Extractor;
  });
}

async function embedQuery(text: string): Promise<Float32Array> {
  if (!extractorPromise) throw new Error('Model not loaded');
  const extract = await extractorPromise;
  const out = await extract(text, { pooling: 'mean', normalize: true });
  return out.data;
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data;

  if (msg.type === 'init') {
    try {
      await loadData(msg.baseUrl);
    } catch (err) {
      post({ type: 'model-error', message: (err as Error).message });
      return;
    }
    // Fuzzy search works straight away. The model loads in the background.
    extractorPromise = loadModel(msg.baseUrl);
    extractorPromise.catch((err: Error) => {
      post({ type: 'model-error', message: err.message });
    });
    return;
  }

  if (msg.type === 'search') {
    try {
      if (!fuzzy || !vectors) throw new Error('Data not loaded yet');
      const { id, query, mode, limit } = msg;
      const q = query.trim();
      if (!q) {
        post({ type: 'results', id, mode, hits: [], embedMs: 0, searchMs: 0 });
        return;
      }

      if (mode === 'fuzzy') {
        const t = performance.now();
        const hits = fuzzy.search(q, limit);
        post({ type: 'results', id, mode, hits, embedMs: 0, searchMs: performance.now() - t });
        return;
      }

      const e0 = performance.now();
      const qv = await embedQuery(q);
      const embedMs = performance.now() - e0;

      const s0 = performance.now();
      const semantic = vectors.search(qv, mode === 'semantic' ? limit : HYBRID_POOL);
      let hits: Hit[] = semantic;
      if (mode === 'hybrid') {
        hits = reciprocalRankFusion([fuzzy.search(q, HYBRID_POOL), semantic], limit);
      }
      post({ type: 'results', id, mode, hits, embedMs, searchMs: performance.now() - s0 });
    } catch (err) {
      post({ type: 'search-error', id: msg.id, message: (err as Error).message });
    }
  }
};
