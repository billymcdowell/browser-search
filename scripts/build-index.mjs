// Build-time step: flattens the commodity taxonomy, embeds every price leaf with
// a sentence-embedding model, and writes everything the browser needs into public/.
//
// Outputs (all served as static files):
//   public/data/prices.json        flat list of price records
//   public/data/embeddings.bin     per-row scale (float32) + int8 vectors
//   public/data/meta.json          counts, model id, input hash
//   public/models/<model>/...      model files, so the browser never calls out to a CDN
//   public/wasm/ort-*              onnxruntime-web WASM runtime, self-hosted
//
// The step is skipped when the source JSON and model have not changed since the last run.

import { readFile, writeFile, mkdir, copyFile, access, rm } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { pipeline, env } from '@huggingface/transformers';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SRC = path.join(ROOT, 'commodity-price-taxonomy.json');
const DATA_DIR = path.join(ROOT, 'public', 'data');
const MODEL_DIR = path.join(ROOT, 'public', 'models');
const WASM_DIR = path.join(ROOT, 'public', 'wasm');
const CACHE_DIR = path.join(ROOT, '.cache', 'models');

const MODEL_ID = 'Xenova/all-MiniLM-L6-v2';
const MODEL_FILES = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'onnx/model_quantized.onnx'];
const WASM_FILES = ['ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm'];
const DOC_TEMPLATE_VERSION = '1';
const BATCH_SIZE = 64;

const exists = (p) => access(p).then(() => true, () => false);

// ---------- 1. Flatten the taxonomy into price records ----------
async function loadPrices(raw) {
  const json = JSON.parse(raw);
  const out = [];
  const walk = (node, trail) => {
    if (node.type === 'price') {
      out.push({ id: node.id, code: node.code, name: node.name, unit: node.unit ?? '', path: trail });
      return;
    }
    const next = node.type === 'category' && node.name ? [...trail, node.name] : trail;
    for (const child of node.children ?? []) walk(child, next);
  };
  for (const root of json.taxonomy ?? []) walk(root, []);
  return out;
}

// Text that gets embedded. Sector / group / commodity give the model context
// for natural-language queries such as "fuel to heat homes".
const docText = (r) =>
  `${r.name}. Category: ${r.path.join(' > ')}. Unit: ${r.unit}. Code: ${r.code}.`;

async function main() {
  const raw = await readFile(SRC, 'utf8');
  const records = await loadPrices(raw);
  console.log(`Flattened ${records.length} price records.`);

  const hash = createHash('sha256')
    .update(raw).update(MODEL_ID).update(DOC_TEMPLATE_VERSION).digest('hex');

  await mkdir(DATA_DIR, { recursive: true });
  await mkdir(MODEL_DIR, { recursive: true });
  await mkdir(WASM_DIR, { recursive: true });

  const metaPath = path.join(DATA_DIR, 'meta.json');
  const embPath = path.join(DATA_DIR, 'embeddings.bin');
  const prevMeta = (await exists(metaPath)) ? JSON.parse(await readFile(metaPath, 'utf8')) : null;

  if (prevMeta?.hash === hash && (await exists(embPath))) {
    console.log('Embeddings are up to date, skipping embedding step.');
  } else {
    env.cacheDir = CACHE_DIR;
    console.log(`Loading ${MODEL_ID} (quantized)...`);
    const extractor = await pipeline('feature-extraction', MODEL_ID, { dtype: 'q8' });

    const t0 = Date.now();
    let dim = 0;
    const texts = records.map(docText);
    const scales = new Float32Array(records.length);
    const vectors = new Int8Array(records.length * 384);

    for (let start = 0; start < texts.length; start += BATCH_SIZE) {
      const batch = texts.slice(start, start + BATCH_SIZE);
      const tensor = await extractor(batch, { pooling: 'mean', normalize: true });
      dim = tensor.dims[1];
      const flat = tensor.data;
      for (let row = 0; row < batch.length; row++) {
        const i = start + row;
        let maxAbs = 0;
        for (let d = 0; d < dim; d++) maxAbs = Math.max(maxAbs, Math.abs(flat[row * dim + d]));
        scales[i] = maxAbs || 1;
        for (let d = 0; d < dim; d++) {
          vectors[i * dim + d] = Math.round((flat[row * dim + d] / scales[i]) * 127);
        }
      }
      if ((start / BATCH_SIZE) % 20 === 0) {
        process.stdout.write(`\r  embedded ${Math.min(start + BATCH_SIZE, texts.length)} / ${texts.length}`);
      }
    }
    process.stdout.write(`\r  embedded ${texts.length} / ${texts.length}\n`);

    // Layout: [count * float32 scales][count * dim int8 values]
    const header = Buffer.from(scales.buffer);
    await writeFile(embPath, Buffer.concat([header, Buffer.from(vectors.buffer)]));
    await writeFile(metaPath, JSON.stringify({
      hash,
      model: MODEL_ID,
      dim,
      count: records.length,
      encoding: 'float32 per-row scale, then int8 vectors (value = q / 127 * scale)',
      generatedAt: new Date().toISOString(),
    }, null, 2));
    console.log(`Embedded in ${((Date.now() - t0) / 1000).toFixed(1)}s, dim ${dim}.`);
  }

  await writeFile(path.join(DATA_DIR, 'prices.json'), JSON.stringify(records));

  // Copy the model and runtime files so the app runs from any static HTTP server.
  for (const f of MODEL_FILES) {
    const dest = path.join(MODEL_DIR, MODEL_ID, f);
    await mkdir(path.dirname(dest), { recursive: true });
    await copyFile(path.join(CACHE_DIR, MODEL_ID, f), dest);
  }
  const ortDir = path.join(ROOT, 'node_modules', 'onnxruntime-web', 'dist');
  for (const f of WASM_FILES) await copyFile(path.join(ortDir, f), path.join(WASM_DIR, f));
  console.log('Model and runtime files copied to public/.');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
