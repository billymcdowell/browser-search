# Commodity price search

A static, browser-only demo for searching 10,000 commodity price series in `commodity-price-taxonomy.json`.
It offers three search modes:

- **Fuzzy**: keyword matching that tolerates typos ("cofee" finds Coffee) and partial words.
- **Semantic**: plain-English search. The query is embedded in the browser with `all-MiniLM-L6-v2`
  (ONNX, quantized) and compared against precomputed price embeddings.
- **Hybrid**: fuzzy and semantic rankings merged with reciprocal rank fusion.

## Build

```bash
npm install
npm run build        # embeds the data (cached), typechecks, and writes dist/
```

`npm run build-index` (run automatically by `build`) flattens the taxonomy, embeds every price leaf
(about 25 seconds on a laptop), and writes:

- `public/data/prices.json`, `public/data/embeddings.bin`, `public/data/meta.json`
- `public/models/…`, the model files, so nothing is downloaded from a CDN at runtime
- `public/wasm/…`, the ONNX Runtime WASM, self-hosted

The step is skipped when the source JSON and model id have not changed.

## Serve

`dist/` is plain static files. Any HTTP server works, with no special headers needed:

```bash
cd dist && python3 -m http.server 8080
# then open http://localhost:8080
```

To publish, upload the contents of `dist/` to any static host. The build uses relative paths, so it also works
from a sub-folder.

## Develop

```bash
npm run dev
```

## Notes

- The model runs single-threaded, so it does not need cross-origin isolation headers.
- The first load downloads roughly 23 MB of model and about 14 MB of WASM. Both are cached by the browser afterward.
- Generated folders (`public/data`, `public/models`, `public/wasm`, `.cache`) are git-ignored and recreated by the build.
