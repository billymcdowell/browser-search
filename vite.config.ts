import { defineConfig, type Plugin } from 'vite';

// onnxruntime-web references its own WASM binaries, and Vite would copy a second
// 26 MB copy into dist/assets. The app loads the runtime from public/wasm/ instead
// (see src/search.worker.ts), so drop the bundled copies.
const dropBundledOrtWasm = (): Plugin => ({
  name: 'drop-bundled-ort-wasm',
  enforce: 'post',
  generateBundle(_options, bundle) {
    for (const file of Object.keys(bundle)) {
      if (/^assets\/ort-wasm.*\.wasm$/.test(file)) delete bundle[file];
    }
  },
});

// Relative base so the built site works from the root of any static server or a sub-folder.
export default defineConfig({
  base: './',
  plugins: [dropBundledOrtWasm()],
  worker: { format: 'es' },
  build: {
    target: 'es2022',
    chunkSizeWarningLimit: 4000,
  },
});
