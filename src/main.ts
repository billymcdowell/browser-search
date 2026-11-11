import './style.css';
import type { PriceRecord, Hit } from './search-core';
import type { SearchMode, WorkerRequest, WorkerResponse } from './search.worker';

const RESULT_LIMIT = 50;
const DEBOUNCE_MS = 150;

const $ = <T extends HTMLElement>(sel: string) => document.querySelector<T>(sel)!;
const input = $<HTMLInputElement>('#query');
const resultsEl = $<HTMLOListElement>('#results');
const statsEl = $<HTMLDivElement>('#stats');
const emptyEl = $<HTMLParagraphElement>('#empty');
const modelEl = $<HTMLDivElement>('#model-status');
const modeButtons = [...document.querySelectorAll<HTMLButtonElement>('.mode')];

let records: PriceRecord[] = [];
let mode: SearchMode = 'fuzzy';
let requestId = 0;
let latestId = 0;
let debounceTimer: number | undefined;
let modelReady = false;
let waitingForModel = false;

const worker = new Worker(new URL('./search.worker.ts', import.meta.url), { type: 'module' });

const send = (msg: WorkerRequest) => worker.postMessage(msg);

function setModelStatus(text: string, state: 'loading' | 'ready' | 'error') {
  modelEl.dataset.state = state;
  modelEl.querySelector('.text')!.textContent = text;
}

worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
  const msg = event.data;
  switch (msg.type) {
    case 'data-ready':
      records = msg.records;
      statsEl.textContent = `${records.length.toLocaleString()} price series loaded.`;
      setModelStatus('AI model: loading…', 'loading');
      break;
    case 'model-progress':
      if (!modelReady && msg.file.endsWith('.onnx')) {
        setModelStatus(`AI model: loading ${Math.round(msg.progress)}%`, 'loading');
      }
      break;
    case 'model-ready':
      modelReady = true;
      setModelStatus(`AI model ready (${msg.ms < 1000 ? Math.round(msg.ms) + ' ms' : (msg.ms / 1000).toFixed(1) + ' s'} to load)`, 'ready');
      if (waitingForModel) {
        waitingForModel = false;
        runSearch();
      }
      break;
    case 'model-error':
      setModelStatus(`AI model unavailable: ${msg.message}`, 'error');
      if (mode !== 'fuzzy') {
        emptyEl.hidden = false;
        emptyEl.textContent = 'Semantic search needs the AI model. Fuzzy search still works.';
      }
      break;
    case 'results':
      if (msg.id !== latestId) return; // stale response
      renderResults(msg.hits, msg.mode, msg.embedMs, msg.searchMs);
      break;
    case 'search-error':
      if (msg.id !== latestId) return;
      emptyEl.hidden = false;
      emptyEl.textContent = `Search failed: ${msg.message}`;
      break;
  }
};

function formatMs(ms: number) {
  return ms < 10 ? `${ms.toFixed(1)} ms` : `${Math.round(ms)} ms`;
}

function renderResults(hits: Hit[], searchMode: SearchMode, embedMs: number, searchMs: number) {
  waitingForModel = false;
  resultsEl.replaceChildren();
  const query = input.value.trim();

  if (!query) {
    statsEl.textContent = '';
    emptyEl.hidden = false;
    emptyEl.textContent = 'Type a search, or pick an example above.';
    return;
  }

  const label = { fuzzy: 'Fuzzy', semantic: 'Semantic', hybrid: 'Hybrid' }[searchMode];
  const timing = searchMode === 'fuzzy'
    ? `search ${formatMs(searchMs)}`
    : `embed query ${formatMs(embedMs)} · search ${formatMs(searchMs)}`;
  statsEl.textContent = hits.length
    ? `${label}: ${hits.length} of ${records.length.toLocaleString()} shown · ${timing}`
    : `${label}: no matches · ${timing}`;

  emptyEl.hidden = hits.length > 0;
  if (!hits.length) {
    emptyEl.textContent = searchMode === 'fuzzy'
      ? 'No keyword matches. Try fewer words, or switch to Semantic for plain-English queries.'
      : 'Nothing close enough found.';
    return;
  }

  const top = hits[0].score || 1;
  const frag = document.createDocumentFragment();
  for (const hit of hits.slice(0, RESULT_LIMIT)) {
    const r = records[hit.index];
    const li = document.createElement('li');
    li.className = 'result';

    const head = document.createElement('div');
    head.className = 'result-head';
    const name = document.createElement('span');
    name.className = 'name';
    name.textContent = r.name;
    const unit = document.createElement('span');
    unit.className = 'unit';
    unit.textContent = r.unit;
    head.append(name, unit);

    const meta = document.createElement('div');
    meta.className = 'meta';
    const code = document.createElement('code');
    code.textContent = r.code;
    const path = document.createElement('span');
    path.className = 'path';
    path.textContent = r.path.join(' › ');
    meta.append(code, path);

    const bar = document.createElement('div');
    bar.className = 'bar';
    bar.title = `Relative match ${Math.round((hit.score / top) * 100)}%`;
    const fill = document.createElement('span');
    fill.style.width = `${Math.max(4, (hit.score / top) * 100)}%`;
    bar.append(fill);

    li.append(head, meta, bar);
    frag.append(li);
  }
  resultsEl.append(frag);
}

function runSearch() {
  const query = input.value;
  const id = ++requestId;
  latestId = id;

  if (!query.trim()) {
    resultsEl.replaceChildren();
    statsEl.textContent = '';
    emptyEl.hidden = false;
    emptyEl.textContent = 'Type a search, or pick an example above.';
    return;
  }

  if (mode !== 'fuzzy' && !modelReady) {
    waitingForModel = true;
    statsEl.textContent = 'Waiting for the AI model to finish loading…';
    return;
  }

  send({ type: 'search', id, query, mode, limit: RESULT_LIMIT });
}

function scheduleSearch(delay = DEBOUNCE_MS) {
  window.clearTimeout(debounceTimer);
  debounceTimer = window.setTimeout(runSearch, delay);
}

input.addEventListener('input', () => scheduleSearch());

modeButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    mode = btn.dataset.mode as SearchMode;
    modeButtons.forEach((b) => b.setAttribute('aria-checked', String(b === btn)));
    scheduleSearch(0);
  });
});

document.querySelectorAll<HTMLButtonElement>('.chip').forEach((chip) => {
  chip.addEventListener('click', () => {
    input.value = chip.dataset.q ?? '';
    input.focus();
    scheduleSearch(0);
  });
});

send({ type: 'init', baseUrl: new URL(import.meta.env.BASE_URL, window.location.href).href });
emptyEl.hidden = false;
emptyEl.textContent = 'Type a search, or pick an example above.';
