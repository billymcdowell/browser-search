// Pure search logic shared by the worker. No DOM, no model code, so it is easy to test.

export interface PriceRecord {
  id: string;
  code: string;
  name: string;
  unit: string;
  path: string[];
}

export interface Hit {
  index: number;
  score: number;
}

/** Lower-case and strip accents so "Café" matches "cafe". */
export function normalize(text: string): string {
  return text.normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase();
}

/** Split normalized text into alphanumeric words. */
function words(text: string): string[] {
  return text.split(/[^a-z0-9]+/).filter(Boolean);
}

/** True when `a` and `b` differ by at most `max` single-character edits. */
function withinEdits(a: string, b: string, max: number): boolean {
  if (Math.abs(a.length - b.length) > max) return false;
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    let rowMin = i;
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      const v = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
      cur.push(v);
      if (v < rowMin) rowMin = v;
    }
    if (rowMin > max) return false;
    prev = cur;
  }
  return prev[b.length] <= max;
}

/** True when every character of `token` appears in `text`, in order. */
function isSubsequence(token: string, text: string): boolean {
  let t = 0;
  for (let i = 0; i < text.length && t < token.length; i++) {
    if (text[i] === token[t]) t++;
  }
  return t === token.length;
}

/**
 * Typo-tolerant keyword search. Every query word has to match something in a
 * record. Matches are graded: word prefix > inside a word > one or two typos >
 * letters in order.
 */
export class FuzzyIndex {
  private readonly haystacks: string[];
  private readonly wordLists: string[][];

  constructor(records: PriceRecord[]) {
    this.haystacks = records.map((r) =>
      normalize(`${r.name} ${r.path.join(' ')} ${r.code.replace(/\./g, ' ')} ${r.unit}`),
    );
    this.wordLists = this.haystacks.map(words);
  }

  private tokenScore(token: string, i: number): number {
    const hay = this.haystacks[i];
    const ws = this.wordLists[i];
    if (ws.some((w) => w.startsWith(token))) return 1;
    if (hay.includes(token)) return 0.8;
    if (token.length >= 4) {
      const maxEdits = token.length >= 7 ? 2 : 1;
      if (ws.some((w) => withinEdits(w, token, maxEdits))) return 0.6;
    }
    if (token.length >= 3 && isSubsequence(token, hay)) return 0.4;
    return 0;
  }

  search(query: string, limit: number): Hit[] {
    const tokens = words(normalize(query)).slice(0, 8);
    if (tokens.length === 0) return [];
    const phrase = tokens.join(' ');
    const hits: Hit[] = [];

    for (let i = 0; i < this.haystacks.length; i++) {
      let sum = 0;
      let ok = true;
      for (const token of tokens) {
        const s = this.tokenScore(token, i);
        if (s === 0) {
          ok = false;
          break;
        }
        sum += s;
      }
      if (!ok) continue;
      let score = sum / tokens.length;
      if (this.haystacks[i].includes(phrase)) score += 0.5;
      hits.push({ index: i, score });
    }

    hits.sort((a, b) => b.score - a.score || a.index - b.index);
    return hits.slice(0, limit);
  }
}

/**
 * Brute-force cosine search over int8-quantized, unit-normalized vectors.
 * Layout matches scripts/build-index.mjs: [count float32 scales][count*dim int8].
 */
export class VectorIndex {
  readonly count: number;
  readonly dim: number;
  private readonly scales: Float32Array;
  private readonly vectors: Int8Array;

  constructor(buffer: ArrayBuffer, count: number, dim: number) {
    this.count = count;
    this.dim = dim;
    this.scales = new Float32Array(buffer, 0, count);
    this.vectors = new Int8Array(buffer, count * 4, count * dim);
  }

  /** Returns the `limit` nearest records to a unit-length query vector. */
  search(query: Float32Array, limit: number): Hit[] {
    const { dim, count, scales, vectors } = this;
    const scores = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const base = i * dim;
      let dot = 0;
      for (let d = 0; d < dim; d++) dot += query[d] * vectors[base + d];
      scores[i] = (dot * scales[i]) / 127;
    }
    const order = Array.from({ length: count }, (_, i) => i);
    order.sort((a, b) => scores[b] - scores[a]);
    return order.slice(0, limit).map((index) => ({ index, score: scores[index] }));
  }
}

/** Reciprocal rank fusion: merges ranked lists without needing comparable scores. */
export function reciprocalRankFusion(lists: Hit[][], limit: number, k = 60): Hit[] {
  const combined = new Map<number, number>();
  for (const list of lists) {
    list.forEach((hit, rank) => {
      combined.set(hit.index, (combined.get(hit.index) ?? 0) + 1 / (k + rank + 1));
    });
  }
  return [...combined.entries()]
    .map(([index, score]) => ({ index, score }))
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}
