#!/usr/bin/env node
// =============================================================================
// make_charts.js — regenerate the README charts (assets/*.svg) from measured data.
//
//   node assets/make_charts.js [benchmarks/<GPU>] [profiling/reports/<GPU>] [output dir]
//   (defaults: the Tesla_T4 results, the profiling folder with the same GPU name, assets/)
//
// Every bar and point is read from the CSV files written by the benchmarks
// (src/benchmark/csv_log.h, python/benchmark.py) and by Nsight Compute
// (profiling/ncu_kernels.sh). Nothing is typed in by hand, so the charts can be
// regenerated after a new run and always match the data.
//
// Design rules (applied to every chart):
//   - colors keep one meaning everywhere: blue = our optimized kernel,
//     orange = PyTorch / cuBLAS, aqua = the comparison case (naive or alternative);
//     palette validated for colorblind separation and contrast, light and dark;
//   - every series is direct-labeled (aqua is < 3:1 on the light surface);
//   - thin marks, rounded data ends, recessive grid, one y-axis per chart;
//   - light/dark follow the viewer's color scheme (prefers-color-scheme).
// =============================================================================
'use strict';
const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const benchDir = path.resolve(process.argv[2] || path.join(root, 'benchmarks', 'Tesla_T4'));
const profDir = path.resolve(process.argv[3] || path.join(root, 'profiling', 'reports', path.basename(benchDir)));
const outDir = path.resolve(process.argv[4] || __dirname);  // e.g. assets/A100 for a second GPU
fs.mkdirSync(outDir, { recursive: true });

// -----------------------------------------------------------------------------
// Data loading
// -----------------------------------------------------------------------------
function parseCsvLine(line) {
  const out = [];
  let cur = '';
  let quoted = false;
  for (const ch of line) {
    if (ch === '"') quoted = !quoted;
    else if (ch === ',' && !quoted) { out.push(cur); cur = ''; }
    else cur += ch;
  }
  out.push(cur);
  return out;
}

// Rows as objects. `headerStart` skips tool output before the header (ncu prints "==PROF==" lines).
function readCsv(file, headerStart) {
  const lines = fs.readFileSync(file, 'utf8').split(/\r?\n/).filter((l) => l.trim() !== '');
  const h = headerStart ? lines.findIndex((l) => l.startsWith(headerStart)) : 0;
  const header = parseCsvLine(lines[h]);
  return lines.slice(h + 1).map((l) => {
    const f = parseCsvLine(l);
    return Object.fromEntries(header.map((k, i) => [k, f[i]]));
  });
}

function pick(rows, criteria, column) {
  const row = rows.find((r) => Object.entries(criteria).every(([k, v]) => r[k] === v));
  if (!row) throw new Error(`no row matching ${JSON.stringify(criteria)}`);
  const v = parseFloat(String(row[column]).replace(/,/g, ''));
  if (Number.isNaN(v)) throw new Error(`no numeric ${column} in row ${JSON.stringify(criteria)}`);
  return v;
}

function noteValue(rows, criteria, key) {
  const row = rows.find((r) => Object.entries(criteria).every(([k, v]) => r[k] === v));
  if (!row) throw new Error(`no row matching ${JSON.stringify(criteria)}`);
  const m = new RegExp(`${key}=([0-9.eE+-]+)`).exec(row.note);
  if (!m) throw new Error(`no ${key} in note of ${JSON.stringify(criteria)}`);
  return parseFloat(m[1]);
}

const bench = (name) => readCsv(path.join(benchDir, `${name}.csv`));
const matmul = bench('matmul');
const vecadd = bench('vector_add');
const softmax = bench('softmax');
const layernorm = bench('layernorm');
const precision = bench('precision');
const attention = bench('attention');
const torch = bench('pytorch');
const ncuLayernorm = readCsv(path.join(profDir, 'ncu_layernorm_metrics.csv'), '"ID"');

// The GPU and its theoretical peak bandwidth come from this run, never from constants,
// so the same script works for a T4, A100 or H100 result folder.
const GPU = matmul[0].gpu;
const peakMatch = /Peak mem bandwidth\s*:\s*([0-9.]+)/.exec(fs.readFileSync(path.join(benchDir, 'vector_add.txt'), 'utf8'));
if (!peakMatch) throw new Error('peak bandwidth not found in vector_add.txt');
const PEAK_GBS = parseFloat(peakMatch[1]);

// Axis ranges scale with the data: a "nice" step (1, 2, 2.5 or 5 x 10^k) giving ~5 ticks.
function niceTicks(maxValue, target = 5) {
  const raw = maxValue / target;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const top = Math.ceil(maxValue / step) * step;
  const ticks = [];
  for (let t = 0; t <= top + step / 2; t += step) ticks.push(Math.round(t * 1e6) / 1e6);
  return { max: top, ticks };
}
// Decades covering [min, max] for log axes.
function decades(minValue, maxValue) {
  const lo = Math.floor(Math.log10(minValue));
  const hi = Math.ceil(Math.log10(maxValue));
  const ticks = [];
  for (let e = lo; e <= hi; e++) ticks.push(10 ** e);
  return { min: 10 ** lo, max: 10 ** hi, ticks };
}
const fmtNum = (t) => (t >= 1000 ? fmtInt(t) : `${t}`);

// -----------------------------------------------------------------------------
// Theme and SVG helpers
// -----------------------------------------------------------------------------
const FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif';
const LIGHT = { surface: '#fcfcfb', border: '#e6e5e0', text: '#0b0b0b', text2: '#52514e', grid: '#ebeae6',
  base: '#b9b8b2', s1: '#2a78d6', s2: '#eb6834', s3: '#1baf7a' };
const DARK = { surface: '#1a1a19', border: '#333331', text: '#ffffff', text2: '#c3c2b7', grid: '#2e2e2c',
  base: '#5c5b57', s1: '#3987e5', s2: '#d95926', s3: '#199e70' };

function themeRules(c) {
  return `
    .bg { fill: ${c.surface}; stroke: ${c.border}; }
    .title, .label, .value { fill: ${c.text}; }
    .sub, .note, .tick { fill: ${c.text2}; }
    .grid { stroke: ${c.grid}; }
    .base { stroke: ${c.base}; }
    .ref { stroke: ${c.text2}; }
    .mark { stroke: ${c.text}; }
    .s1 { fill: ${c.s1}; } .s2 { fill: ${c.s2}; } .s3 { fill: ${c.s3}; }
    .l1 { stroke: ${c.s1}; } .l2 { stroke: ${c.s2}; } .l3 { stroke: ${c.s3}; }
    .ring { stroke: ${c.surface}; }`;
}

function svgStart(w, h, title, desc) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-labelledby="t d">
  <title id="t">${esc(title)}</title>
  <desc id="d">${esc(desc)}</desc>
  <style>
    .title { font: 600 17px ${FONT}; }
    .sub   { font: 400 13px ${FONT}; }
    .label { font: 500 13px ${FONT}; }
    .value { font: 600 13px ${FONT}; font-variant-numeric: tabular-nums; }
    .note  { font: 400 12px ${FONT}; }
    .tick  { font: 400 11px ${FONT}; font-variant-numeric: tabular-nums; }
    .grid, .base, .ref, .mark { stroke-width: 1; fill: none; }
    .ref { stroke-dasharray: 4 3; }
    .mark { stroke-width: 2; }
    .line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
    .ring { stroke-width: 2; }${themeRules(LIGHT)}
    @media (prefers-color-scheme: dark) {${themeRules(DARK)}
    }
  </style>
  <rect class="bg" x="0.5" y="0.5" width="${w - 1}" height="${h - 1}" rx="12"/>
`;
}

function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
const r1 = (v) => Math.round(v * 10) / 10;
const text = (cls, x, y, s, anchor = 'start') =>
  `  <text class="${cls}" x="${r1(x)}" y="${r1(y)}"${anchor !== 'start' ? ` text-anchor="${anchor}"` : ''}>${esc(s)}</text>\n`;
const line = (cls, x1, y1, x2, y2) =>
  `  <line class="${cls}" x1="${r1(x1)}" y1="${r1(y1)}" x2="${r1(x2)}" y2="${r1(y2)}"/>\n`;
const approxWidth = (s, px) => s.length * px * 0.58;  // rough text width, for layout only

function header(title, sub) {
  return text('title', 24, 36, title) + text('sub', 24, 58, sub);
}

function legend(items, y = 74) {
  let x = 24;
  let out = '';
  for (const it of items) {
    out += `  <rect class="${it.cls}" x="${x}" y="${y}" width="12" height="12" rx="2"/>\n`;
    out += text('note', x + 18, y + 10, it.label);
    x += 18 + approxWidth(it.label, 12) + 28;
  }
  return out;
}

// Horizontal bar with a 4px rounded data end and a square baseline.
function bar(cls, x0, y, w, h) {
  if (w < 8) return `  <rect class="${cls}" x="${r1(x0)}" y="${r1(y)}" width="${r1(Math.max(w, 2))}" height="${h}"/>\n`;
  const x1 = x0 + w;
  return `  <path class="${cls}" d="M${r1(x0)} ${r1(y)} H${r1(x1 - 4)} Q${r1(x1)} ${r1(y)} ${r1(x1)} ${r1(y + 4)} V${r1(y + h - 4)} Q${r1(x1)} ${r1(y + h)} ${r1(x1 - 4)} ${r1(y + h)} H${r1(x0)} Z"/>\n`;
}

const fmtInt = (v) => Math.round(v).toLocaleString('en-US');

// -----------------------------------------------------------------------------
// Chart: horizontal (grouped) bars
// -----------------------------------------------------------------------------
function barChart(spec) {
  const W = 800;
  const labelX = spec.labelW;            // right edge of row labels
  const x0 = labelX + 12;                // baseline
  const plotW = spec.plotW;
  const sx = (v) => x0 + (v / spec.xmax) * plotW;
  const barH = spec.barH || 22;
  const top = spec.legend ? 110 : 84;
  let y = top;
  let body = '';
  const rowsOut = [];
  for (const row of spec.rows) {
    const groupH = row.bars.length * barH + (row.bars.length - 1) * 2;  // 2px surface gap between bars
    rowsOut.push({ row, y, groupH });
    y += groupH + (spec.groupGap ?? 16);
  }
  const plotBottom = y - (spec.groupGap ?? 16) + 6;
  // grid + ticks
  for (const t of spec.ticks) {
    body += line(t === 0 ? 'base' : 'grid', sx(t), top - 6, sx(t), plotBottom);
    body += text('tick', sx(t), plotBottom + 20, spec.tickFmt(t), 'middle');
  }
  if (spec.axisLabel) body += text('tick', x0 + plotW, plotBottom + 36, spec.axisLabel, 'end');
  // reference line
  if (spec.ref) {
    body += line('ref', sx(spec.ref.value), top - 14, sx(spec.ref.value), plotBottom);
    body += text('note', sx(spec.ref.value) + 6, top - 4, spec.ref.label);
  }
  // bars
  for (const { row, y: gy, groupH } of rowsOut) {
    body += text('label', labelX, gy + groupH / 2 + 4.5, row.label, 'end');
    row.bars.forEach((b, i) => {
      const by = gy + i * (barH + 2);
      const w = sx(b.value) - x0;
      body += bar(b.cls, x0, by, w, barH);
      for (const m of b.markers || []) {
        body += line('mark', sx(m.value), by - 4, sx(m.value), by + barH + 4);
      }
      const vx = Math.max(sx(b.value), ...(b.markers || []).map((m) => sx(m.value))) + 8;
      body += text('value', vx, by + barH / 2 + 4.5, b.valueText);
      // Notes go right after the value, or into a fixed column (spec.noteX) when a
      // reference line would otherwise cut through them.
      const nx = spec.noteX ?? vx + approxWidth(b.valueText, 13) + 10;
      if (b.note) body += text('note', nx, by + barH / 2 + 4.5, b.note);
    });
  }
  const H = plotBottom + (spec.axisLabel ? 50 : 34);
  return svgStart(W, H, spec.title, spec.desc) + header(spec.title, spec.sub) +
    (spec.legend ? legend(spec.legend) : '') + body + '</svg>\n';
}

// -----------------------------------------------------------------------------
// Chart: lines (log2 x axis; linear or log10 y axis)
// -----------------------------------------------------------------------------
function lineChart(spec) {
  const W = 800;
  const H = spec.height || 380;
  const left = 84;
  const right = W - (spec.rightPad || 170);
  const top = 128;  // leaves room for the legend row and the y-axis title
  const bottom = H - 56;
  const lx = spec.xvals.map(Math.log2);
  const sx = (v) => left + ((Math.log2(v) - lx[0]) / (lx[lx.length - 1] - lx[0])) * (right - left);
  const ymap = spec.y.log ? (v) => Math.log10(v) : (v) => v;
  const y0 = ymap(spec.y.min);
  const y1 = ymap(spec.y.max);
  const sy = (v) => bottom - ((ymap(v) - y0) / (y1 - y0)) * (bottom - top);

  let body = '';
  for (const t of spec.y.ticks) {
    body += line(t === spec.y.min ? 'base' : 'grid', left, sy(t), right, sy(t));
    body += text('tick', left - 10, sy(t) + 4, spec.y.fmt(t), 'end');
  }
  body += text('tick', left - 10, top - 14, spec.y.label, 'end');
  spec.xvals.forEach((v, i) => {
    body += text('tick', sx(v), bottom + 20, spec.xlabels[i], 'middle');
  });
  body += text('tick', right, bottom + 38, spec.xLabel, 'end');

  for (const a of spec.brackets || []) {
    const x = sx(a.x);
    body += line('ref', x, sy(a.from), x, sy(a.to));
    body += text('value', x + (a.dx ?? 10), (sy(a.from) + sy(a.to)) / 2 + 4.5, a.text, a.anchor || 'start');
  }
  for (const a of spec.notes || []) {
    body += text('note', a.px ?? sx(a.x), a.py ?? sy(a.y), a.text, a.anchor || 'start');
  }

  // series: line, then markers with a 2px surface ring
  const ends = [];
  spec.series.forEach((s) => {
    const pts = s.values.map((v, i) => [sx(spec.xvals[i]), sy(v)]);
    body += `  <polyline class="line ${s.line}" points="${pts.map(([x, yy]) => `${r1(x)},${r1(yy)}`).join(' ')}"/>\n`;
    for (const [x, yy] of pts) body += `  <circle class="${s.fill} ring" cx="${r1(x)}" cy="${r1(yy)}" r="4.5"/>\n`;
    ends.push({ y: pts[pts.length - 1][1], x: pts[pts.length - 1][0], label: s.endLabel });
  });
  // direct end labels, pushed apart if they would overlap
  ends.sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) {
    if (ends[i].y - ends[i - 1].y < 18) ends[i].y = ends[i - 1].y + 18;
  }
  for (const e of ends) body += text('label', e.x + 12, e.y + 4.5, e.label);

  return svgStart(W, H, spec.title, spec.desc) + header(spec.title, spec.sub) +
    legend(spec.series.map((s) => ({ cls: s.fill, label: s.legend }))) + body + '</svg>\n';
}

function write(name, svg) {
  fs.writeFileSync(path.join(outDir, name), svg);
  console.log(`wrote ${path.relative(root, path.join(outDir, name))}`);
}

const sup = (n) => String(n).replace('-', '⁻').replace(/[0-9]/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d]);

// -----------------------------------------------------------------------------
// 1. GEMM progression
// -----------------------------------------------------------------------------
{
  const shape = 'square 1024x1024x1024';
  const g = (impl) => pick(matmul, { experiment: 'A square', impl, shape }, 'gflops');
  const cpu = g('cpu (1 thread)');
  const v1 = g('v1 naive');
  const v2 = g('v2 coalesced');
  const v3 = g('v3 tiled-32');
  const v4 = g('v4 register 4x4');
  const cublas = pick(torch, { op: 'matmul', impl: 'cuBLAS fp32', shape }, 'gflops');
  const one = (label, cls, value, valueText, note) => ({ label, bars: [{ cls, value, valueText, note }] });
  write('gemm_progression.svg', barChart({
    title: `FP32 GEMM, 1024 × 1024 × 1024 on ${GPU}`,
    sub: 'GFLOP/s, higher is better · each version removes one measured bottleneck',
    desc: `GFLOP/s: CPU single thread ${cpu}; v1 naive ${v1}; v2 coalesced ${v2}; v3 shared-memory tiles ${v3}; v4 register tiles ${v4}; cuBLAS ${cublas}.`,
    legend: [{ cls: 's1', label: 'Our CUDA kernels' }, { cls: 's2', label: 'cuBLAS reference (PyTorch, TF32 off)' }],
    labelW: 210, plotW: 480, xmax: niceTicks(Math.max(cublas, v4) * 1.05).max, ticks: niceTicks(Math.max(cublas, v4) * 1.05).ticks, tickFmt: fmtNum, groupGap: 16,
    rows: [
      one('CPU, 1 thread', 's1', cpu, `${cpu}`),
      one('v1 naive', 's1', v1, fmtInt(v1)),
      one('v2 coalesced', 's1', v2, fmtInt(v2), `${(v2 / v1).toFixed(1)}× (coalescing)`),
      one('v3 shared-memory tiles', 's1', v3, fmtInt(v3), `${(v3 / v2).toFixed(1)}× (data reuse)`),
      one('v4 register tiles', 's1', v4, fmtInt(v4),
        `${(v4 / v3).toFixed(1)}× · ${(v4 / v1).toFixed(1)}× over v1 · ${Math.round((100 * v4) / cublas)}% of cuBLAS`),
      one('cuBLAS', 's2', cublas, fmtInt(cublas)),
    ],
  }));
}

// -----------------------------------------------------------------------------
// 2. Memory-bound kernels vs the bandwidth ceiling
// -----------------------------------------------------------------------------
{
  const peak = PEAK_GBS;  // the GPU's theoretical peak, from this run's device query
  const va = pick(vecadd, { experiment: 'A sizes', impl: 'naive', shape: 'n=67108864' }, 'gbs');
  const vaT = pick(torch, { op: 'vector_add', shape: 'n=67108864' }, 'gbs');
  const smV1 = pick(softmax, { experiment: 'A shapes', impl: 'v1 thread/row', shape: '12288x1024' }, 'gbs');
  const smBest = pick(softmax, { experiment: 'A shapes', impl: 'v2 block/row', shape: '12288x1024' }, 'gbs');
  const smT = pick(torch, { op: 'softmax', shape: '12288x1024' }, 'gbs');
  const lnV1 = pick(layernorm, { experiment: 'A shapes', impl: 'v1 thread/row', shape: '8192x4096' }, 'gbs');
  const lnBest = pick(layernorm, { experiment: 'A shapes', impl: 'v3 block/row regs', shape: '8192x4096' }, 'gbs');
  const lnT = pick(torch, { op: 'layernorm', shape: '8192x4096' }, 'gbs');
  const pct = (v) => `${Math.round((100 * v) / peak)}%`;
  const b = (cls, v, note) => ({ cls, value: v, valueText: `${r1(v)}`, note });
  write('memory_bandwidth.svg', barChart({
    title: 'Memory-bound kernels: achieved DRAM bandwidth',
    sub: `GB/s on ${GPU} (minimum bytes / kernel time) · dashed line = theoretical peak`,
    desc: `Vector add: ours ${va}, PyTorch ${vaT}. Softmax 12288x1024: naive ${smV1}, ours ${smBest}, PyTorch ${smT}. LayerNorm 8192x4096: naive ${lnV1}, ours ${lnBest}, PyTorch ${lnT}. Peak ${peak} GB/s.`,
    legend: [{ cls: 's3', label: 'Naive (thread per row)' }, { cls: 's1', label: 'Ours, optimized' },
      { cls: 's2', label: 'PyTorch' }],
    labelW: 190, plotW: 470, xmax: niceTicks(Math.max(peak, va, vaT, smBest, smT, lnBest, lnT) * 1.1).max, ticks: niceTicks(Math.max(peak, va, vaT, smBest, smT, lnBest, lnT) * 1.1).ticks, tickFmt: fmtNum,
    groupGap: 20, ref: { value: peak, label: `peak ${Math.round(peak)}` }, noteX: 700,
    rows: [
      { label: 'Vector add, 2²⁶', bars: [b('s1', va, pct(va)), b('s2', vaT, pct(vaT))] },
      { label: 'Softmax, 12288 × 1024', bars: [b('s3', smV1, pct(smV1)), b('s1', smBest, pct(smBest)), b('s2', smT, pct(smT))] },
      { label: 'LayerNorm, 8192 × 4096', bars: [b('s3', lnV1, pct(lnV1)), b('s1', lnBest, pct(lnBest)), b('s2', lnT, pct(lnT))] },
    ],
  }));
}

// -----------------------------------------------------------------------------
// 3. Kernel fusion measured in DRAM bytes (Nsight Compute)
// -----------------------------------------------------------------------------
{
  const elements = 8192 * 4096;
  const bytesOf = (id) => ['dram__bytes_read.sum', 'dram__bytes_write.sum']
    .reduce((s, m) => s + pick(ncuLayernorm, { ID: id, 'Metric Name': m }, 'Metric Value'), 0);
  // profile_targets layernorm order: 0 v2, 1 v3, 2+3 = unfused (vector_add, v3), 4 = fused
  const unfused = (bytesOf('2') + bytesOf('3')) / elements;
  const fused = bytesOf('4') / elements;
  const tUnfused = pick(layernorm, { experiment: 'B fusion', impl: 'unfused (vector_add + layernorm v3)', shape: '8192x4096' }, 'ms');
  const tFused = pick(layernorm, { experiment: 'B fusion', impl: 'fused add_layernorm', shape: '8192x4096' }, 'ms');
  write('fusion_bytes.svg', barChart({
    title: 'Kernel fusion, measured in DRAM bytes',
    sub: 'Residual add + LayerNorm, 8192 × 4096 · Nsight Compute DRAM counters · black tick = algorithmic minimum',
    desc: `Bytes per element: unfused ${unfused.toFixed(2)} (minimum 20), fused ${fused.toFixed(2)} (minimum 16). Time ${tUnfused} ms vs ${tFused} ms.`,
    labelW: 210, plotW: 400, xmax: 25, ticks: [0, 5, 10, 15, 20, 25], tickFmt: (t) => `${t}`, groupGap: 16,
    axisLabel: 'DRAM bytes per element',
    rows: [
      { label: 'Unfused (2 kernels)', bars: [{ cls: 's3', value: unfused, valueText: unfused.toFixed(2),
        markers: [{ value: 20 }], note: `${tUnfused.toFixed(2)} ms` }] },
      { label: 'Fused (1 kernel)', bars: [{ cls: 's1', value: fused, valueText: fused.toFixed(2),
        markers: [{ value: 16 }], note: `${tFused.toFixed(2)} ms · ${(tUnfused / tFused).toFixed(2)}× faster` }] },
    ],
  }));
}

// -----------------------------------------------------------------------------
// 4. Softmax: which design wins depends on the row length
// -----------------------------------------------------------------------------
{
  const shapes = [[524288, 32], [65536, 256], [16384, 1024], [4096, 4096], [1024, 16384], [256, 65536]];
  const gbs = (impl) => shapes.map(([r, c]) => pick(softmax, { experiment: 'B row length', impl, shape: `${r}x${c}` }, 'gbs'));
  const block = gbs('v2 block/row');
  const warp = gbs('v3 warp/row');
  write('softmax_row_length.svg', lineChart({
    title: 'Softmax: the best design depends on the row length',
    sub: `~16M elements per point, rows getting longer and fewer · GB/s on ${GPU}`,
    desc: `Columns 32..65536. Block per row: ${block.join(', ')} GB/s. Warp per row: ${warp.join(', ')} GB/s.`,
    xvals: shapes.map(([, c]) => c), xlabels: ['32', '256', '1K', '4K', '16K', '64K'], xLabel: 'columns per row',
    y: { min: 0, ...niceTicks(Math.max(...block, ...warp) * 1.1), fmt: fmtNum, label: 'GB/s' },
    series: [
      { legend: 'Block per row (shared-memory tree)', fill: 's1', line: 'l1', values: block, endLabel: 'block per row' },
      { legend: 'Warp per row (shuffles)', fill: 's3', line: 'l3', values: warp, endLabel: 'warp per row' },
    ],
  }));
}

// -----------------------------------------------------------------------------
// 5. Precision: FP16 vs FP32 accumulation error
// -----------------------------------------------------------------------------
{
  const ks = [64, 256, 1024, 4096, 16384];
  const err = (impl) => ks.map((k) => noteValue(precision, { experiment: 'B accuracy vs K', impl, shape: `64x64x${k}` }, 'arith_err'));
  const acc32 = err('fp16 tiled, fp32 acc');
  const acc16 = err('fp16 tiled, fp16 acc');
  const last = ks.length - 1;
  write('precision_error.svg', lineChart({
    title: 'Why accumulation stays in FP32',
    sub: `GEMM with FP16 inputs, 64 × 64 × K · max error / max |exact| (log scale) · ${GPU}`,
    desc: `K = ${ks.join(', ')}. FP32 accumulation error: ${acc32.join(', ')}. FP16 accumulation error: ${acc16.join(', ')}.`,
    xvals: ks, xlabels: ks.map(fmtInt), xLabel: 'K (length of each dot product)',
    y: { ...decades(Math.min(...acc32), Math.max(...acc16)), log: true,
      fmt: (t) => `10${sup(Math.round(Math.log10(t)))}`, label: 'error' },
    series: [
      { legend: 'FP16 accumulator', fill: 's3', line: 'l3', values: acc16, endLabel: 'FP16 accumulator' },
      { legend: 'FP32 accumulator', fill: 's1', line: 'l1', values: acc32, endLabel: 'FP32 accumulator' },
    ],
    brackets: [{ x: ks[last], from: acc32[last], to: acc16[last], dx: -10, anchor: 'end',
      text: `${fmtInt(Math.round(acc16[last] / acc32[last] / 100) * 100)}× more error` }],
  }));
}

// -----------------------------------------------------------------------------
// 6. KV cache: cost of generating one token
// -----------------------------------------------------------------------------
{
  const ctx = [128, 512, 1024, 2048];
  const ms = (impl) => ctx.map((c) => pick(attention, { experiment: 'B kv cache', impl, shape: `context=${c}` }, 'ms'));
  const cached = ms('decode with KV cache (q_len=1)');
  const full = ms('recompute all (q_len=L)');
  const last = ctx.length - 1;
  write('kv_cache.svg', lineChart({
    title: 'KV cache: cost of generating one token',
    sub: `Attention time for one decoding step, 12 heads, d = 64 · ms (log scale) · ${GPU}`,
    desc: `Context ${ctx.join(', ')}. With KV cache: ${cached.join(', ')} ms. Recomputing attention: ${full.join(', ')} ms.`,
    xvals: ctx, xlabels: ctx.map(fmtInt), xLabel: 'context length (tokens)',
    y: { ...decades(Math.min(...cached), Math.max(...full)), log: true, fmt: (t) => `${t}`, label: 'ms' },
    series: [
      { legend: 'Recompute attention for all tokens', fill: 's3', line: 'l3', values: full, endLabel: 'recompute all' },
      { legend: 'With KV cache (1 query)', fill: 's1', line: 'l1', values: cached, endLabel: 'with KV cache' },
    ],
    brackets: [
      { x: ctx[last], from: cached[last], to: full[last], dx: -10, anchor: 'end',
        text: `${Math.round(full[last] / cached[last])}× cheaper` },
    ],
  }));
}
