// pivot_metrics.js — turn an ncu --csv metrics file into one row per kernel.
// Usage: node profiling/pivot_metrics.js profiling/reports/Tesla_T4/ncu_gemm_metrics.csv
// Pivot ncu --csv metric rows into one line per kernel launch.
const fs = require('fs');
function parseLine(l) { const out = []; let cur = '', q = false;
  for (const ch of l) { if (ch === '"') q = !q; else if (ch === ',' && !q) { out.push(cur); cur = ''; } else cur += ch; }
  out.push(cur); return out; }
const short = {
  'gpu__time_duration.sum': 'us', 'sm__cycles_elapsed.avg.per_second': 'clkMHz',
  'sm__throughput.avg.pct_of_peak_sustained_elapsed': 'SM%', 'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed': 'Mem%',
  'dram__throughput.avg.pct_of_peak_sustained_elapsed': 'DRAM%', 'dram__bytes_read.sum': 'DRAMrdMB', 'dram__bytes_write.sum': 'DRAMwrMB',
  'lts__t_sector_hit_rate.pct': 'L2hit%', 'l1tex__t_sector_hit_rate.pct': 'L1hit%',
  'l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum': 'ldReq', 'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum': 'ldSec',
  'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum': 'bankLd', 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum': 'bankSt',
  'sm__warps_active.avg.pct_of_peak_sustained_active': 'occAch%', 'launch__registers_per_thread': 'reg',
  'launch__occupancy_limit_registers': 'limReg', 'launch__occupancy_limit_shared_mem': 'limSmem', 'launch__occupancy_limit_blocks': 'limBlk',
  'launch__waves_per_multiprocessor': 'waves' };
const file = process.argv[2];
const lines = fs.readFileSync(file, 'utf8').split('\n').filter(l => l.startsWith('"'));
const header = parseLine(lines[0]); const idx = n => header.indexOf(n);
const kernels = new Map();
for (const l of lines.slice(1)) { const f = parseLine(l);
  const id = f[idx('ID')]; if (!kernels.has(id)) kernels.set(id, { name: f[idx('Kernel Name')].replace(/\(.*$/, '').replace(/^void /, ''), block: f[idx('Block Size')], grid: f[idx('Grid Size')], m: {} });
  let v = parseFloat(f[idx('Metric Value')].replace(/,/g, '')); const name = f[idx('Metric Name')]; const unit = f[idx('Metric Unit')];
  if (name === 'gpu__time_duration.sum') v = (unit === 'msecond' || unit === 'ms') ? v * 1000 : (unit === 'nsecond' || unit === 'ns') ? v / 1000 : v;
  if (name.startsWith('dram__bytes')) v = unit === 'Gbyte' ? v * 1000 : unit === 'Kbyte' ? v / 1000 : unit === 'byte' ? v / 1e6 : v;
  if (name === 'sm__cycles_elapsed.avg.per_second') v = unit === 'Ghz' || unit === 'GHz' ? v * 1000 : unit === 'hz' || unit === 'Hz' ? v / 1e6 : v;
  kernels.get(id).m[short[name] || name] = v; }
const cols = ['us','clkMHz','SM%','Mem%','DRAM%','DRAMrdMB','DRAMwrMB','L1hit%','L2hit%','sec/req','bankLd','bankSt','occAch%','reg','limReg','limSmem','limBlk','waves'];
console.log(['id','kernel','block','grid',...cols].join(' | '));
for (const [id, k] of kernels) { const m = k.m; m['sec/req'] = m.ldReq ? m.ldSec / m.ldReq : NaN;
  console.log([id, k.name, k.block, k.grid, ...cols.map(c => m[c] === undefined || isNaN(m[c]) ? '-' : (Math.abs(m[c]) >= 100 ? m[c].toFixed(0) : m[c].toFixed(2)))].join(' | ')); }
