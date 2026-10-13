// Headless simulation of the REAL engine in index.html.
// It extracts the pure engine block from the page and plays games with several
// policies, so we can tell "engine is broken" from "policy is bad".
//
//   node tools/sim.js [games] [pieces]
const fs = require('fs');
const path = require('path');

const page = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
const m = page.match(/<script>([\s\S]*?)\/\* -+ state \*\//);
if (!m) { console.error('engine block not found'); process.exit(1); }
const engineSrc = m[1];

const sandbox = {};
const fn = new Function(engineSrc + `
  return {COLS,ROWS,BASE,ROTS,emptyBoard,enumerate,buildCandidates,place,clearLines,
          features,heuristicScore,landing,fits,LABELS};
`);
const E = fn();

function playOne(policy, maxPieces, rng) {
  let bag = [], next = null;
  const all = Object.keys(E.BASE);
  const refill = () => { bag = all.slice(); for (let i = bag.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [bag[i], bag[j]] = [bag[j], bag[i]]; } };
  let board = E.emptyBoard(), lines = 0, pieces = 0, topOut = false;

  for (let n = 0; n < maxPieces; n++) {
    if (!next) { if (!bag.length) refill(); next = bag.pop(); }
    const type = next;
    if (!bag.length) refill();
    next = bag.pop();

    const cands = E.buildCandidates(board, type, 8);
    if (!cands.length) { topOut = true; break; }

    const pick = policy(cands, rng);
    const [nb, cl] = E.clearLines(E.place(board, pick.cells, pick.row, pick.col, 1));
    board = nb; lines += cl; pieces++;
    const h = E.features(board);
    if (h.maxh >= 20) { topOut = true; break; }
  }
  return { lines, pieces, topOut, maxh: E.features(board).maxh };
}

const rng = (() => { let s = 123456789; return () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; }; })();

const key = p => [p.out.L, -p.out.H, -p.out.T];
const better = (a, b) => { const A = key(a), B = key(b);
  for (let i = 0; i < 3; i++) if (A[i] !== B[i]) return A[i] > B[i]; return false; };

const policies = {
  'heuristic-best  ': (c) => c.reduce((a, b) => (b.score > a.score ? b : a)),
  'heuristic-worst ': (c) => c.reduce((a, b) => (b.score < a.score ? b : a)),
  'uniform in 8    ': (c, r) => c[Math.floor(r() * c.length)],
  'rule L>H>T      ': (c) => c.reduce((a, b) => (better(b, a) ? b : a)),
  'rule 3L-2H-T    ': (c) => c.reduce((a, b) =>
    (b.out.L * 3 - b.out.H * 2 - b.out.T > a.out.L * 3 - a.out.H * 2 - a.out.T ? b : a)),
};

const games = parseInt(process.argv[2] || '40', 10);
const pieces = parseInt(process.argv[3] || '400', 10);

console.log(`engine from index.html | ${games} games x up to ${pieces} pieces\n`);
for (const [name, pol] of Object.entries(policies)) {
  const rs = [];
  for (let g = 0; g < games; g++) rs.push(playOne(pol, pieces, rng));
  const avg = k => (rs.reduce((s, r) => s + r[k], 0) / rs.length).toFixed(1);
  const outs = rs.filter(r => r.topOut).length;
  console.log(`${name} lines/game ${String(avg('lines')).padStart(6)}   pieces ${String(avg('pieces')).padStart(6)}   max-height ${String(avg('maxh')).padStart(5)}   topped-out ${outs}/${games}`);
}
