// Plays a real game of Tetris against the real model.
//  - engine  = the exact engine block from index.html
//  - prompt  = the bridge's prompt (we POST to the bridge, same body as the page)
// So this measures what the page will actually do.
//
//   node tools/play.js [maxPieces]
const fs = require('fs');
const path = require('path');

const page = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
const m = page.match(/<script>([\s\S]*?)\/\* -+ state \*\//);
const E = new Function(m[1] + `
  return {COLS,ROWS,BASE,ROTS,emptyBoard,buildCandidates,place,clearLines,
          features,heuristicScore,gridStrings,LABELS};`)();

const BRIDGE = process.env.BRIDGE || 'http://127.0.0.1:18090';
const MAXP = parseInt(process.argv[2] || '150', 10);

async function ask(type, board, cands) {
  const f = E.features(board);
  const body = {
    repr: 'compact', mode: 'jev', piece: type,
    grid: E.gridStrings(board), heights: f.h, holes: null,
    cands: cands.map(c => ({ label: c.label, rot: c.rot, col: c.col,
                             L: c.out.L, H: c.out.H, T: c.out.T,
                             B: c.out.B, S: c.out.S })),
  };
  const t0 = Date.now();
  const r = await fetch(BRIDGE + '/api/decide', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const d = await r.json();
  return { d, ms: Date.now() - t0 };
}

(async () => {
  let bag = [], next = null;
  const types = Object.keys(E.BASE);
  const refill = () => { bag = types.slice(); for (let i = bag.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [bag[i], bag[j]] = [bag[j], bag[i]]; } };
  const pickNext = () => { if (!bag.length) refill(); return bag.pop(); };

  let board = E.emptyBoard(), lines = 0, pieces = 0, fallback = 0, agree = 0;
  const lat = [];
  let topOut = false, promptN = null;

  for (let n = 0; n < MAXP; n++) {
    if (!next) next = pickNext();
    const type = next; next = pickNext();

    const cands = E.buildCandidates(board, type, 8);
    if (!cands.length) { topOut = true; break; }

    const { d, ms } = await ask(type, board, cands);
    if (d.error) { console.log('ERROR from bridge:', d.error); process.exit(2); }
    lat.push(ms); if (promptN === null) promptN = d.prompt_n;

    const best = cands.reduce((a, b) => (b.score > a.score ? b : a));
    let idx = cands.findIndex(c => c.label === d.label);
    if (idx < 0) { idx = cands.indexOf(best); fallback++; }
    const pick = cands[idx];
    if (pick === best) agree++;

    const [nb, cl] = E.clearLines(E.place(board, pick.cells, pick.row, pick.col, 1));
    board = nb; lines += cl; pieces++;
    if (E.features(board).maxh >= 20) { topOut = true; break; }
    if (n % 25 === 24) {
      console.log(`  ...${pieces} pieces, ${lines} lines, last ${Math.round(ms)} ms`);
    }
  }

  const med = a => { const s = [...a].sort((x, y) => x - y); const k = s.length >> 1; return s.length % 2 ? s[k] : (s[k - 1] + s[k]) / 2; };
  const out = {
    pieces, lines, topOut,
    lines_per_100: +(lines / Math.max(pieces, 1) * 100).toFixed(1),
    ms_median: Math.round(med(lat)), ms_p95: Math.round([...lat].sort((a, b) => a - b)[Math.floor(lat.length * 0.95)] || 0),
    fallback, agree, agree_pct: +(agree / Math.max(pieces, 1) * 100).toFixed(1),
    prompt_n_first: promptN,
  };
  console.log('\nPLAY ' + JSON.stringify(out));
  process.exit(topOut && pieces < 20 ? 3 : 0);
})();
