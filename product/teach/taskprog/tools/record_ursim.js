// Records URSim's screen (Move tab, 3D robot) while the take's program runs. Output: rec/*.webm
// Needs URSim running with port 6080 open, and Playwright. Usage: node record_ursim.js TASK CELL
const { chromium } = require('playwright');
const { spawn } = require('child_process');
(async () => {
  const b = await chromium.launch();
  const ctx = await b.newContext({ viewport: { width: 1300, height: 820 }, recordVideo: { dir: 'rec', size: { width: 1300, height: 820 } } });
  const p = await ctx.newPage();
  await p.goto('http://localhost:6080/vnc.html?autoconnect=true&resize=scale');
  await p.waitForTimeout(5000);
  await p.mouse.click(227, 35); await p.waitForTimeout(1500);
  // drag the 3D view to a side view
  await p.mouse.move(450, 300); await p.mouse.down();
  for (let i = 1; i <= 20; i++) { await p.mouse.move(450 + i * 6, 300 + i * 2); await p.waitForTimeout(40); }
  await p.mouse.up(); await p.waitForTimeout(2000);
  console.log('T0', Date.now());
  const t0 = Date.now();
  const py = spawn('python3', ['-m', 'taskprog', 'ursim', process.argv[2] || 'taskprog/examples/take1/task.json', '--cell', process.argv[3] || 'taskprog/cells/ur5e_table.json'], { cwd: require('path').resolve(__dirname, '..', '..') });
  let out = ''; py.stdout.on('data', d => out += d);
  await new Promise(r => py.on('close', r));
  console.log(out.split('\n').slice(-3).join('\n'), 'elapsed', Date.now() - t0);
  await p.waitForTimeout(2000);
  await p.screenshot({ path: 'end.png' });
  await ctx.close(); await b.close();
})();
