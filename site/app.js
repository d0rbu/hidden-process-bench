import { filter, sampleSequence, splitStates } from './process.mjs';

const $ = (id) => document.getElementById(id);
const svgNamespace = 'http://www.w3.org/2000/svg';
const percent = (p) => p > 0 && p < .001 ? '<0.1%' : `${(100 * p).toFixed(p > 0 && p < 0.1 ? 1 : 0)}%`;
const tokenLabel = (token) => String.fromCharCode(65 + token);
const number = new Intl.NumberFormat('en-US');
let data;
let dataset;
let component = 0;
let selectedState = 0;
let seed = 20261021;
let length = 8;
let sequence;

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function svgElement(tag, attributes, text) {
  const node = document.createElementNS(svgNamespace, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}

function duplicated() { return $('duplicate-states').checked; }
function currentHmm() { return duplicated() ? splitStates(dataset.hmm) : dataset.hmm; }
function stateLabel(state) { return duplicated() ? `${Math.floor(state / 2) + 1}${state % 2 ? 'b' : 'a'}` : String(state + 1); }

function renderBars(target, entries) {
  target.replaceChildren();
  for (const { label, probability, other = false } of entries) {
    const row = element('div', undefined, `bar-row${other ? ' other' : ''}`);
    row.append(element('span', label, 'bar-label'));
    const track = element('div', undefined, 'bar-track');
    track.setAttribute('aria-hidden', 'true');
    const fill = element('div', undefined, 'bar-fill');
    fill.style.width = `${100 * probability}%`;
    track.append(fill);
    row.append(track, element('span', percent(probability), 'bar-value'));
    row.title = `${label}: ${(probability * 100).toFixed(4)}%`;
    target.append(row);
  }
}

function topTokens(probabilities) {
  const ranked = probabilities.map((probability, token) => ({ label: tokenLabel(token), probability }));
  ranked.sort((a, b) => b.probability - a.probability);
  const top = ranked.slice(0, 5);
  top.push({ label: 'Other', probability: ranked.slice(5).reduce((sum, entry) => sum + entry.probability, 0), other: true });
  return top;
}

function renderComponents() {
  $('components').replaceChildren();
  dataset.hmm.weights.forEach((_, index) => {
    const button = element('button', `Process ${index + 1}`);
    button.type = 'button';
    button.setAttribute('aria-pressed', String(index === component));
    button.addEventListener('click', () => {
      component = index;
      selectedState = 0;
      newSequence();
      $('components').children[index].focus({ preventScroll: true });
    });
    $('components').append(button);
  });
}

function renderGraph() {
  const hmm = currentHmm();
  const count = hmm.initial[component].length;
  const svg = $('process-graph');
  const width = Math.max(380, svg.clientWidth);
  const center = { x: width / 2, y: 175 };
  const radius = count > 6 ? 110 : 100;
  const nodeRadius = count > 6 ? 19 : 27;
  const positions = Array.from({ length: count }, (_, index) => {
    const angle = index * 2 * Math.PI / count - Math.PI / 2;
    return { x: center.x + radius * Math.cos(angle), y: center.y + radius * Math.sin(angle) };
  });
  svg.setAttribute('viewBox', `0 0 ${width} 350`);
  svg.replaceChildren();
  svg.append(svgElement('title', {}, `Process ${component + 1}. ${count} states. Transitions from state ${stateLabel(selectedState)}.`));
  const defs = svgElement('defs', {});
  const marker = svgElement('marker', { id: 'arrow', viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse' });
  marker.append(svgElement('path', { d: 'M 0 0 L 8 4 L 0 8 z', fill: '#246b5e' }));
  defs.append(marker);
  svg.append(defs);
  const start = positions[selectedState];
  hmm.transition[component][selectedState].forEach((probability, index) => {
    if (probability === 0) return;
    let path;
    let labelX;
    let labelY;
    if (index === selectedState) {
      const ux = (start.x - center.x) / radius;
      const uy = (start.y - center.y) / radius;
      const tx = -uy;
      const ty = ux;
      const point = (out, side) => `${start.x + ux * out + tx * side},${start.y + uy * out + ty * side}`;
      path = `M ${point(nodeRadius * .7, nodeRadius * .7)} C ${point(72, 54)} ${point(72, -54)} ${point(nodeRadius * .7, -nodeRadius * .7)}`;
      labelX = start.x + ux * (count > 6 ? 55 : 62);
      labelY = start.y + uy * (count > 6 ? 55 : 62);
    } else {
      const end = positions[index];
      const dx = end.x - start.x;
      const dy = end.y - start.y;
      const distance = Math.hypot(dx, dy);
      const ux = dx / distance;
      const uy = dy / distance;
      const x1 = start.x + ux * (nodeRadius + 2);
      const y1 = start.y + uy * (nodeRadius + 2);
      const x2 = end.x - ux * (nodeRadius + 5);
      const y2 = end.y - uy * (nodeRadius + 5);
      const cx = (x1 + x2) / 2 - uy * 16;
      const cy = (y1 + y2) / 2 + ux * 16;
      path = `M ${x1},${y1} Q ${cx},${cy} ${x2},${y2}`;
      labelX = .25 * x1 + .5 * cx + .25 * x2 - uy * 7;
      labelY = .25 * y1 + .5 * cy + .25 * y2 + ux * 7;
      if (count > 6) {
        labelX = end.x + (end.x - center.x) / radius * 43;
        labelY = end.y + (end.y - center.y) / radius * 43;
      }
    }
    const edge = svgElement('path', { d: path, class: 'edge', 'stroke-width': 1.3 + 2.4 * probability, 'marker-end': 'url(#arrow)' });
    edge.append(svgElement('title', {}, `State ${stateLabel(selectedState)} to ${stateLabel(index)}: ${(100 * probability).toFixed(4)}%`));
    svg.append(edge, svgElement('text', { x: labelX, y: labelY, class: 'edge-label' }, percent(probability)));
  });
  positions.forEach((position, index) => {
    const node = svgElement('g', { class: 'state-node', role: 'button', tabindex: 0, 'aria-label': `State ${stateLabel(index)}`, 'aria-pressed': index === selectedState });
    node.append(svgElement('circle', { cx: position.x, cy: position.y, r: nodeRadius }));
    node.append(svgElement('text', { x: position.x, y: position.y }, stateLabel(index)));
    const select = () => {
      selectedState = index;
      renderGraph();
      renderEmissions();
      $('process-graph').querySelectorAll('.state-node')[index].focus({ preventScroll: true });
    };
    node.addEventListener('click', select);
    node.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        select();
      }
    });
    svg.append(node);
  });
}

function renderEmissions() {
  const hmm = currentHmm();
  const probabilities = hmm.emission[component][selectedState];
  $('emission-title').textContent = `Tokens from state ${stateLabel(selectedState)}`;
  renderBars($('emissions'), topTokens(probabilities));
  const table = element('table');
  const caption = element('caption', 'Token letters are aliases for IDs 0, 1, 2, …');
  const head = element('thead');
  const headRow = element('tr');
  for (const label of ['Token', 'Probability']) {
    const cell = element('th', label);
    cell.scope = 'col';
    headRow.append(cell);
  }
  head.append(headRow);
  const body = element('tbody');
  probabilities.forEach((p, token) => {
    const row = element('tr');
    row.append(element('td', tokenLabel(token)), element('td', `${(100 * p).toFixed(4)}%`));
    body.append(row);
  });
  table.append(caption, head, body);
  $('all-probabilities').replaceChildren(table);
}

function renderSequence() {
  const showStates = $('show-states').checked;
  $('sequence').replaceChildren();
  sequence.tokens.forEach((token, index) => {
    const item = element('span', undefined, 'token');
    item.append(element('span', tokenLabel(token), 'token-letter'));
    item.title = `Token ${index + 1}: ${tokenLabel(token)} (ID ${token})`;
    if (showStates) {
      const state = element('span', stateLabel(sequence.states[index]), 'token-state');
      state.setAttribute('aria-label', `Hidden state ${stateLabel(sequence.states[index])}`);
      item.append(state);
    }
    $('sequence').append(item);
  });
  $('sequence-status').textContent = `${length} / 24 tokens · Process ${component + 1}`;
  $('next-token').disabled = length >= 24;
}

function renderOracle() {
  const result = filter(currentHmm(), sequence.tokens);
  renderBars($('component-beliefs'), result.beliefs.at(-1).map((row, index) => ({ label: `Process ${index + 1}`, probability: row.reduce((sum, p) => sum + p, 0) })));
  renderBars($('next-prediction'), topTokens(result.predictions.at(-1)));
}

function renderExplorer() {
  const hmm = currentHmm();
  sequence = sampleSequence(hmm, component, length, seed);
  renderComponents();
  renderGraph();
  renderEmissions();
  renderSequence();
  renderOracle();
  const states = hmm.initial[0].length;
  const vocab = hmm.emission[0][0].length;
  $('process-size').textContent = `${states} states · ${vocab} tokens`;
  $('equivalence-status').textContent = duplicated()
    ? `${states * hmm.weights.length} joint states now, but observable rank is still ${dataset.complexity.short_hankel_rank}. The same random draws produce the same token sequence.`
    : `${dataset.complexity.joint_states} joint states; observable rank ${dataset.complexity.short_hankel_rank}. This example passes the numerical minimality screen.`;
  $('generator-source').href = dataset.report_url;
}

function newSequence() {
  seed = crypto.getRandomValues(new Uint32Array(1))[0];
  length = 8;
  renderExplorer();
}

function renderResults() {
  const demo = data.runs.find((run) => run.id === 'demo');
  $('demo-summary').textContent = `After ${number.format(demo.training_steps)} training steps, test KL was ${demo.test_kl.toFixed(4)} nats/token. All component checks passed the 0.03 threshold.`;
  const methods = [['samples', 'Samples'], ['probabilities', 'Token probabilities'], ['activations', 'Activations + probabilities']];
  const maximum = Math.max(...Object.values(demo.recovery).map((result) => result.kl));
  $('recovery-results').replaceChildren();
  for (const [key, name] of methods) {
    const row = element('tr', undefined, key === 'probabilities' ? 'best' : '');
    row.append(element('td', name));
    const metric = element('td');
    const inner = element('div', undefined, 'metric-cell');
    const track = element('span', undefined, 'metric-track');
    track.setAttribute('aria-hidden', 'true');
    const bar = element('span');
    bar.style.width = `${100 * demo.recovery[key].kl / maximum}%`;
    track.append(bar);
    inner.append(track, element('span', demo.recovery[key].kl.toFixed(4)));
    metric.append(inner);
    row.append(metric, element('td', demo.recovery[key].mae.toFixed(4)));
    $('recovery-results').append(row);
  }
  $('scaling-results').replaceChildren();
  for (const run of data.runs.filter((item) => item.id !== 'demo')) {
    const row = element('tr');
    for (const value of [run.states, run.width, run.test_kl.toFixed(4), run.gate_upper.toFixed(4)]) row.append(element('td', String(value)));
    const status = element('td');
    const link = element('a', run.qualified ? 'Yes' : 'No');
    link.href = run.report_url;
    link.setAttribute('aria-label', `${run.qualified ? 'Qualified' : 'Unqualified'}, ${run.states} states, width ${run.width}: raw report`);
    status.append(link);
    row.append(status);
    $('scaling-results').append(row);
  }
  renderScaling();
}

function renderStudy() {
  const checkpoint = data.method_study.cases.find((item) => item.id === $('study-case').value);
  const names = [
    ['adam_samples', 'Samples (Adam)'], ['adam_probabilities', 'Probabilities (control)'],
    ['em_samples', 'Baum–Welch / EM'], ['spectral_hmm', 'Spectral → HMM'],
    ['belief_layer_0', 'Belief geometry · block 1'], ['belief_layer_1', 'Belief geometry · block 2'],
    ['belief_all_layers', 'Belief geometry · both blocks'], ['pca', 'PCA'],
    ['sae', 'Sparse autoencoder'], ['transcoder', 'MLP transcoder'], ['circuit', 'Pruned-model HMM'],
    ['agent_static', 'Agent · static weights'], ['agent_observations', 'Agent · cached observations'],
  ];
  $('study-results').replaceChildren();
  for (const [key, name] of names) {
    const result = checkpoint.methods[key];
    if (!result) continue;
    const row = element('tr', undefined, key === 'adam_probabilities' ? 'best' : '');
    row.append(element('td', name), element('td', result.kl.toFixed(4)), element('td', result.mae.toFixed(4)));
    $('study-results').append(row);
  }
}

function initializeStudy() {
  const task = data.method_study.cases.find((item) => item.id === 'new-a');
  $('method-finding').textContent = `With the HMM family and size supplied, belief features reduced predictive KL from ${task.methods.adam_probabilities.kl.toFixed(4)} to ${task.methods.belief_layer_0.kl.toFixed(4)} on one task. This does not test discovering the HMM structure.`;
  for (const item of data.method_study.cases) {
    const option = element('option', item.label);
    option.value = item.id;
    $('study-case').append(option);
  }
  $('study-case').value = 'new-a';
  $('study-case').addEventListener('change', renderStudy);
  $('study-report').href = `${data.repository}/blob/main/docs/results/five-method-pilot.md`;
  renderStudy();
}

function renderScaling() {
  const svg = $('scaling-chart');
  const width = Math.max(330, svg.clientWidth);
  const height = 300;
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  // Preserve the accessible title and description.
  svg.querySelectorAll(':scope > :not(title):not(desc)').forEach((node) => node.remove());
  const left = 55;
  const right = width - 22;
  const x = (modelWidth) => modelWidth === 32 ? left + (right - left) * .2 : left + (right - left) * .78;
  const y = (value) => 248 - value / .075 * 205;
  svg.append(svgElement('text', { x: 0, y: 17, class: 'chart-label' }, 'Worst prediction bound (nats/token) ↓'));
  for (const tick of [.02, .04, .06]) {
    svg.append(svgElement('line', { x1: left, x2: right, y1: y(tick), y2: y(tick), stroke: '#e0e2d9' }));
    svg.append(svgElement('text', { x: left - 11, y: y(tick) + 4, 'text-anchor': 'end', class: 'chart-label' }, tick.toFixed(2)));
  }
  const gate = data.runs[0].threshold;
  svg.append(svgElement('line', { x1: left, x2: right, y1: y(gate), y2: y(gate), stroke: '#7a847a', 'stroke-dasharray': '5 5' }));
  svg.append(svgElement('text', { x: right, y: y(gate) - 9, 'text-anchor': 'end', class: 'chart-label' }, 'Pass ≤ 0.03'));
  for (const [states, color] of [[3, '#246b5e'], [6, '#ad5131']]) {
    const runs = data.runs.filter((run) => run.id !== 'demo' && run.states === states).sort((a, b) => a.width - b.width);
    svg.append(svgElement('path', { d: runs.map((run, index) => `${index ? 'L' : 'M'} ${x(run.width)},${y(run.gate_upper)}`).join(' '), fill: 'none', stroke: color, 'stroke-width': 2.5 }));
    for (const run of runs) {
      const point = svgElement('circle', { cx: x(run.width), cy: y(run.gate_upper), r: 5.5, fill: color });
      point.append(svgElement('title', {}, `${states} states, width ${run.width}: ${run.gate_upper.toFixed(6)}; ${run.qualified ? 'qualified' : 'unqualified'}`));
      svg.append(point);
      svg.append(svgElement('text', { x: x(run.width), y: y(run.gate_upper) + (states === 3 ? 23 : -13), 'text-anchor': 'middle', fill: color, class: 'point-label' }, run.gate_upper.toFixed(4)));
    }
  }
  for (const modelWidth of [32, 64]) svg.append(svgElement('text', { x: x(modelWidth), y: 270, 'text-anchor': 'middle', class: 'chart-label' }, `Width ${modelWidth}`));
  svg.append(svgElement('text', { x: (left + right) / 2, y: 294, 'text-anchor': 'middle', class: 'chart-label' }, 'Transformer size'));
}

function renderAgentAttempt() {
  const teacherCases = new Map(data.operator_attempt.cases.map((item) => [item.id, item]));
  const labels = { 'case-01': 'Delayed rule A', 'case-02': 'Delayed rule B', 'case-03': 'HMM mixture A', 'case-04': 'HMM mixture B', 'case-05': 'Stack rule' };
  for (const item of data.agent_attempt.cases) {
    const row = document.createElement('tr');
    const name = element('th', labels[item.id]);
    name.scope = 'row';
    row.append(name);
    row.append(element('td', teacherCases.get(item.id).heldout.oracle_to_teacher.mean_kl.toFixed(4)));
    row.append(element('td', item.heldout ? item.heldout.oracle_to_candidate.mean_kl.toFixed(4) : '—'));
    row.append(element('td', item.status === 'submitted' ? 'Six-regime filter' : 'Call timeout'));
    $('agent-results').append(row);
  }
}

function renderDiscoveryAttempts() {
  const labels = { 'case-01': 'Delayed rule A', 'case-02': 'Delayed rule B', 'case-03': 'HMM mixture A', 'case-04': 'HMM mixture B', 'case-05': 'Stack rule' };
  for (const [id, label] of Object.entries(labels)) {
    const row = document.createElement('tr');
    const name = element('th', label);
    name.scope = 'row';
    row.append(name);
    for (const method of ['geometry', 'pca', 'sae', 'transcoder', 'causal']) {
      const item = data.discovery_attempts.cases.find((entry) => entry.case === id && entry.method === method);
      row.append(element('td', item.heldout.oracle_to_candidate.mean_kl.toFixed(4)));
    }
    $('discovery-results').append(row);
  }
}

function renderOperatorAttempt() {
  const labels = { lag_rule: 'Delayed rule', mixture_hmm: 'HMM mixture', stack_rule: 'Stack rule' };
  const counts = {};
  for (const item of data.operator_attempt.cases) {
    const family = item.family_after_registration;
    counts[family] = (counts[family] || 0) + 1;
    const label = `${labels[family]}${family === 'stack_rule' ? '' : ` ${counts[family] === 1 ? 'A' : 'B'}`}`;
    const row = document.createElement('tr');
    const name = element('th', label);
    name.scope = 'row';
    row.append(name);
    for (const value of [item.heldout.oracle_to_teacher.mean_kl, item.heldout.oracle_to_candidate.mean_kl, item.long.oracle_to_candidate.mean_kl]) {
      row.append(element('td', value.toFixed(4)));
    }
    $('operator-results').append(row);
  }
  $('operator-budget').textContent = 'Each score uses 128 private histories: length 24 for the transformer and submitted program, and length 72 for the program alone. The teacher has a 24-token context. No neural calls were added during extraction.';
  $('attempts').hidden = false;
}

async function initialize() {
  try {
    const response = await fetch('data.json');
    if (!response.ok) throw new Error(`Data request failed (${response.status})`);
    data = await response.json();
    dataset = data.datasets[0];
    for (const item of data.datasets) {
      const option = element('option', item.label);
      option.value = item.id;
      $('dataset').append(option);
    }
    $('load-status').hidden = true;
    $('explore').hidden = false;
    $('results').hidden = false;
    $('dataset').addEventListener('change', () => {
      dataset = data.datasets.find((item) => item.id === $('dataset').value);
      $('duplicate-states').checked = false;
      selectedState = 0;
      newSequence();
    });
    $('new-sequence').addEventListener('click', newSequence);
    $('next-token').addEventListener('click', () => {
      length += 1;
      renderExplorer();
    });
    $('show-states').addEventListener('change', renderSequence);
    $('duplicate-states').addEventListener('change', () => {
      selectedState = duplicated() ? selectedState * 2 : Math.floor(selectedState / 2);
      renderExplorer();
    });
    window.addEventListener('resize', () => { renderGraph(); renderScaling(); });
    renderExplorer();
    renderResults();
    initializeStudy();
    renderAgentAttempt();
    renderDiscoveryAttempts();
    renderOperatorAttempt();
  } catch (error) {
    $('load-status').textContent = 'The interactive data could not load. Reload the page, or read the linked reports below.';
    console.error(error);
  }
}

initialize();
