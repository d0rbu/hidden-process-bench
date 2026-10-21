import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { categorical, filter, randomSource, sampleSequence, splitStates } from './process.mjs';

const data = JSON.parse(readFileSync(new URL('./data.json', import.meta.url), 'utf8'));

function close(actual, expected) {
  if (Array.isArray(expected)) {
    assert.equal(actual.length, expected.length);
    actual.forEach((value, index) => close(value, expected[index]));
  } else {
    assert.ok(Number.isFinite(actual));
    assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} != ${expected}`);
  }
}

for (const dataset of data.datasets) {
  test(`${dataset.id}: every browser belief and prediction matches the Python oracle`, () => {
    for (const check of dataset.oracle_checks) {
      const result = filter(dataset.hmm, check.tokens);
      close(result.predictions, check.predictions);
      close(result.beliefs, check.beliefs);
    }
  });

  test(`${dataset.id}: duplicated states preserve tokens, latent parents, and predictions`, () => {
    const split = splitStates(dataset.hmm);
    for (let component = 0; component < dataset.hmm.weights.length; component += 1) {
      const original = sampleSequence(dataset.hmm, component, 24, 20261021);
      const doubled = sampleSequence(split, component, 24, 20261021);
      assert.deepEqual(doubled.tokens, original.tokens);
      assert.deepEqual(doubled.states.map((state) => Math.floor(state / 2)), original.states);
      close(filter(split, original.tokens).predictions, filter(dataset.hmm, original.tokens).predictions);
      assert.deepEqual(sampleSequence(dataset.hmm, component, 8, 20261021).tokens, original.tokens.slice(0, 8));
    }
  });
}

test('sampling emits before transitioning and keeps the selected component fixed', () => {
  const alternating = {
    weights: [1], initial: [[1, 0]], transition: [[[0, 1], [1, 0]]], emission: [[[1, 0], [0, 1]]],
  };
  assert.deepEqual(sampleSequence(alternating, 0, 6, 4), { tokens: [0, 1, 0, 1, 0, 1], states: [0, 1, 0, 1, 0, 1] });
  const mixture = { weights: [.5, .5], initial: [[1], [1]], transition: [[[1]], [[1]]], emission: [[[1, 0]], [[0, 1]]] };
  assert.deepEqual(sampleSequence(mixture, 1, 24, 4).tokens, Array(24).fill(1));
  assert.deepEqual(filter(mixture, [1]).beliefs.at(-1), [[0], [1]]);
});

test('categorical boundaries skip zero mass and include the final bin', () => {
  assert.equal(categorical([0, .4, 0, .6], 0), 1);
  assert.equal(categorical([0, .4, 0, .6], .399999), 1);
  assert.equal(categorical([0, .4, 0, .6], .4), 3);
  assert.equal(categorical([0, .4, 0, .6], .999999), 3);
});

test('impossible histories and invalid tokens are not treated as valid predictions', () => {
  const hmm = { weights: [1], initial: [[1]], transition: [[[1]]], emission: [[[1, 0]]] };
  assert.throws(() => filter(hmm, [1]), /zero probability/);
  for (const token of [-1, 2, .5, NaN]) assert.throws(() => filter(hmm, [token]), /outside vocabulary/);
});

test('seeded browser samples are reproducible and uniforms stay inside the unit interval', () => {
  const first = randomSource(8);
  const second = randomSource(8);
  const other = randomSource(9);
  const values = Array.from({ length: 1000 }, () => first());
  assert.deepEqual(values, Array.from({ length: 1000 }, () => second()));
  assert.notEqual(values[0], other());
  assert.ok(values.every((value) => value >= 0 && value < 1));
});
