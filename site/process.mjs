// Node-emitting HMMs: emit from the current state, then transition.
export function categorical(probabilities, uniform) {
  let cumulative = 0;
  for (let index = 0; index < probabilities.length - 1; index += 1) {
    cumulative += probabilities[index];
    if (uniform < cumulative) return index;
  }
  return probabilities.length - 1;
}

export function randomSource(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = Math.imul(state ^ (state >>> 15), state | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export function sampleSequence(hmm, component, length, seed) {
  const random = randomSource(seed);
  let state = categorical(hmm.initial[component], random());
  const tokens = [];
  const states = [];
  for (let time = 0; time < length; time += 1) {
    states.push(state);
    tokens.push(categorical(hmm.emission[component][state], random()));
    state = categorical(hmm.transition[component][state], random());
  }
  return { tokens, states };
}

export function filter(hmm, tokens) {
  const vocab = hmm.emission[0][0].length;
  let belief = hmm.initial.map((row, component) => row.map((p) => p * hmm.weights[component]));
  const beliefs = [];
  const predictions = [];
  for (let time = 0; time <= tokens.length; time += 1) {
    const prediction = Array(vocab).fill(0);
    belief.forEach((row, component) => row.forEach((mass, state) => {
      hmm.emission[component][state].forEach((p, token) => {
        prediction[token] += mass * p;
      });
    }));
    predictions.push(prediction);
    beliefs.push(belief);
    if (time < tokens.length) {
      const token = tokens[time];
      if (!Number.isInteger(token) || token < 0 || token >= vocab) {
        throw new Error('Token outside vocabulary');
      }
      const evidence = prediction[token];
      if (!(evidence > 0)) throw new Error('History has zero probability');
      belief = belief.map((row, component) => row.map((_, next) =>
        row.reduce((sum, mass, state) => sum + mass * hmm.emission[component][state][token]
          * hmm.transition[component][state][next] / evidence, 0)));
    }
  }
  return { predictions, beliefs };
}

export function splitStates(hmm) {
  return {
    weights: [...hmm.weights],
    initial: hmm.initial.map((row) => row.flatMap((p) => [p / 2, p / 2])),
    transition: hmm.transition.map((matrix) => matrix.flatMap((row) => {
      const split = row.flatMap((p) => [p / 2, p / 2]);
      return [[...split], [...split]];
    })),
    emission: hmm.emission.map((matrix) => matrix.flatMap((row) => [[...row], [...row]])),
  };
}

export function additionTrace(base, digits, a, b) {
  const limit = base ** digits;
  if (![base, digits, a, b].every(Number.isInteger) || base < 2 || base > 10 || digits < 1 || digits > 6 || a < 0 || b < 0 || a >= limit || b >= limit) throw new RangeError('Invalid addition input');
  const rows = [];
  let carry = 0;
  for (let column = 0; column < digits; column += 1) {
    const x = Math.floor(a / base ** column) % base;
    const y = Math.floor(b / base ** column) % base;
    const total = x + y + carry;
    rows.push({ a: x, b: y, carry_in: carry, digit: total % base, carry_out: Math.floor(total / base) });
    carry = Math.floor(total / base);
  }
  return rows;
}
