// Small numerical builder double: exercises emitted signatures and dtype
// agreement, not a substitute for execution on a WebNN implementation.
import assert from 'node:assert/strict';

function roundHalf(value) {
  if (!Number.isFinite(value) || value === 0) return value;
  const magnitude = Math.abs(value);
  const step = 2 ** Math.max(-24, Math.floor(Math.log2(magnitude)) - 10);
  const scaled = magnitude / step;
  const lower = Math.floor(scaled);
  const rounded = scaled - lower === 0.5 ? lower + (lower % 2) : Math.round(scaled);
  const result = rounded * step;
  return Math.sign(value) * (result > 65504 ? Infinity : result);
}

class MLGraphBuilder {
  constructor(data) { this.data = data; }
  input(name, descriptor) {
    const round = descriptor.dataType === 'float16' ? roundHalf : Math.fround;
    return {type: descriptor.dataType, data: this.data[name].map(round)};
  }
  constant(descriptor, bytes) {
    assert.equal(descriptor.dataType, 'float32');
    assert.deepEqual(descriptor.shape, []);
    return {type: 'float32', data: Array.from(new Float32Array(bytes))};
  }
  cast(input, type, options = {}) {
    assert.ok(type === 'float16' || type === 'float32');
    assert.deepEqual(options, {});
    return {type, data: input.data.map(type === 'float16' ? roundHalf : Math.fround)};
  }
  binary(a, b, operation) {
    assert.equal(a.type, b.type);
    assert.equal(a.type, 'float32');
    const length = Math.max(a.data.length, b.data.length);
    return {type: a.type, data: Array.from({length}, (_, i) =>
      Math.fround(operation(a.data[a.data.length === 1 ? 0 : i], b.data[b.data.length === 1 ? 0 : i])))};
  }
  mul(a, b) { return this.binary(a, b, (x, y) => x * y); }
  add(a, b) { return this.binary(a, b, (x, y) => x + y); }
  tanh(input) {
    return {type: input.type, data: input.data.map(x => Math.fround(Math.tanh(x)))};
  }
  async build(outputs) { return outputs; }
}
