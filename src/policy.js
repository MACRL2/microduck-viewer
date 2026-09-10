// Pure-JS inference for the microduck locomotion policy — the ORT replacement.
// Structure (from the ONNX): (obs-mean)/std -> 4x [Gemm + ELU(last is linear)].
// Weights come from policy.bin (raw <f4) + policy.meta.json. No onnxruntime.
export function loadPolicy(buf, meta) {
  const f32 = new Float32Array(buf);
  const blob = (name) => {
    const b = meta.blobs.find((x) => x.name === name);
    return f32.subarray(b.offset / 4, b.offset / 4 + b.count);
  };
  const mean = blob('mean'), std = blob('std');
  const layers = meta.order.map((L) => ({ W: blob(L.W), b: blob(L.b), out: L.shape[0], in: L.shape[1] }));
  const elu = (x) => (x > 0 ? x : Math.expm1(x));       // alpha=1

  return function forward(obs) {
    let x = new Float32Array(meta.in);
    for (let i = 0; i < meta.in; i++) x[i] = (obs[i] - mean[i]) / std[i];
    for (let l = 0; l < layers.length; l++) {
      const { W, b, out, in: nin } = layers[l];
      const y = new Float32Array(out);
      for (let o = 0; o < out; o++) {                    // Gemm transB: y = x·Wᵀ + b
        let s = b[o]; const base = o * nin;
        for (let i = 0; i < nin; i++) s += x[i] * W[base + i];
        y[o] = (l < layers.length - 1) ? elu(s) : s;     // last layer linear
      }
      x = y;
    }
    return x;
  };
}
