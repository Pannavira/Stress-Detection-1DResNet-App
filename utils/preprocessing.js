const SEQ_LEN = 7680; // 60 seconds @ 128Hz (Match Training)

export function zScoreNormalize(signal) {
  if (!signal || signal.length === 0) return new Float32Array(SEQ_LEN);
  const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
  let sqDiffSum = 0;
  for (let i = 0; i < signal.length; i++) sqDiffSum += Math.pow(signal[i] - mean, 2);
  const std = Math.sqrt(sqDiffSum / signal.length);
  const result = new Float32Array(signal.length);
  const eps = 1e-8; 
  for (let i = 0; i < signal.length; i++) result[i] = (signal[i] - mean) / (std + eps);
  return result;
}

export function preprocessWindow(rawEcg, rawEda) {
  // Use 7680 samples for a full 60-second window
  const ecg = new Float32Array(rawEcg).slice(0, SEQ_LEN);
  const eda = new Float32Array(rawEda).slice(0, SEQ_LEN);

  return { ecg: zScoreNormalize(ecg), eda: zScoreNormalize(eda) };
}
