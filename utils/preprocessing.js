const SEQ_LEN = 7680; // 60 seconds @ 128Hz (Match ONNX Model Input Shape [1, 2, 7680])

/**
 * Formats an input array to target length (SEQ_LEN = 7680) while preserving sampling rate & frequency.
 * For shorter signals (e.g. 3840), repeats/tiles the signal to maintain exact R-R beat frequency.
 */
function resampleToSeqLen(signal, targetLen = SEQ_LEN) {
  if (!signal || signal.length === 0) return new Float32Array(targetLen);
  if (signal.length === targetLen) return new Float32Array(signal);
  
  const result = new Float32Array(targetLen);
  const origLen = signal.length;
  
  if (origLen >= 21000) {
    // If raw 700Hz signal (21000 samples for 30s), downsample by picking every 21000/7680 step
    const step = origLen / targetLen;
    for (let i = 0; i < targetLen; i++) {
      const idx = Math.min(Math.floor(i * step), origLen - 1);
      result[i] = signal[idx];
    }
    return result;
  }

  // For shorter signals (e.g. 3840), tile/repeat to preserve true physical beat frequency (BPM)
  for (let i = 0; i < targetLen; i++) {
    result[i] = signal[i % origLen];
  }
  return result;
}

export function zScoreNormalize(signal) {
  if (!signal || signal.length === 0) return new Float32Array(SEQ_LEN);
  
  const resampled = resampleToSeqLen(signal, SEQ_LEN);
  
  // Check if signal is all zero or missing
  let sum = 0;
  let isAllZero = true;
  for (let i = 0; i < resampled.length; i++) {
    sum += resampled[i];
    if (Math.abs(resampled[i]) > 1e-6) isAllZero = false;
  }

  if (isAllZero) return new Float32Array(SEQ_LEN);

  const mean = sum / resampled.length;
  let sqDiffSum = 0;
  for (let i = 0; i < resampled.length; i++) {
    sqDiffSum += Math.pow(resampled[i] - mean, 2);
  }
  const std = Math.sqrt(sqDiffSum / resampled.length);
  const eps = 1e-8;

  if (std < eps) return new Float32Array(SEQ_LEN);

  const result = new Float32Array(SEQ_LEN);
  for (let i = 0; i < resampled.length; i++) {
    result[i] = (resampled[i] - mean) / (std + eps);
  }
  return result;
}

export function preprocessWindow(rawEcg, rawEda) {
  const ecg = zScoreNormalize(rawEcg);
  const eda = zScoreNormalize(rawEda);
  return { ecg, eda };
}

