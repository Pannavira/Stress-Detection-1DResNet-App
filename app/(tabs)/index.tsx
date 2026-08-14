import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ScrollView, ActivityIndicator, Dimensions } from 'react-native';
import { Asset } from 'expo-asset';
import { preprocessWindow } from '../../utils/preprocessing';
import dummyData from '../../assets/dummy_data.json';

let ort: any = null;
const screenWidth = Dimensions.get('window').width;

export default function StressScreen() {
  const [session, setSession] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number>(0);
  const [modalityMode, setModalityMode] = useState<string>('DUAL');
  const [logs, setLogs] = useState<string[]>([]);

  const LOGIT_GAIN = 1.0;

  const addLog = (msg: string) => {
    const time = new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    setLogs((prev) => [`[${time}] ${msg}`, ...prev].slice(0, 25));
  };

  useEffect(() => {
    initApp();
  }, []);

  const initApp = async () => {
    addLog('Initializing application...');
    try {
      ort = require('onnxruntime-react-native');
      addLog('ONNX Runtime module loaded.');
      setTimeout(loadModel, 500);
    } catch (e: any) {
      addLog('ONNX Runtime Error: ' + e.message);
    }
  };

  const loadModel = async () => {
    try {
      setLoading(true);
      addLog('Loading ResNet-1D model asset...');
      const modelAsset = Asset.fromModule(require('../../assets/model.onnx'));
      await modelAsset.downloadAsync();
      
      if (!modelAsset.localUri) {
         addLog('Error: Local URI not found.');
         return;
      }

      addLog('Creating Inference Session...');
      const sess = await ort.InferenceSession.create(modelAsset.localUri);
      setSession(sess);
      addLog('Model loaded successfully.');
    } catch (e: any) {
      addLog('Model Load Error: ' + e.message);
    } finally {
      setLoading(false);
    }
  };

  const softmax = (logits: number[]) => {
    const scaled = logits.map(l => l * LOGIT_GAIN);
    const maxLogit = Math.max(...scaled);
    const scores = scaled.map(l => Math.exp(l - maxLogit));
    const sumScores = scores.reduce((a, b) => a + b, 0);
    return scores.map(s => s / sumScores);
  };

  const runInference = async () => {
    if (!session || !ort) {
      addLog('Error: Model or ORT not ready.');
      return;
    }

    try {
      setLoading(true);
      addLog('Preprocessing window (7680 samples)...');

      const rawEcg = (dummyData as any).ecg || [];
      const rawEda = (dummyData as any).eda || [];

      const { ecg, eda } = preprocessWindow(rawEcg, rawEda);

      const inputData = new Float32Array(2 * 7680);
      inputData.set(ecg, 0);
      inputData.set(eda, 7680);

      const inputTensor = new ort.Tensor('float32', inputData, [1, 2, 7680]);

      addLog('Running ONNX inference...');
      const startTime = Date.now();
      const outputs = await session.run({ input: inputTensor });
      const duration = Date.now() - startTime;

      const rawLogits = Array.from(outputs.output.data as Float32Array); 
      const probs = softmax(rawLogits);
      const isStress = rawLogits[1] > rawLogits[0];
      
      setResult(isStress ? 'STRESS' : 'NON-STRESS');
      setConfidence(probs[isStress ? 1 : 0] * 100);

      addLog(`Success: ${duration}ms`);
      addLog(`Logits: [L0: ${rawLogits[0].toFixed(3)}, L1: ${rawLogits[1].toFixed(3)}]`);
    } catch (e: any) {
      addLog('Inference Error: ' + e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Stress Detection AI</Text>
      
      <View style={styles.statusCard}>
        <Text style={styles.statusLabel}>Classification Result:</Text>
        <Text style={[styles.resultText, result === 'STRESS' ? styles.stress : styles.normal]}>
          {result || 'READY'}
        </Text>
        {confidence > 0 && (
          <Text style={styles.confidenceText}>Confidence: {confidence.toFixed(1)}%</Text>
        )}
      </View>

      <TouchableOpacity 
        style={[styles.button, (loading || !session) && styles.buttonDisabled]} 
        onPress={runInference}
        disabled={loading || !session}
      >
        {loading ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.buttonText}>RUN RESNET DETECTION</Text>
        )}
      </TouchableOpacity>

      <View style={styles.logContainer}>
        <Text style={styles.logTitle}>System Logs:</Text>
        <ScrollView style={styles.logScroll}>
          {logs.map((log, i) => (
            <Text key={i} style={styles.logText}>• {log}</Text>
          ))}
        </ScrollView>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f5f5f5', alignItems: 'center', paddingTop: 60, paddingHorizontal: 20 },
  title: { fontSize: 26, fontWeight: 'bold', color: '#333', marginBottom: 20 },
  statusCard: { width: '100%', backgroundColor: '#fff', borderRadius: 15, padding: 20, alignItems: 'center', elevation: 4, marginBottom: 20 },
  statusLabel: { fontSize: 14, color: '#666' },
  resultText: { fontSize: 36, fontWeight: 'bold', marginVertical: 5 },
  confidenceText: { fontSize: 14, color: '#7f8c8d', fontWeight: '600' },
  normal: { color: '#4CAF50' },
  stress: { color: '#F44336' },
  button: { backgroundColor: '#2196F3', paddingVertical: 14, paddingHorizontal: 30, borderRadius: 25 },
  buttonDisabled: { backgroundColor: '#90CAF9' },
  buttonText: { color: '#fff', fontSize: 15, fontWeight: 'bold' },
  logContainer: { flex: 1, width: '100%', marginTop: 20 },
  logTitle: { fontSize: 12, fontWeight: 'bold', color: '#999', marginBottom: 5 },
  logScroll: { flex: 1, backgroundColor: '#eee', borderRadius: 8, padding: 10 },
  logText: { fontSize: 11, color: '#333', marginBottom: 2 },
});
