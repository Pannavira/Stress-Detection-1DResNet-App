import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ScrollView, ActivityIndicator } from 'react-native';
import { Asset } from 'expo-asset';
import { preprocessWindow } from '../../utils/preprocessing';
import dummyData from '../../assets/dummy_data.json';

// Dynamic import for ORT to prevent crash on app start
let ort: any = null;

export default function StressScreen() {
  const [session, setSession] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);

  const addLog = (msg: string) => {
    console.log(msg);
    setLogs((prev) => [msg, ...prev].slice(0, 20));
  };

  useEffect(() => {
    initApp();
  }, []);

  const initApp = async () => {
    addLog('Initializing application...');
    try {
      // Import the library only when the component is mounted
      ort = require('onnxruntime-react-native');
      addLog('ONNX Runtime module required.');
      
      // Give the native bridge a moment to stabilize
      setTimeout(loadModel, 1000);
    } catch (e: any) {
      addLog('ONNX Runtime Error: ' + e.message);
      console.error(e);
    }
  };

  const loadModel = async () => {
    try {
      setLoading(true);
      addLog('Loading model asset...');
      
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

  const runInference = async () => {
    if (!session || !ort) {
      addLog('Error: Model or ORT not ready.');
      return;
    }

    try {
      setLoading(true);
      addLog('Preprocessing 30s window...');

      const rawEcg = dummyData.ecg.slice(0, 21000);
      const rawEda = dummyData.eda.slice(0, 21000);

      const { ecg, eda } = preprocessWindow(rawEcg, rawEda);

      const inputData = new Float32Array(1 * 2 * 3840);
      inputData.set(ecg, 0);
      inputData.set(eda, 3840);

      const inputTensor = new ort.Tensor('float32', inputData, [1, 2, 3840]);

      addLog('Running ONNX inference...');
      const startTime = Date.now();
      const outputs = await session.run({ input: inputTensor });
      const duration = Date.now() - startTime;

      const outputTensor = outputs.output; 
      const logits = outputTensor.data;
      
      addLog(`Success: ${duration}ms`);
      addLog(`Logits: [${logits[0].toFixed(3)}, ${logits[1].toFixed(3)}]`);

      setResult(logits[1] > logits[0] ? 'STRESS' : 'NON-STRESS');
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
        <Text style={styles.statusLabel}>Result:</Text>
        <Text style={[styles.resultText, result === 'STRESS' ? styles.stress : styles.normal]}>
          {result || 'READY'}
        </Text>
      </View>

      <TouchableOpacity 
        style={[styles.button, (loading || !session) && styles.buttonDisabled]} 
        onPress={runInference}
        disabled={loading || !session}
      >
        {loading ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.buttonText}>RUN DETECTION</Text>
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
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    alignItems: 'center',
    paddingTop: 60,
    paddingHorizontal: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 20,
  },
  statusCard: {
    width: '100%',
    backgroundColor: '#fff',
    borderRadius: 15,
    padding: 20,
    alignItems: 'center',
    elevation: 4,
    marginBottom: 20,
  },
  statusLabel: {
    fontSize: 14,
    color: '#666',
  },
  resultText: {
    fontSize: 36,
    fontWeight: 'bold',
  },
  normal: { color: '#4CAF50' },
  stress: { color: '#F44336' },
  button: {
    backgroundColor: '#2196F3',
    paddingVertical: 12,
    paddingHorizontal: 30,
    borderRadius: 25,
  },
  buttonDisabled: { backgroundColor: '#90CAF9' },
  buttonText: { color: '#fff', fontSize: 16, fontWeight: 'bold' },
  logContainer: { flex: 1, width: '100%', marginTop: 20 },
  logTitle: { fontSize: 12, fontWeight: 'bold', color: '#999', marginBottom: 5 },
  logScroll: { flex: 1, backgroundColor: '#eee', borderRadius: 8, padding: 10 },
  logText: { fontSize: 11, color: '#333', marginBottom: 2 },
});
