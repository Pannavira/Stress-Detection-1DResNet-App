import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ScrollView, ActivityIndicator, Dimensions, SafeAreaView } from 'react-native';
import * as ort from 'onnxruntime-react-native';
import { Asset } from 'expo-asset';
import * as FileSystem from 'expo-file-system/legacy';
import * as DocumentPicker from 'expo-document-picker';
import { LineChart } from 'react-native-chart-kit';
import { preprocessWindow } from './utils/preprocessing';

const screenWidth = Dimensions.get('window').width;

export default function App() {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [latency, setLatency] = useState(0);
  const [selectedData, setSelectedData] = useState(null);
  const [fileName, setFileName] = useState('No data loaded');
  const [modalityMode, setModalityMode] = useState(null); // 'DUAL', 'ECG_ONLY', 'EDA_ONLY'
  const [logs, setLogs] = useState([]);

  const LOGIT_GAIN = 1.0;

  const addLog = (msg) => {
    const time = new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    setLogs((prev) => [`[${time}] ${msg}`, ...prev].slice(0, 30));
  };

  useEffect(() => { loadModel(); }, []);

  const loadModel = async () => {
    try {
      setLoading(true);
      addLog('Initializing Model Engine...');
      const modelAsset = Asset.fromModule(require('./assets/model.onnx'));
      await modelAsset.downloadAsync();
      const session = await ort.InferenceSession.create(modelAsset.localUri);
      setSession(session);
      addLog('✅ Success: ResNet-1D ONNX weights loaded');
    } catch (e) { 
      addLog('❌ Error: Load failed - ' + e.message); 
    } finally { setLoading(false); }
  };

  const softmax = (logits, gain = 1.0) => {
    const scaled = logits.map(l => l * gain);
    const maxLogit = Math.max(...scaled);
    const scores = scaled.map(l => Math.exp(l - maxLogit));
    const sumScores = scores.reduce((a, b) => a + b, 0);
    return scores.map(s => s / sumScores);
  };

  const isSignalActive = (sig) => {
    if (!sig || !Array.isArray(sig) || sig.length === 0) return false;
    const first = sig[0];
    for (let i = 1; i < Math.min(sig.length, 500); i++) {
      if (Math.abs(sig[i] - first) > 1e-5) return true;
    }
    return false;
  };

  const analyzeModality = (data) => {
    if (data.modality === 'DUAL') return 'DUAL';
    if (data.modality === 'ECG-only') return 'ECG_ONLY';
    if (data.modality === 'EDA-only') return 'EDA_ONLY';

    const hasEcg = isSignalActive(data.ecg);
    const hasEda = isSignalActive(data.eda);

    if (hasEcg && hasEda) return 'DUAL';
    if (hasEcg && !hasEda) return 'ECG_ONLY';
    if (!hasEcg && hasEda) return 'EDA_ONLY';
    return 'UNKNOWN';
  };

  const pickDocument = async () => {
    try {
      addLog('Opening file browser...');
      const res = await DocumentPicker.getDocumentAsync({ type: '*/*', copyToCacheDirectory: true });
      if (!res || res.canceled) {
        addLog('File selection cancelled');
        return;
      }
      
      const file = res.assets[0];
      setFileName(file.name);
      setLoading(true);
      
      addLog(`Reading ${file.name}...`);
      const jsonContent = await FileSystem.readAsStringAsync(file.uri);
      const data = JSON.parse(jsonContent);
      const finalData = data.ecg || data.eda ? data : (data.default || data);
      
      const mode = analyzeModality(finalData);
      setModalityMode(mode);
      setSelectedData(finalData);
      setResult(null);
      setConfidence(0);

      if (mode === 'DUAL') {
        addLog('✅ Data Parsed: Dual Modality (ECG + EDA) [Gain 8.0]');
      } else if (mode === 'ECG_ONLY') {
        addLog('⚡ Data Parsed: Missing Modality (ECG ONLY - EDA Zeroed) [Gain 1.0]');
      } else if (mode === 'EDA_ONLY') {
        addLog('💧 Data Parsed: Missing Modality (EDA ONLY - ECG Zeroed) [Gain 1.0]');
      } else {
        addLog('⚠️ Data Parsed: Partial / Custom Modality [Gain 1.0]');
      }
    } catch (e) { 
        addLog('❌ Error: ' + e.message); 
    } finally { setLoading(false); }
  };

  const runInference = async () => {
    if (!session || !selectedData) return;
    try {
      setLoading(true);
      addLog(`Preprocessing window for [${modalityMode}] mode...`);
      
      const rawEcg = selectedData.ecg || [];
      const rawEda = selectedData.eda || [];

      const { ecg, eda } = preprocessWindow(rawEcg, rawEda);
      
      const inputData = new Float32Array(2 * 7680);
      inputData.set(ecg, 0);
      inputData.set(eda, 7680);

      const inputTensor = new ort.Tensor('float32', inputData, [1, 2, 7680]);
      
      addLog('Executing 1D-ResNet ONNX inference...');
      const startTime = Date.now();
      const outputs = await session.run({ input: inputTensor });
      const endTime = Date.now();
      const duration = endTime - startTime;
      
      setLatency(duration);
      
      const rawLogits = Array.from(outputs.output.data);
      // Use 8.0 gain for full DUAL modality (original state), 1.0 for missing/partial modalities
      const gain = modalityMode === 'DUAL' ? 8.0 : 1.0;
      const probs = softmax(rawLogits, gain);
      const isStress = rawLogits[1] > rawLogits[0];
      
      setResult(isStress ? 'STRESS' : 'NORMAL');
      setConfidence(probs[isStress ? 1 : 0] * 100);
      
      addLog(`✨ Inference Completed (${duration}ms, Gain: ${gain}x)`);
      addLog(`Logits: [L0: ${rawLogits[0].toFixed(3)}, L1: ${rawLogits[1].toFixed(3)}]`);
    } catch (e) { 
        addLog('❌ Inference failed: ' + e.message); 
    } finally { setLoading(false); }
  };

  const renderChart = (data, color, label, isMissing = false) => {
    if (isMissing || !data || !isSignalActive(data)) {
      return (
        <View style={styles.missingChannelBox}>
          <Text style={styles.missingChannelTitle}>{label}</Text>
          <Text style={styles.missingChannelBadge}>⚡ MISSING MODALITY (ZERO-MASKED)</Text>
          <Text style={styles.missingChannelSubtext}>Channel set to 0.0 for ResNet KD robustness</Text>
        </View>
      );
    }

    const stride = Math.max(1, Math.floor(data.length / 50));
    const chartData = data.filter((_, i) => i % stride === 0).slice(0, 50);

    return (
      <View style={styles.chartContainer}>
        <Text style={styles.chartLabel}>{label}</Text>
        <LineChart
          data={{ datasets: [{ data: chartData }] }}
          width={screenWidth - 60}
          height={80}
          chartConfig={{
            backgroundColor: '#fff',
            backgroundGradientFrom: '#fff',
            backgroundGradientTo: '#fff',
            color: (opacity = 1) => color,
            strokeWidth: 2,
            decimalPlaces: 1,
          }}
          bezier
          withDots={false}
          withInnerLines={false}
          withOuterLines={false}
          withVerticalLabels={false}
          withHorizontalLabels={false}
          style={styles.chart}
        />
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <Text style={styles.headerTitle}>Stress Detection (2022100031)</Text>
        <Text style={styles.headerSubtitle}>Edge KD Architecture Prototype</Text>

        {/* 1. Visualization Section */}
        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Text style={styles.cardTitle}>Signal Input Context</Text>
            {modalityMode && (
              <View style={[
                styles.modalityBadge, 
                modalityMode === 'DUAL' ? styles.badgeDual : styles.badgeMissing
              ]}>
                <Text style={styles.modalityBadgeText}>
                  {modalityMode === 'DUAL' ? '🟢 DUAL (ECG+EDA)' : 
                   modalityMode === 'ECG_ONLY' ? '⚡ ECG ONLY' : '💧 EDA ONLY'}
                </Text>
              </View>
            )}
          </View>

          {selectedData ? (
            <>
              {renderChart(selectedData.ecg, '#e74c3c', 'ECG (Beat Intervals)', modalityMode === 'EDA_ONLY')}
              {renderChart(selectedData.eda, '#3498db', 'EDA (Skin Response)', modalityMode === 'ECG_ONLY')}
              <Text style={styles.dataMeta}>
                WESAD Subject: {selectedData.subject || 'S2'} | Mode: {selectedData.modality || modalityMode} | Label: {selectedData.label || 'Research'}
              </Text>
            </>
          ) : (
            <View style={styles.emptyState}>
              <Text style={styles.emptyText}>Select research JSON data (Dual or ECG/EDA-Only) to begin</Text>
            </View>
          )}
          
          <TouchableOpacity style={styles.pickButton} onPress={pickDocument}>
            <Text style={styles.pickButtonText}>{fileName}</Text>
          </TouchableOpacity>
        </View>

        {/* 2. Analysis Section */}
        <TouchableOpacity 
          style={[styles.runButton, (!selectedData || loading) && styles.runButtonDisabled]} 
          onPress={runInference}
          disabled={!selectedData || loading}
        >
          {loading ? <ActivityIndicator color="#fff" /> : <Text style={styles.runButtonText}>ANALYZE RESNET</Text>}
        </TouchableOpacity>

        {/* 3. Results Section */}
        {result && (
          <View style={[styles.resultCard, result === 'STRESS' ? styles.resultCardStress : styles.resultCardNormal]}>
            <View style={styles.resultHeader}>
              <Text style={styles.resultLabel}>CLASSIFICATION</Text>
              <Text style={styles.latencyTag}>{latency}ms latency</Text>
            </View>
            <Text style={styles.resultValue}>{result}</Text>
            <Text style={styles.confText}>{confidence.toFixed(1)}% Confidence Score</Text>
            <View style={styles.explanationBox}>
              <Text style={styles.explanationText}>
                {result === 'STRESS' 
                  ? `ResNet student model detected elevated stress pattern [${modalityMode} mode]. Features indicate high sympathetic arousal.`
                  : `Physiological signals remain in baseline range [${modalityMode} mode]. Parasympathetic dominance detected.`}
              </Text>
            </View>
          </View>
        )}

        {/* 4. Persistent Logs */}
        <View style={styles.logSection}>
          <Text style={styles.logHeader}>Execution Logs</Text>
          <View style={styles.logContainer}>
            <ScrollView nestedScrollEnabled={true} style={styles.logScroll}>
              {logs.map((log, i) => <Text key={i} style={styles.logText}>{log}</Text>)}
              {logs.length === 0 && <Text style={styles.logText}>System idle. Ready for data input.</Text>}
            </ScrollView>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f8f9fa' },
  scrollContent: { padding: 20 },
  headerTitle: { fontSize: 24, fontWeight: '900', color: '#1a1a1a', textAlign: 'center' },
  headerSubtitle: { fontSize: 11, color: '#95a5a6', textAlign: 'center', marginBottom: 20, textTransform: 'uppercase', letterSpacing: 2 },
  card: { backgroundColor: '#fff', borderRadius: 24, padding: 20, elevation: 6, shadowColor: '#000', shadowOpacity: 0.1, shadowRadius: 15, marginBottom: 20 },
  cardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 15 },
  cardTitle: { fontSize: 15, fontWeight: '800', color: '#2c3e50' },
  modalityBadge: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 12 },
  badgeDual: { backgroundColor: '#e8f8f5' },
  badgeMissing: { backgroundColor: '#fef9e7' },
  modalityBadgeText: { fontSize: 10, fontWeight: '800', color: '#27ae60' },
  chartContainer: { marginBottom: 10 },
  chartLabel: { fontSize: 10, color: '#bdc3c7', marginBottom: 2, fontWeight: '700', textTransform: 'uppercase' },
  chart: { borderRadius: 12, marginLeft: -25 },
  missingChannelBox: { backgroundColor: '#fdfefe', borderColor: '#f39c12', borderWidth: 1, borderStyle: 'dashed', padding: 12, borderRadius: 14, marginBottom: 10 },
  missingChannelTitle: { fontSize: 10, color: '#7f8c8d', fontWeight: '700', textTransform: 'uppercase', marginBottom: 4 },
  missingChannelBadge: { fontSize: 10, fontWeight: '900', color: '#d35400', marginBottom: 2 },
  missingChannelSubtext: { fontSize: 9, color: '#95a5a6' },
  dataMeta: { fontSize: 10, color: '#bdc3c7', textAlign: 'center', marginTop: 8, fontWeight: '600' },
  emptyState: { height: 120, justifyContent: 'center', alignItems: 'center', borderStyle: 'dashed', borderWidth: 1, borderColor: '#dcdde1', borderRadius: 20, marginBottom: 10 },
  emptyText: { fontSize: 11, color: '#95a5a6', textAlign: 'center', paddingHorizontal: 20 },
  pickButton: { backgroundColor: '#f1f3f5', padding: 12, borderRadius: 14, alignItems: 'center', marginTop: 10 },
  pickButtonText: { color: '#3498db', fontSize: 12, fontWeight: 'bold' },
  runButton: { backgroundColor: '#2d3436', padding: 18, borderRadius: 18, alignItems: 'center', elevation: 8, marginBottom: 20 },
  runButtonDisabled: { backgroundColor: '#dfe6e9' },
  runButtonText: { color: '#fff', fontSize: 15, fontWeight: '900', letterSpacing: 1 },
  resultCard: { borderRadius: 24, padding: 25, elevation: 12 },
  resultCardNormal: { backgroundColor: '#00b894' },
  resultCardStress: { backgroundColor: '#d63031' },
  resultHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  resultLabel: { color: 'rgba(255,255,255,0.6)', fontSize: 11, fontWeight: '800', letterSpacing: 1.5 },
  latencyTag: { color: '#fff', fontSize: 10, backgroundColor: 'rgba(0,0,0,0.2)', paddingHorizontal: 8, paddingVertical: 3, borderRadius: 8, fontWeight: 'bold' },
  resultValue: { color: '#fff', fontSize: 48, fontWeight: '900', marginVertical: 2 },
  confText: { color: '#fff', fontSize: 15, fontWeight: '700', marginBottom: 10 },
  explanationBox: { backgroundColor: 'rgba(255,255,255,0.12)', padding: 15, borderRadius: 16 },
  explanationText: { color: '#fff', fontSize: 12, lineHeight: 18, fontWeight: '500' },
  logSection: { marginTop: 25 },
  logHeader: { fontSize: 12, fontWeight: '900', color: '#2d3436', marginBottom: 10, textTransform: 'uppercase', letterSpacing: 1 },
  logContainer: { backgroundColor: '#1e272e', borderRadius: 18, padding: 15, height: 180 },
  logScroll: { flex: 1 },
  logText: { fontSize: 10, color: '#00d8d6', marginBottom: 4, fontFamily: 'monospace' },
});
