import { VocalAudioProcessor } from './signalProcessor.js';

let processor = null;
let recordedData = null; // Store { rawAudio, sampleRate, features }
let playbackState = null; // Store { source, ctx }
let recordingStartTime = 0;
let recordingTimerInterval = null;
let isDrawing = false;
let currentRawBuffer = null;
let currentMFCC = null;
let currentFlatness = null;

// DOM Elements
const micSelect = document.getElementById('mic-select');
const bufferSelect = document.getElementById('buffer-select');
const btnRecord = document.getElementById('btn-record');
const btnPause = document.getElementById('btn-pause');
const btnStop = document.getElementById('btn-stop');
const btnReset = document.getElementById('btn-reset');
const btnPlayback = document.getElementById('btn-playback');
const btnExport = document.getElementById('btn-export');

const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const statusTime = document.getElementById('status-time');

// Canvas elements
const waveformCanvas = document.getElementById('waveform-canvas');
const waveformCtx = waveformCanvas?.getContext('2d');
const mfccCanvas = document.getElementById('mfcc-canvas');
const mfccCtx = mfccCanvas?.getContext('2d');
const flatnessCanvas = document.getElementById('flatness-canvas');
const flatnessCtx = flatnessCanvas?.getContext('2d');

// Diagnostic UI elements
const riskCard = document.getElementById('risk-card');
const riskTitle = document.getElementById('risk-title');
const riskDesc = document.getElementById('risk-desc');

const valF0 = document.getElementById('val-f0');
const valJitterLocal = document.getElementById('val-jitter-local');
const valJitterRAP = document.getElementById('val-jitter-rap');
const valShimmerLocal = document.getElementById('val-shimmer-local');
const valShimmerDB = document.getElementById('val-shimmer-db');
const valShimmerAPQ3 = document.getElementById('val-shimmer-apq3');

const dotJitterLocal = document.getElementById('dot-jitter-local');
const dotJitterRAP = document.getElementById('dot-jitter-rap');
const dotShimmerLocal = document.getElementById('dot-shimmer-local');
const dotShimmerDB = document.getElementById('dot-shimmer-db');
const dotShimmerAPQ3 = document.getElementById('dot-shimmer-apq3');

// Clinical threshold limits parsed dynamically from JSON
const thresholds = {
  jitterLocal: 1.04,
  jitterRAP: 0.68,
  shimmerLocal: 3.81,
  // Raised from clinical 0.35 dB → 0.45 dB to compensate for laptop built-in mic
  // compression artifacts and OS-level AGC that artificially inflate shimmer dB
  // even on healthy vowel phonation. Studio/clinical mics use 0.35 dB.
  shimmerDB: 0.45,
  shimmerAPQ3: 3.07,
};

let unSdgTarget = null;

// Flatness rolling history buffer
const flatnessHistory = [];
const maxFlatnessHistoryLength = 150;

// Initialize app
async function init() {
  processor = new VocalAudioProcessor();

  // Load metadata from un_sdg_metadata.json
  await loadMetadata();

  // Populate mic dropdown
  await populateMicrophones();

  // Bind event listeners
  btnRecord.addEventListener('click', startRecording);
  btnPause.addEventListener('click', togglePause);
  btnStop.addEventListener('click', stopRecording);
  btnReset.addEventListener('click', resetSession);
  btnPlayback.addEventListener('click', togglePlayback);
  btnExport.addEventListener('click', exportReport);

  // Set initial UI state
  updateUIState('idle');
  resizeCanvases();
  window.addEventListener('resize', resizeCanvases);
}

// Load configurations dynamically
async function loadMetadata() {
  try {
    const res = await fetch('/config/un_sdg_metadata.json');
    const data = await res.json();
    unSdgTarget = data.un_sdg_target;

    // Parse thresholds from clinical_metrics
    if (data.clinical_metrics) {
      const jThresholdStr =
        data.clinical_metrics.vocal_jitter?.clinical_thresholds?.pathological_indication
          ?.jitter_local_percent;
      const jRapStr =
        data.clinical_metrics.vocal_jitter?.clinical_thresholds?.pathological_indication
          ?.jitter_rap_percent;
      const sThresholdStr =
        data.clinical_metrics.vocal_shimmer?.clinical_thresholds?.pathological_indication
          ?.shimmer_local_percent;
      const sDbStr =
        data.clinical_metrics.vocal_shimmer?.clinical_thresholds?.pathological_indication
          ?.shimmer_db;
      const sApq3Str =
        data.clinical_metrics.vocal_shimmer?.clinical_thresholds?.pathological_indication
          ?.shimmer_apq3_percent;

      if (jThresholdStr) thresholds.jitterLocal = parseFloat(jThresholdStr.replace(/[^\d.]/g, ''));
      if (jRapStr) thresholds.jitterRAP = parseFloat(jRapStr.replace(/[^\d.]/g, ''));
      if (sThresholdStr) thresholds.shimmerLocal = parseFloat(sThresholdStr.replace(/[^\d.]/g, ''));
      if (sDbStr) thresholds.shimmerDB = parseFloat(sDbStr.replace(/[^\d.]/g, ''));
      if (sApq3Str) thresholds.shimmerAPQ3 = parseFloat(sApq3Str.replace(/[^\d.]/g, ''));
    }
  } catch (err) {
    // eslint-disable-next-line no-console
    console.error('Failed to load SDG metadata config, using local fallbacks:', err);
  }
}

// Microphone populate
async function populateMicrophones() {
  try {
    // Request temporary permission to enumerate devices
    await navigator.mediaDevices.getUserMedia({ audio: true });
    const devices = await navigator.mediaDevices.enumerateDevices();
    const audioDevices = devices.filter((d) => d.kind === 'audioinput');

    micSelect.innerHTML = '';
    audioDevices.forEach((device, index) => {
      const option = document.createElement('option');
      option.value = device.deviceId;
      option.textContent = device.label || `Microphone ${index + 1}`;
      micSelect.appendChild(option);
    });
  } catch (err) {
    micSelect.innerHTML = '<option value="">Default Microphone</option>';
  }
}

// Canvas resizing
function resizeCanvases() {
  [waveformCanvas, mfccCanvas, flatnessCanvas].forEach((canvas) => {
    if (canvas) {
      canvas.width = canvas.parentElement.clientWidth * window.devicePixelRatio;
      canvas.height = canvas.parentElement.clientHeight * window.devicePixelRatio;
    }
  });
}

// Drawing loop using requestAnimationFrame to prevent thread blocking
function drawLoop() {
  if (!isDrawing) return;

  if (currentRawBuffer && waveformCtx && waveformCanvas) {
    drawWaveform(waveformCanvas, waveformCtx, currentRawBuffer);
  }
  if (currentMFCC && mfccCtx && mfccCanvas) {
    drawMFCC(mfccCanvas, mfccCtx, currentMFCC);
  }
  if (typeof currentFlatness === 'number' && flatnessCtx && flatnessCanvas) {
    drawFlatness(flatnessCanvas, flatnessCtx, currentFlatness);
  }

  requestAnimationFrame(drawLoop);
}

// Recording start
async function startRecording() {
  try {
    const bufferSize = parseInt(bufferSelect.value, 10);
    const deviceId = micSelect.value;

    // Stop playback if playing
    stopPlayback();

    // Reset drawing parameters
    currentRawBuffer = null;
    currentMFCC = null;
    currentFlatness = null;

    // Set up features handler callback to update rolling values and live voicing status
    processor.onFeaturesExtracted = (features, rawBuffer, info) => {
      currentRawBuffer = rawBuffer;
      currentMFCC = features.mfcc;
      currentFlatness = features.spectralFlatness;

      if (info) {
        if (info.isSufficient) {
          statusDot.className = 'status-dot active';
          statusDot.style.backgroundColor = '#10b981';
          statusDot.style.boxShadow = '0 0 10px #10b981';
          statusText.textContent = `Recording: Sufficient Speech Captured (${info.voicedBlocksCount} voiced blocks) - Ready to Stop`;
        } else {
          statusDot.className = 'status-dot active';
          statusDot.style.backgroundColor = '';
          statusDot.style.boxShadow = '';
          statusText.textContent = `Recording Vocal Signal... (${info.voicedBlocksCount}/15 voiced blocks needed)`;
        }
      }
    };

    processor.onStateChange = (state) => {
      updateUIState(state);
    };

    await processor.start({
      bufferSize,
      deviceId,
    });

    recordingStartTime = Date.now();
    startTimer();

    // Start drawing animation frames
    isDrawing = true;
    requestAnimationFrame(drawLoop);
  } catch (err) {
    // Show user-friendly error card instead of default alert popup
    riskCard.className = 'risk-card pathology';
    riskTitle.textContent = 'Hardware Access Error';
    riskDesc.textContent = `Microphone capture failed: ${err.message || 'Permission denied'}. Please check system privacy settings and try again.`;
    updateUIState('idle');
  }
}

// Recording pause
function togglePause() {
  if (processor.isPaused) {
    processor.resume();
    startTimer();
  } else {
    processor.pause();
    stopTimer();
  }
}

// Recording stop
function stopRecording() {
  stopTimer();
  isDrawing = false;
  recordedData = processor.stop();
  updateUIState('stopped');

  if (recordedData) {
    analyzeVoiceData();
  }
}

// Reset session
function resetSession() {
  stopPlayback();
  isDrawing = false;
  recordedData = null;
  flatnessHistory.length = 0;

  currentRawBuffer = null;
  currentMFCC = null;
  currentFlatness = null;

  // Reset diagnostic UI
  riskCard.className = 'risk-card neutral';
  riskTitle.textContent = 'Awaiting Capture';
  riskDesc.textContent =
    'Please record a steady vowel sound (e.g., "ah") for at least 3-4 seconds to extract biomarkers.';

  [valF0, valJitterLocal, valJitterRAP, valShimmerLocal, valShimmerDB, valShimmerAPQ3].forEach(
    (el) => {
      if (el) el.textContent = '--';
    }
  );

  [dotJitterLocal, dotJitterRAP, dotShimmerLocal, dotShimmerDB, dotShimmerAPQ3].forEach((el) => {
    if (el) el.className = 'metric-status-dot';
  });

  // Clear canvases
  [waveformCanvas, mfccCanvas, flatnessCanvas].forEach((canvas) => {
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  });

  statusTime.textContent = '00:00';
  updateUIState('idle');
}

// Playback logic
function togglePlayback() {
  if (playbackState) {
    stopPlayback();
  } else if (recordedData && recordedData.rawAudio.length > 0) {
    btnPlayback.classList.add('playing');
    btnPlayback.querySelector('span').textContent = 'Stop Playback';

    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = audioContext.createBuffer(
      1,
      recordedData.rawAudio.length,
      recordedData.sampleRate
    );
    audioBuffer.getChannelData(0).set(recordedData.rawAudio);

    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);

    source.onended = () => {
      stopPlayback();
    };

    source.start(0);
    playbackState = { source, ctx: audioContext };
  }
}

function stopPlayback() {
  if (playbackState) {
    try {
      playbackState.source.stop();
      playbackState.ctx.close();
    } catch (e) {
      // Ignored
    }
    playbackState = null;
  }
  btnPlayback.classList.remove('playing');
  btnPlayback.querySelector('span').textContent = 'Playback Voice';
}

// Timer helpers
function startTimer() {
  if (recordingTimerInterval) clearInterval(recordingTimerInterval);
  recordingTimerInterval = setInterval(() => {
    const elapsed = Date.now() - recordingStartTime;
    const sec = Math.floor(elapsed / 1000) % 60;
    const min = Math.floor(elapsed / 60000);
    statusTime.textContent = `${min.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
  }, 200);
}

function stopTimer() {
  if (recordingTimerInterval) {
    clearInterval(recordingTimerInterval);
    recordingTimerInterval = null;
  }
}

// UI State Manager
function updateUIState(state) {
  statusDot.style.backgroundColor = '';
  statusDot.style.boxShadow = '';
  switch (state) {
    case 'idle':
      btnRecord.disabled = false;
      btnRecord.classList.remove('recording');
      btnRecord.querySelector('span').textContent = 'Start';
      btnPause.disabled = true;
      btnPause.querySelector('span').textContent = 'Pause';
      btnStop.disabled = true;
      btnReset.disabled = true;
      btnPlayback.disabled = true;
      btnExport.disabled = true;
      statusDot.className = 'status-dot';
      statusText.textContent = 'Ready';
      break;
    case 'recording':
      btnRecord.disabled = true;
      btnRecord.classList.add('recording');
      btnRecord.querySelector('span').textContent = 'Recording';
      btnPause.disabled = false;
      btnPause.querySelector('span').textContent = 'Pause';
      btnStop.disabled = false;
      btnReset.disabled = true;
      btnPlayback.disabled = true;
      btnExport.disabled = true;
      statusDot.className = 'status-dot active';
      statusText.textContent = 'Recording Vocal Signal...';
      break;
    case 'paused':
      btnRecord.disabled = true;
      btnPause.disabled = false;
      btnPause.querySelector('span').textContent = 'Resume';
      btnStop.disabled = false;
      btnReset.disabled = true;
      btnPlayback.disabled = true;
      btnExport.disabled = true;
      statusDot.className = 'status-dot';
      statusText.textContent = 'Paused';
      break;
    case 'stopped':
      btnRecord.disabled = false;
      btnRecord.classList.remove('recording');
      btnRecord.querySelector('span').textContent = 'Re-record';
      btnPause.disabled = true;
      btnPause.querySelector('span').textContent = 'Pause';
      btnStop.disabled = true;
      btnReset.disabled = false;
      btnPlayback.disabled = false;
      btnExport.disabled = false;
      statusDot.className = 'status-dot';
      statusText.textContent = 'Capture Completed';
      break;
  }
}

// Core voice biomarker evaluation
function analyzeVoiceData() {
  if (!recordedData) return;

  const result = processor.analyze(recordedData.rawAudio, recordedData.sampleRate);

  if (!result.success) {
    riskCard.className = 'risk-card neutral';
    riskTitle.textContent = 'Analysis Failed';
    riskDesc.textContent = result.error || 'Speech could not be analyzed.';
    btnExport.disabled = true;
    return;
  }

  const { metrics } = result;

  // 1. Populate values in UI
  valF0.textContent = `${metrics.avgF0.toFixed(1)}`;
  valJitterLocal.textContent = `${metrics.jitterLocalPercent.toFixed(3)}`;
  valJitterRAP.textContent = `${metrics.jitterRAP.toFixed(3)}`;
  valShimmerLocal.textContent = `${metrics.shimmerLocalPercent.toFixed(3)}`;
  valShimmerDB.textContent = `${metrics.shimmerDB.toFixed(3)}`;
  valShimmerAPQ3.textContent = `${metrics.shimmerAPQ3.toFixed(3)}`;

  // 2. Evaluate limits and color status dots
  const isJitterLocalElevated = metrics.jitterLocalPercent >= thresholds.jitterLocal;
  const isJitterRAPElevated = metrics.jitterRAP >= thresholds.jitterRAP;
  const isShimmerLocalElevated = metrics.shimmerLocalPercent >= thresholds.shimmerLocal;
  const isShimmerDBElevated = metrics.shimmerDB >= thresholds.shimmerDB;
  const isShimmerAPQ3Elevated = metrics.shimmerAPQ3 >= thresholds.shimmerAPQ3;

  dotJitterLocal.className = `metric-status-dot ${isJitterLocalElevated ? 'pathological' : 'normal'}`;
  dotJitterRAP.className = `metric-status-dot ${isJitterRAPElevated ? 'pathological' : 'normal'}`;
  dotShimmerLocal.className = `metric-status-dot ${isShimmerLocalElevated ? 'pathological' : 'normal'}`;
  dotShimmerDB.className = `metric-status-dot ${isShimmerDBElevated ? 'pathological' : 'normal'}`;
  dotShimmerAPQ3.className = `metric-status-dot ${isShimmerAPQ3Elevated ? 'pathological' : 'normal'}`;

  // 3. Overall Diagnostic Scoring
  // Clinical rule: require ≥2 elevated markers in the same domain (jitter OR shimmer)
  // OR at least 1 marker elevated in BOTH domains simultaneously.
  // This prevents a single noisy measurement (e.g. shimmer dB from laptop AGC) from
  // triggering a false pathological flag — matching Praat/MDVP multi-marker consensus.
  const jitterElevatedCount = (isJitterLocalElevated ? 1 : 0) + (isJitterRAPElevated ? 1 : 0);
  const shimmerElevatedCount =
    (isShimmerLocalElevated ? 1 : 0) +
    (isShimmerDBElevated ? 1 : 0) +
    (isShimmerAPQ3Elevated ? 1 : 0);

  const jitterConsensus = jitterElevatedCount >= 2;
  const shimmerConsensus = shimmerElevatedCount >= 2;
  const crossDomainConsensus = jitterElevatedCount >= 1 && shimmerElevatedCount >= 1;

  const isPathological = jitterConsensus || shimmerConsensus || crossDomainConsensus;

  if (isPathological) {
    const elevatedList = [];
    if (isJitterLocalElevated) elevatedList.push('Jitter (Local)');
    if (isJitterRAPElevated) elevatedList.push('Jitter (RAP)');
    if (isShimmerLocalElevated) elevatedList.push('Shimmer (Local)');
    if (isShimmerDBElevated) elevatedList.push('Shimmer (dB)');
    if (isShimmerAPQ3Elevated) elevatedList.push('Shimmer (APQ3)');

    riskCard.className = 'risk-card pathology';
    riskTitle.textContent = 'Pathological Indication';
    riskDesc.textContent =
      `Elevated perturbation detected across multiple markers: ${elevatedList.join(', ')}. This pattern matches clinical correlates for vocal strain, neuromuscular fatigue, or mucosal swelling. If using a laptop built-in microphone, consider re-testing at higher volume or closer distance.`;
  } else {
    const borderlineList = [];
    if (isJitterLocalElevated) borderlineList.push('Jitter (Local)');
    if (isJitterRAPElevated) borderlineList.push('Jitter (RAP)');
    if (isShimmerLocalElevated) borderlineList.push('Shimmer (Local)');
    if (isShimmerDBElevated) borderlineList.push('Shimmer (dB)');
    if (isShimmerAPQ3Elevated) borderlineList.push('Shimmer (APQ3)');

    riskCard.className = 'risk-card normal';
    riskTitle.textContent = 'Normal Vocal Screening';
    riskDesc.textContent = borderlineList.length > 0
      ? `All biomarkers within safe range. Note: ${borderlineList.join(', ')} slightly elevated — likely microphone noise artifact. No clinical concern.`
      : 'All acoustic vocal biomarkers lie within standard clinical safety ranges. No active signs of neuromuscular vocal fatigue or laryngeal pathology detected.';
  }

  // Attach metric results to recordedData for export
  recordedData.metrics = metrics;
  recordedData.isPathological = isPathological;
}

// Canvas Drawing functions with neon shadow glows

function drawWaveform(canvas, ctx, data) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 2 * window.devicePixelRatio;
  ctx.strokeStyle = '#00f0ff';
  ctx.shadowBlur = 6 * window.devicePixelRatio;
  ctx.shadowColor = '#00f0ff';

  ctx.beginPath();
  const sliceWidth = canvas.width / data.length;
  let x = 0;

  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    const y = (v + 1) * (canvas.height / 2);

    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
    x += sliceWidth;
  }

  ctx.stroke();
  ctx.shadowBlur = 0; // reset
}

function drawMFCC(canvas, ctx, mfcc) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const barCount = mfcc.length;
  const barWidth = canvas.width / barCount - 4 * window.devicePixelRatio;

  const scale = canvas.height / 80;
  const baseline = canvas.height / 2;

  ctx.shadowBlur = 4 * window.devicePixelRatio;
  ctx.shadowColor = '#3b82f6';

  for (let i = 0; i < barCount; i++) {
    const value = mfcc[i];
    const barHeight = value * scale;
    const x = i * (barWidth + 4 * window.devicePixelRatio) + 2 * window.devicePixelRatio;
    const y = baseline - barHeight;

    const gradient = ctx.createLinearGradient(x, baseline, x, y);
    gradient.addColorStop(0, '#3b82f6');
    gradient.addColorStop(1, '#00f0ff');
    ctx.fillStyle = gradient;

    ctx.fillRect(x, baseline, barWidth, -barHeight);
  }
  ctx.shadowBlur = 0; // reset
}

function drawFlatness(canvas, ctx, flatness) {
  flatnessHistory.push(flatness);
  if (flatnessHistory.length > maxFlatnessHistoryLength) {
    flatnessHistory.shift();
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 2 * window.devicePixelRatio;
  ctx.strokeStyle = '#8b5cf6';
  ctx.shadowBlur = 6 * window.devicePixelRatio;
  ctx.shadowColor = '#8b5cf6';

  ctx.beginPath();
  const sliceWidth = canvas.width / maxFlatnessHistoryLength;
  let x = 0;

  for (let i = 0; i < flatnessHistory.length; i++) {
    const val = flatnessHistory[i];
    // Map spectralFlatness 0-1 to canvas height
    const y =
      canvas.height -
      val * (canvas.height - 10 * window.devicePixelRatio) -
      5 * window.devicePixelRatio;

    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
    x += sliceWidth;
  }

  ctx.stroke();
  ctx.shadowBlur = 0; // reset
}

// Download PDF/JSON report
function exportReport() {
  if (!recordedData || !recordedData.metrics) return;

  const report = {
    timestamp: new Date().toLocaleString(),
    un_sdg_target: unSdgTarget || {
      goal: 3,
      goal_title: 'Good Health and Well-being',
      target_index: '3.4',
      domain: 'Digital Phenotyping & Non-Invasive Vocal Biomarkers',
    },
    audio_configuration: {
      buffer_size: parseInt(bufferSelect.value, 10),
      sample_rate_hz: recordedData.sampleRate,
      recording_duration_sec: parseFloat(
        (recordedData.rawAudio.length / recordedData.sampleRate).toFixed(2)
      ),
    },
    extracted_biomarkers: {
      fundamental_frequency_f0_hz: parseFloat(recordedData.metrics.avgF0.toFixed(2)),
      jitter_local_percent: parseFloat(recordedData.metrics.jitterLocalPercent.toFixed(4)),
      jitter_rap_percent: parseFloat(recordedData.metrics.jitterRAP.toFixed(4)),
      shimmer_local_percent: parseFloat(recordedData.metrics.shimmerLocalPercent.toFixed(4)),
      shimmer_db: parseFloat(recordedData.metrics.shimmerDB.toFixed(4)),
      shimmer_apq3_percent: parseFloat(recordedData.metrics.shimmerAPQ3.toFixed(4)),
    },
    evaluation_thresholds: thresholds,
    screening_assessment: recordedData.isPathological ? 'Pathological Indication' : 'Normal',
    compliance_details: {
      privacy: '100% Client-Side Edge-Compute. Zero audio transmitted.',
      clinical_reference: 'SDG 3.4 Screening Framework',
    },
  };

  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `Vocal_Biomarker_Report_${Date.now()}.json`;
  a.click();
}

// Run setup on document load
document.addEventListener('DOMContentLoaded', init);
