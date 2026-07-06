/**
 * Edge-Compute Acoustic Vocal Biomarker Platform
 * Signal Processing and Audio Capture Pipeline
 */

import {
  detectPitchAutocorrelation,
  extractCyclesAndAmplitudes,
  calculateJitterLocalPercent,
  calculateJitterRAP,
  calculateShimmerLocalPercent,
  calculateShimmerDB,
  calculateShimmerAPQ3,
} from './algorithms/index.js';

/**
 * 1st-order IIR High-Pass Filter run twice (steeper 12dB/octave slope)
 * to remove DC offsets, breathing low-frequency rumble, and 50/60 Hz power hum.
 */
function applyHighPassFilter(samples, sampleRate, cutoff = 80) {
  const rc = 1.0 / (2.0 * Math.PI * cutoff);
  const dt = 1.0 / sampleRate;
  const alpha = rc / (rc + dt);

  // Pass 1
  let prevRaw = 0;
  let prevFiltered = 0;
  for (let i = 0; i < samples.length; i++) {
    const raw = samples[i];
    const filtered = alpha * (prevFiltered + raw - prevRaw);
    prevRaw = raw;
    prevFiltered = filtered;
    samples[i] = filtered;
  }

  // Pass 2
  prevRaw = 0;
  prevFiltered = 0;
  for (let i = 0; i < samples.length; i++) {
    const raw = samples[i];
    const filtered = alpha * (prevFiltered + raw - prevRaw);
    prevRaw = raw;
    prevFiltered = filtered;
    samples[i] = filtered;
  }
}

export class VocalAudioProcessor {
  constructor() {
    this.audioContext = null;
    this.stream = null;
    this.source = null;
    this.processor = null;
    this.isRecording = false;
    this.isPaused = false;

    // Accumulators
    this.accumulatedSamples = []; // Array of Float32Array chunks
    this.featuresHistory = []; // Array of { mfcc, spectralFlatness, rms, timestamp }
    this.sampleRate = 44100;
    this.bufferSize = 2048;

    // Callbacks
    this.onFeaturesExtracted = null; // (features, rawTimeDomainData) => {}
    this.onStateChange = null; // (state) => {}
  }

  /**
   * Starts capturing audio from microphone.
   * @param {object} options
   * @param {number} [options.bufferSize] - Audio buffer size (1024, 2048, 4096)
   * @param {number} [options.sampleRate] - Requested sampling rate
   */
  async start(options = {}) {
    if (this.isRecording) return;

    this.bufferSize = options.bufferSize || 2048;
    const requestedSampleRate = options.sampleRate || 44100;

    // 1. Request microphone permissions
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });

    // 2. Initialize AudioContext
    try {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: requestedSampleRate,
      });
    } catch (e) {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }

    this.sampleRate = this.audioContext.sampleRate;
    this.source = this.audioContext.createMediaStreamSource(this.stream);

    // 3. Reset state
    this.accumulatedSamples = [];
    this.featuresHistory = [];
    this.voicedBlocksCount = 0;
    this.isRecording = true;
    this.isPaused = false;

    // 4. Create ScriptProcessorNode
    this.processor = this.audioContext.createScriptProcessor(this.bufferSize, 1, 1);
    this.processor.onaudioprocess = (e) => {
      if (!this.isRecording || this.isPaused) return;

      const inputData = e.inputBuffer.getChannelData(0);

      // Clone buffer to avoid data changing before callback processing
      const bufferClone = new Float32Array(inputData);

      // Accumulate raw sample chunk
      this.accumulatedSamples.push(bufferClone);

      // Extract real-time features using Meyda (loaded from CDN in browser)
      if (typeof window.Meyda !== 'undefined') {
        try {
          const features = window.Meyda.extract(['mfcc', 'spectralFlatness', 'rms'], bufferClone);
          if (features) {
            // Live voicing check for real-time UI indicator
            let isVoiced = false;
            if (features.rms >= 0.04) {
              const pitchRes = detectPitchAutocorrelation(bufferClone, this.sampleRate, {
                minFreq: 50,
                maxFreq: 500,
                voicedThreshold: 0.50,
              });
              isVoiced = pitchRes.isVoiced && pitchRes.frequency > 0;
            }
            if (isVoiced) {
              this.voicedBlocksCount = (this.voicedBlocksCount || 0) + 1;
            }

            this.featuresHistory.push({
              mfcc: features.mfcc,
              spectralFlatness: features.spectralFlatness,
              rms: features.rms,
              timestamp: this.audioContext.currentTime,
            });

            if (this.onFeaturesExtracted) {
              this.onFeaturesExtracted(features, bufferClone, {
                totalBlocks: this.accumulatedSamples.length,
                voicedBlocksCount: this.voicedBlocksCount || 0,
                isSufficient: (this.voicedBlocksCount || 0) >= 15,
              });
            }
          }
        } catch (err) {
          // eslint-disable-next-line no-console
          console.error('Meyda feature extraction failed:', err);
        }
      }
    };

    // Connect components
    this.source.connect(this.processor);
    this.processor.connect(this.audioContext.destination);

    if (this.onStateChange) this.onStateChange('recording');
  }

  /**
   * Pauses audio recording.
   */
  pause() {
    if (!this.isRecording || this.isPaused) return;
    this.isPaused = true;
    if (this.onStateChange) this.onStateChange('paused');
  }

  /**
   * Resumes audio recording.
   */
  resume() {
    if (!this.isRecording || !this.isPaused) return;
    this.isPaused = false;
    if (this.onStateChange) this.onStateChange('recording');
  }

  /**
   * Stops recording and returns consolidated data.
   * @returns {object|null}
   */
  stop() {
    if (!this.isRecording) return null;

    this.isRecording = false;

    if (this.processor) {
      this.processor.disconnect();
      this.processor.onaudioprocess = null;
      this.processor = null;
    }
    if (this.source) {
      this.source.disconnect();
      this.source = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }

    if (this.onStateChange) this.onStateChange('stopped');

    // Consolidate raw audio buffer
    const totalLength = this.accumulatedSamples.reduce((sum, chunk) => sum + chunk.length, 0);
    const rawAudio = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of this.accumulatedSamples) {
      rawAudio.set(chunk, offset);
      offset += chunk.length;
    }

    const features = [...this.featuresHistory];

    this.accumulatedSamples = [];
    this.featuresHistory = [];

    return {
      rawAudio,
      sampleRate: this.sampleRate,
      features,
    };
  }

  /**
   * Feeds raw captured audio into core jitter & shimmer analysis algorithms.
   * @param {Float32Array} rawAudio - The accumulated raw audio samples.
   * @param {number} sampleRate - Sample rate of raw audio.
   * @returns {object} Analysis metrics or error message.
   */
  analyze(rawAudio, sampleRate) {
    if (!rawAudio || rawAudio.length === 0) {
      return {
        success: false,
        error: 'No audio data captured to analyze.',
      };
    }

    // Apply High-Pass Filter to remove breathing drift, DC offset, and low-frequency electrical hum
    applyHighPassFilter(rawAudio, sampleRate, 80);

    const blockSize = this.bufferSize;
    const hopSize = Math.floor(blockSize / 2);
    const sampleCount = rawAudio.length;

    const voicedBlocks = [];
    const voicedChunks = [];

    // Step 1: Analyze block-by-block using autocorrelation to find average F0 and detect voiced segments
    for (let i = 0; i + blockSize <= sampleCount; i += hopSize) {
      const block = rawAudio.subarray(i, i + blockSize);

      // Amplitude Gate: filter out background room noise and silence.
      let sumSq = 0;
      for (let j = 0; j < block.length; j++) {
        sumSq += block[j] * block[j];
      }
      const rms = Math.sqrt(sumSq / block.length);
      if (rms < 0.04) continue;

      const pitchResult = detectPitchAutocorrelation(block, sampleRate, {
        minFreq: 50,
        maxFreq: 500,
        voicedThreshold: 0.50,
      });

      if (pitchResult.isVoiced && pitchResult.frequency > 0) {
        voicedBlocks.push({
          frequency: pitchResult.frequency,
          periodSamples: pitchResult.periodSamples,
        });
        voicedChunks.push(block);
      }
    }

    if (voicedBlocks.length < 15) {
      return {
        success: false,
        error: 'No clear, steady vowel sound detected (too much background noise or silence). Please record a steady "ahh" clearly for at least 3-4 seconds.',
      };
    }

    // Step 2: Median-based F0 outlier rejection.
    // Keep only blocks within 20% of the median F0 to discard plosive transients and
    // pitch-doubled detection artifacts. Both blocks and their audio chunks are filtered together.
    const allFreqs = voicedBlocks.map((b) => b.frequency);
    const sortedF = [...allFreqs].sort((a, b) => a - b);
    const medianF0 = sortedF[Math.floor(sortedF.length / 2)];

    const stableIndices = voicedBlocks
      .map((b, idx) => ({ ok: Math.abs(b.frequency - medianF0) / medianF0 <= 0.20, idx }))
      .filter((x) => x.ok)
      .map((x) => x.idx);

    if (stableIndices.length < 10) {
      return {
        success: false,
        error: 'Voice pitch varied too much during recording. Please sustain a steady "ahh" vowel sound for at least 3-4 seconds.',
      };
    }

    const stableBlocks = stableIndices.map((i) => voicedBlocks[i]);
    const stableChunks = stableIndices.map((i) => voicedChunks[i]);

    const frequencies = stableBlocks.map((b) => b.frequency);
    const avgF0 = frequencies.reduce((a, b) => a + b, 0) / frequencies.length;
    const minF0 = Math.min(...frequencies);
    const maxF0 = Math.max(...frequencies);

    const periodsList = stableBlocks.map((b) => b.periodSamples);
    const avgPeriodSamples = Math.round(
      periodsList.reduce((a, b) => a + b, 0) / periodsList.length
    );

    // Step 3: Concatenate ONLY voiced audio chunks from stable blocks.
    // CRITICAL FIX: The previous implementation ran extractCyclesAndAmplitudes on the full
    // rawAudio buffer, which includes silence, consonants, and pauses between words. This caused
    // the peak-picker to track amplitude and period ACROSS silent gaps — inflating jitter by ~5x
    // and shimmer by ~4-5x above actual values. Voiced-only concatenation eliminates this entirely.
    const voicedTotalLen = stableChunks.reduce((sum, c) => sum + c.length, 0);
    const voicedOnlyAudio = new Float32Array(voicedTotalLen);
    let vOffset = 0;
    for (const chunk of stableChunks) {
      voicedOnlyAudio.set(chunk, vOffset);
      vOffset += chunk.length;
    }

    const { periods: pitchPeriods, amplitudes: peakAmplitudes } = extractCyclesAndAmplitudes(
      voicedOnlyAudio,
      sampleRate,
      avgPeriodSamples
    );

    if (pitchPeriods.length < 3 || peakAmplitudes.length < 3) {
      return {
        success: false,
        error: 'Insufficient voiced cycles extracted. Please speak steadily and try again.',
      };
    }

    // Vocal Jitter calculations
    const jitterLocalPercent = calculateJitterLocalPercent(pitchPeriods);
    const jitterRAP = calculateJitterRAP(pitchPeriods);

    // Vocal Shimmer calculations
    const shimmerLocalPercent = calculateShimmerLocalPercent(peakAmplitudes);
    const shimmerDB = calculateShimmerDB(peakAmplitudes);
    const shimmerAPQ3 = calculateShimmerAPQ3(peakAmplitudes);

    return {
      success: true,
      metrics: {
        avgF0,
        minF0,
        maxF0,
        jitterLocalPercent,
        jitterRAP,
        shimmerLocalPercent,
        shimmerDB,
        shimmerAPQ3,
        voicedRatio: voicedBlocks.length / (sampleCount / hopSize),
      },
    };
  }
}
