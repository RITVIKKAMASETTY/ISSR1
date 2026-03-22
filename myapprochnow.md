# Screening Round Pipeline with AliMeeting (Eval_Ali) Dataset

## Project Context
**Screening round for GSoC 2026: Team Communication Processing in Human-Factors Simulated Environment**
Choosing a real multi-speaker meeting dataset ensures realistic team communication scenarios for preprocessing and future analysis.

---

## Notebook 1: Dataset Selection & Sample Analysis
**Goal:** Show understanding of the data and justify it for the project.

### 1. Dataset Info Section
- **Name:** AliMeeting (Eval_Ali)
- **Source link:** https://www.openslr.org/119
- **Language:** Mandarin (Audio enhancement techniques are language-agnostic and operate on acoustic features rather than linguistic content.)
- **Speakers per session:** 2–4
- **Audio types:** Far-field (array), near-field (headset)
- **Why it matters:** Mentors need to see an appropriate, real multi-speaker dataset was chosen.

### 2. Extract Sample
- **Action:** Extract a 3–5 minute segment from one near-field meeting session in the `audio_dir/near/` folder, and convert/save it to `.wav` for processing.
- **Why it matters:** The screening round does not require the full dataset. A short, manageable sample is enough to demonstrate processing skills without heavy compute.

### 3. Exploratory Analysis
- **Action:** 
  - Plot waveform (amplitude over time).
  - Plot spectrogram (frequency content).
  - Show duration, sampling rate, and number of channels.
  - Optional: Play audio inline and add a histogram of amplitude levels.
- **Why it matters:** Visuals confirm that the audio is multi-speaker, noisy, and in need of enhancement.

### 4. Key Questions Section
- **How will it be used?** 
  "This audio sample will be used for preprocessing and enhancement (improving clarity and uniformity), preparing it for team communication analysis such as speaker diarization and coordination patterns."
- **Why is this dataset best?** 
  "Eval_Ali contains real multi-speaker meetings with overlapping speech and natural pauses. It is realistic for testing enhancement algorithms and small enough to handle efficiently for this screening round."

---

## Notebook 2: Audio Enhancement & Evaluation
**Goal:** Show practical skills in audio preprocessing and improving audio quality.

### 1. Load & Inspect Audio
- **Action:** Load the 3–5 minute original audio using `librosa` (`librosa.display.waveplot`) or `pydub`.
- **Why it matters:** Allows mentors to hear the raw audio quality and analyze specific acoustic issues before enhancement.

### 2. Preprocessing Pipeline
- **Resample (16 kHz):** Standardizes audio for all samples and ML models.
- **Mono Conversion:** Converts stereo to mono. Simplifies processing and reduces file size.
- **Noise Reduction:** Removes background noise (e.g., fan, AC). Leaves only speech for better transcription clarity.
- **Volume Normalization:** Balances speaker levels. Ensures quiet speakers are just as audible as loud speakers.

### 3. Show Results: Before vs After Audio
The goal is to demonstrate that the enhancement worked clearly, even to someone who doesn't know audio processing.

* **Visual Waveform Comparison:** 
  - **Action:** Plot original vs enhanced waveform using `matplotlib` + `librosa.display.waveshow`. (Original: noisy spikes, uneven amplitude; Enhanced: smoother, peaks normalized).
  - **Why:** Shows noise reduction visually and validates volume leveling, making the improvement obvious.
* **Spectrogram / Frequency Analysis:** 
  - **Action:** Plot spectrograms using `librosa.display.specshow`. (Noise shows as high-frequency "cloud"; Enhanced audio is cleaner and more structured).
  - **Why:** Shows clarity in both time and frequency, and quantifies improvement.
* **Listen:** 
  - **Action:** Play before and after audio embedded directly in the notebook using `IPython.display.Audio`.
  - **Why:** Audio is a human experience—seeing is good, hearing is better! Mentors can hear the actual difference.

### 4. Quantitative Metrics
- **Action:** Display specific metrics using `librosa` or `numpy` like:
  - **SNR (Signal-to-Noise Ratio):** Higher SNR → better clarity.
  - **RMS (Root Mean Square amplitude):** Shows volume consistency.
- **Why:** Mentors love numbers alongside visuals to provide objective proof.

### 5. Explanation Text & Context (Required Markdown)
Always add a markdown explanation summarizing the impact:
*Example: "Noise reduction removed background fan noise, normalization balanced the volume across speakers, and optional silence removal made the audio ready for further transcription or analysis."*
- **Why it matters:** Mentors aren’t just analyzing the code, they value reasoning. Visual + audio + short explanation is the perfect combination.

### 6. Save Enhanced Sample
- **Action:** Save the processed audio to disk as a `.wav` file.
- **Why it matters:** Prepares data for future team communication analysis and shows the ability to produce reusable outputs.

### Bonus / Optional Extra Features
- **Silence Removal:** Show a before/after plot for the silence removal step, explaining how it removes useless pauses and creates cleaner data for analysis.
- **Speaker Diarization:** Provide rough segments (e.g., Speaker 1 / Speaker 2) over time to show who spoke when, hinting at advanced communication analysis for later phases.

---

## Data Structure of Eval_Ali
- **Use:** `audio_dir/near/*.wav`
- **Why:** Near-field (headset) audio provides clearer speech signals with less background noise, making them ideal for demonstrating audio enhancement techniques compared to far-field mics.
- **Ignore:** `textgrid_dir`, `segments`, `utt2spk`, `wav.scp` (These are used for tasks like speaker diarization, which are unnecessary for the screening round.)
