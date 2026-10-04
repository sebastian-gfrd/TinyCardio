# Clinical and Technical Analysis of PhysioNet Databases
**TinyCardio Biomedical Engineering Documentation**

This document presents an in-depth clinical, pathophysiological, and technical analysis of the four electrocardiography (ECG) datasets organized within this repository: **`sddb`**, **`vfdb`**, **`edb`**, and **`nsrdb`**.

---

## 1. Global Repository Summary

| Database | Project Role | Records | Sampling Rate | Channels | Total Duration | Raw Data Size (.dat) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`nsrdb`** | **Control Group** (Normal Sinus Rhythm) | 18 | **128 Hz** | 2 (ECG1, ECG2) | 437.5 hours | 576.8 MB |
| **`edb`** | **Myocardial Ischemia** (ST-T Dynamic Deviations) | 90 | **250 Hz** | 2 (V1–V5, MLI–III) | 180.0 hours | 463.5 MB |
| **`vfdb`** | **Lethal Arrhythmias** (Malignant Ventricular Ectopy) | 22 | **250 Hz** | 2 (ECG, ECG) | 12.8 hours | 33.0 MB |
| **`sddb`** | **Cardiac Arrest / Sudden Death** (SDDB Holter) | 23 | **250 Hz** | 2 (ECG, ECG) | 446.6 hours | 1,150.0 MB |
| **TOTAL** | — | **153** | — | — | **1,076.9 hours** | **~2,223 MB** |

---

## 2. Individual Database Breakdown

### 2.1. NSRDB — MIT-BIH Normal Sinus Rhythm Database (Control Group)
* **Clinical Purpose:** Establishes the electrophysiological baseline of normal cardiac activity in subjects free of significant arrhythmias, structural heart disease, or autonomic collapse.
* **Patient Cohort:** 18 subjects (5 men aged 26–45; 13 women aged 20–50) evaluated at the Arrhythmia Laboratory of Beth Israel Hospital in Boston.
* **Signal Characteristics:**
  * Sampling Rate: **128 Hz**.
  * Data Encoding: WFDB Format 212 (12-bit resolution per sample).
  * Channels: 2 modified bipolar ECG leads (`ECG1`, `ECG2`).
  * Duration: Continuous ~24-hour ambulatory recordings per subject (total: 437.5 hours).
* **Annotations:**
  * Beat-by-beat annotation files (`.atr`). Dominated by normal sinus beats (`N` or `code 1`), with occasional isolated benign ectopic beats or motion artifacts typical of daily ambulatory activity.

---

### 2.2. EDB — European ST-T Database (Myocardial Ischemia)
* **Clinical Purpose:** Detection and quantification of transient myocardial ischemic episodes through dynamic morphological alterations of the ST-segment and T-wave.
* **Patient Cohort:** 90 ambulatory recordings from 79 patients (70 men and 9 women, mean age 55.3 years) diagnosed with exertional or mixed angina, prior myocardial infarction, and angiographically confirmed coronary artery disease (1-, 2-, or 3-vessel disease: LAD, LCX, RCA).
* **Signal Characteristics:**
  * Sampling Rate: **250 Hz**.
  * Channels: 2 simultaneous leads selected to maximize ischemic visualization across the cardiac wall at risk (modified precordial leads V1–V5 and limb leads MLI, MLIII, D3).
  * Duration: Exactly 2 hours per recording (total: 180 hours).
* **Ischemia Annotation Criteria:**
  * **ST Episode:** Absolute ST-segment displacement $\ge 0.1\text{ mV}$ ($1\text{ mm}$ at standard clinical calibration) lasting at least 30 seconds. Annotations delineate episode onset, peak deviation, and termination.
  * **T-wave Episode:** Sustained changes in T-wave amplitude $\ge 0.12\text{ mV}$.

---

### 2.3. VFDB — MIT-BIH Malignant Ventricular Ectopy Database (Lethal Ventricular Arrhythmias)
* **Clinical Purpose:** Automated identification and instantaneous discrimination of life-threatening ventricular tachycardias and fibrillations from supraventricular rhythms.
* **Patient Cohort:** 22 recordings of approximately 35 minutes each (total: 12.8 hours), selected from ambulatory Holter recordings of patients who experienced major arrhythmogenic events.
* **Signal Characteristics:**
  * Sampling Rate: **250 Hz**.
  * Channels: 2 analog ECG channels.
* **Rhythm Annotations (`.atr`):**
  * Delineate exact transition moments between rhythm states using `AUX` markers:
    * `(VF` or `(VFIB`: Ventricular Fibrillation (chaotic electrical activity, immediate circulatory arrest).
    * `(VFL`: Ventricular Flutter (rapid sinusoidal oscillations, often > 200 bpm).
    * `(VT`: Sustained Ventricular Tachycardia (organized wide QRS complexes).
    * `(AFIB`: Atrial Fibrillation.
    * `(ASYS`: Asystole.
    * `(N`: Restored baseline or normal sinus rhythm.

---

### 2.4. SDDB — Sudden Cardiac Death Holter Database (Cardiac Arrest and Sudden Death)
* **Clinical Purpose:** Predictive analysis of the electrophysiological timeline and autonomic degradation culminating in out-of-hospital Sudden Cardiac Death (SCD).
* **Key Relationship with VFDB:** The 22 extracts of 35 minutes in `vfdb` were originally derived from these identical 23 full Holter tapes compiled by Scott Greenwald at MIT. While `vfdb` isolates only the acute event window, `sddb` provides the complete continuous recording (spanning 4 to 25 hours leading up to, during, and after arrest).
* **Signal Characteristics:**
  * Sampling Rate: **250 Hz**.
  * Channels: 2 ECG channels.
  * Duration: Full continuous recordings (total: 446.6 hours).
* **Critical Metadata:**
  * Header files (`.hea`) explicitly record the timestamp comment `# vfon: HH:MM:SS`, denoting the exact clock time when ventricular fibrillation/tachycardia triggered terminal collapse.
* **Annotations:**
  * Automated beat annotations (`.ari`) for all 23 records.
  * Expert-curated reference annotations (`.atr`) for 12 records in the series.

---

## 3. Methodological and Engineering Considerations for AI / TinyML

When training Deep Learning models (1D-CNNs, Temporal Convolutions) or biomedical digital signal processing pipelines on these datasets, the following constraints must be strictly addressed:

```
                                      ┌────────────────┐
                                      │ Preprocessing  │
                                      └───────┬────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
            Sampling Rate (Fs)                              Amplitude & Calibration
    ┌─────────────────────────────────────┐         ┌─────────────────────────────────────┐
    │ • nsrdb: 128 Hz                     │         │ • nsrdb, edb, vfdb: 200 ADC units/mV│
    │ • edb, vfdb, sddb: 250 Hz           │         │ • sddb: 800 ADC units/mV            │
    │ ➔ Resample to common Fs (250 Hz)    │         │ ➔ Convert to mV: (raw - base) /gain │
    └─────────────────────────────────────┘         └─────────────────────────────────────┘
```

### 3.1. Sampling Rate Discrepancy ($F_s$)
* `nsrdb` is sampled at **128 Hz**, while `edb`, `vfdb`, and `sddb` are sampled at **250 Hz**.
* **Mandatory Action:** Apply polyphase anti-aliasing resampling to all `nsrdb` records (128 Hz $\rightarrow$ 250 Hz) to maintain temporal consistency across the 250-sample model input window.

### 3.2. Physical Amplitude Calibration
* Hardware ADC gains differ across archives:
  * `nsrdb`: 200 ADC units / mV.
  * `edb`: 200 ADC units / mV.
  * `vfdb`: 200 ADC units / mV.
  * `sddb`: 800 ADC units / mV (and 200 in select channels).
* **Mandatory Action:** Convert all raw integer values to calibrated millivolts:
  $$V(\text{mV}) = \frac{\text{ADC}_{\text{raw}} - \text{baseline}}{\text{gain}}$$
  The `utils/ecg_loader.py` module applies this transformation automatically when `physical=True` is specified.

### 3.3. Prevention of Data Leakage
* **Cross-Database Overlap (VFDB vs. SDDB):** Because `vfdb` records are direct excerpts of `sddb` recordings, **never** assign `vfdb/418` to the training set and `sddb/30` to the test set (or vice versa), as they represent the same patient and lethal event.
* **Patient-Level Stratification:** The train / validation / test split must be partitioned strictly **by patient ID**, ensuring no sliding windows from the same subject cross partition boundaries.

### 3.4. Extreme Class Imbalance
* Lethal arrhythmias and acute arrest episodes comprise a small temporal fraction compared to hundreds of hours of baseline sinus rhythm.
* **Mitigation:** The TinyCardio pipeline employs **Binary Focal Loss** ($\gamma = 2.0, \alpha = 0.55$) and balanced window subsampling to ensure high alert recall without suffering from excessive false alarms.
