# TinyCardio: 3-Minute Video Pitch Script
**Event:** World Bank Group & Korea MSIT/MOFE Global AI Summit Hackathon 2026  
**Category:** Artificial Intelligence for Global Health Equity & Frugal Innovation  
**Duration:** Exactly 3 Minutes (180 Seconds)

---

## Pitch Structure & Timetable

| Time Range | Section | Visual Focus / On-Screen Content | Key Message |
|:---:|:---|:---|:---|
| **0:00 – 0:30** | The Crisis & Inequality | Map of rural cardiac deserts; statistics on global CVD mortality in LMICs | Over 75% of cardiac deaths occur in low- and middle-income countries due to lack of doctors and connectivity. |
| **0:30 – 1:10** | The Breakthrough | Hardware close-up: $4.30 USD board with RP2040 + AD8232 + 2G module | TinyCardio brings hospital-grade AI to a $4 USD chip that requires no internet, no smartphone, and only 540 bytes of ROM. |
| **1:10 – 1:55** | Live Demonstration | Real-time split screen: Patient ECG wave, CLI monitor, and instant SMS alert | Ingestion of PhysioNet patient 418 (VFDB); ventricular fibrillation detected in <0.8 ms; SMS dispatched. |
| **1:55 – 2:30** | Technical Rigor & Architecture | 3-Pillar diagram: JAX GPU (RTX 5070) -> CPU Gateway -> C99 INT8 MCU | Cross-engine parity, 93.2%–100% sensitivity on lethal events, zero false alarms on normal rhythm. |
| **2:30 – 3:00** | Scalability & World Bank SDGs | Deployment model: Community Health Workers, SDG 3.4 & 9.5, Call to Action | Empowering the bottom 40% with scalable, autonomous edge healthcare. |

---

## Full Script & Directional Notes

### [0:00 – 0:30] Hook: The Cardiac Care Desert
* **Visual:** Fade in from black. World map highlighting rural Sub-Saharan Africa, South Asia, and Latin America. Graphic showing 1 cardiologist per 500,000 people.
* **Speaker (Voiceover / On-Camera):**
  > "Every two seconds, someone suffers a fatal cardiac event. Over 75% of these deaths happen in low- and middle-income nations—not because heart disease cannot be treated, but because it is detected too late.
  > 
  > In rural clinics, modern diagnostic ECGs costing thousands of dollars do not exist. Cloud-connected smartwatches require 4G smartphones and expensive cellular plans that the bottom 40% simply cannot afford. 
  > 
  > What if lifesaving artificial intelligence didn't need the cloud, didn't need a smartphone, and cost less than five dollars?"

---

### [0:30 – 1:10] The Solution: TinyCardio
* **Visual:** Cut to presenter holding a breadboard prototype with an RP2040 chip, single-lead AD8232 sensor, and a SIM800L module. Overlay text: *"TinyCardio Edge Node — BOM: $4.30 USD"*.
* **Speaker:**
  > "Meet **TinyCardio**: the world’s first autonomous, ultra-frugal Edge AI cardiac triage monitor engineered for global health equity.
  > 
  > Instead of running massive neural networks in distant server farms, TinyCardio runs a high-precision 1D Convolutional Neural Network directly inside the internal memory of microcontrollers costing as little as seventy cents.
  > 
  > The entire AI model takes only **540 bytes of Flash memory** and under **2.5 kilobytes of RAM**. It runs continuously on coin cells or small solar panels, analyzing single-lead ECG rhythms in real time at 250 Hertz, with zero memory allocations and zero internet dependency."

---

### [1:10 – 1:55] Live Demonstration: Real-Time Life-Saving Triage
* **Visual:** Screen capture of TinyCardio in action. 
  * Left side: `02_model_cpu/cli_monitor.py` rendering the live ECG oscilloscope in real time.
  * Right side: Physical phone receiving an incoming SMS.
* **Speaker:**
  > "Let's see it in action. Here we stream unseen clinical telemetry from PhysioNet Record 418—a patient suffering sudden ventricular fibrillation.
  > 
  > Watch the monitor: within **less than a millisecond**, TinyCardio's inference engine identifies the lethal rhythm, calculates a 98% risk score, and raises an emergency flag.
  > 
  > Because there is no internet, TinyCardio's autonomous driver wakes a low-cost 2G modem and sends this: an ultra-compressed **32-character SMS message** containing the patient ID, risk score, and heart rate directly to the local Community Health Worker's basic feature phone.
  > 
  > When normal sinus rhythm is observed, the system stays silent—zero false alarms, zero unnecessary battery drain."

---

### [1:55 – 2:30] Engineering Rigor: The Three Pillars
* **Visual:** Fast-paced animated architecture diagram highlighting the repository structure:
  1. JAX GPU Acceleration (RTX 5070)
  2. CPU Gateway Server & Dashboard
  3. C99 INT8 Serialized Firmware
* **Speaker:**
  > "Behind this frugal device lies industrial-grade engineering organized into three transparent pillars:
  > 
  > **Pillar 1:** An advanced JAX and CUDA pipeline trained on four gold-standard PhysioNet databases, utilizing Focal Loss and biological noise augmentations to reach 100% sensitivity on malignant ventricular flutter and 93.2% on sudden cardiac arrest.
  > 
  > **Pillar 2:** A deterministic CPU gateway server with a live HTML5 diagnostic canvas for rural clinic laptops.
  > 
  > **Pillar 3:** An INT8 fixed-point C99 engine with verified bit-for-bit numerical parity against our GPU training models, complete with turnkey firmware for Raspberry Pi Pico, STM32, and ESP32."

---

### [2:30 – 3:00] Impact & Call to Action
* **Visual:** Photo montage of Community Health Workers in rural communities. On-screen logos: UN SDG 3.4, UN SDG 9.5, World Bank Group.
* **Speaker:**
  > "TinyCardio aligns directly with UN Sustainable Development Goal 3.4 to reduce non-communicable disease mortality by one-third, and Goal 9.5 for frugal innovation.
  > 
  > By bringing the cost of cardiac edge AI down to four dollars and thirty cents, we transform rural primary healthcare from reactive despair into proactive, life-saving triage.
  > 
  > Artificial intelligence should not be a luxury for the privileged. With TinyCardio, it becomes a universal human right.
  > 
  > Thank you."
* **Visual:** Fade out to repository URL: `https://github.com/sebastian-gfrd/TinyCardio` and team credits.
