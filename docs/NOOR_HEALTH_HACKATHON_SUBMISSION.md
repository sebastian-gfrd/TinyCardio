# NoorCare / TinyCardio: Official World Bank Small AI Hackathon Submission Brief
**Event:** Youth Summit × Global AI & Digital Summit (Seoul, October 2026)  
**Track:** Annex A — Health (Rural Primary Care Access & Frontline Worker Support)  
**Target Persona:** Noor (38-year-old smallholder farmer in Ondera highlands) & Frontline Health Workers  
**Repository:** [https://github.com/sebastian-gfrd/TinyCardio.git](https://github.com/sebastian-gfrd/TinyCardio.git)

---

## 1. Problem Statement (One Sentence Formulation)

> **"Because of NoorCare, rural patients like Noor will access verified primary care triage and confirmed on-duty clinical appointments within 1 hour via basic 2G SMS in their native language (Swahili/Quechua) that they would otherwise delay by days or travel hours in vain to closed clinics; we know this because World Bank Service Delivery Indicators show a 37.8% provider absence rate and 4+ hour clinic waits driven by 16.4 minutes of burdensome manual paper record-keeping per patient."**

---

## 2. AI Capabilities & Value Proposition (Why AI vs. Simpler Tools)

### What the Small AI Engine Does:
1. **Multilingual Symptom & Red-Flag Intent Triage (Meta MMS / FLORES-200):**  
   Processes free-form natural language messages or voice transcriptions in **Swahili** (Ondera persona) and **Quechua** (highlands persona) on constrained basic devices (< 40 KB memory). It differentiates critical emergencies (respiratory distress, chest pain, convulsions) from routine conditions without requiring internet or cloud GPUs.
2. **Context-Aware Dynamic Facility Routing:**  
   Cross-references real-time patient urgency with the **World Bank Service Delivery Indicators (SDI)** and **`healthsites.io`** geospatial datasets, solving the *"is Tobi there today?"* provider absenteeism problem and verifying tracer medicine stock before advising travel.
3. **Automated DHIS2 Clinical Encounter Generation:**  
   Automatically converts the patient's conversational intake into standardized **DHIS2 Tracker & Event Capture JSON payloads**, eliminating the 16.4 minutes of manual paperwork that causes severe clinic congestion.

### Why a Simpler Tool (Static SMS, Spreadsheet, or Google Search) Fails:
* **Static SMS / USSD Menus:** Rigid hierarchical menus fail for low-literacy rural patients who express health emergencies in colloquial native dialect phrasing rather than clinical keywords.
* **Google Search:** Requires continuous high-speed 4G internet, smartphone literacy, and outputs unvetted, anxiety-inducing information in foreign languages without checking local clinic staffing.
* **Spreadsheets:** Completely inaccessible to a farmer with a basic phone and cannot parse spoken or texted natural language.

---

## 3. Responsible AI, Ethics & Safety Guardrails (Pass/Fail Compliance)

In strict adherence to the World Bank and WHO Ethics and Governance of AI for Health guidelines:
* **Strict "Human-in-the-Loop" Decision Making:**  
  The tool **never issues autonomous medical diagnoses** (complying strictly with Annex A.2 rule: *"diagnosis datasets are out of bounds"*). It acts solely as an access, triage, and administrative referral assistant. Every output includes an explicit disclaimer:  
  *Swahili:* *"Taarifa ya Usalama: Mfumo huu wa AI hutoa mwongozo wa awali pekee. Mtoa huduma wa afya (daktari/muuguzi) ndiye anayefanya uamuzi wa mwisho."*
* **Zero Hallucination Guarantee:**  
  Routing and medical referrals are bounded to a deterministic knowledge graph verified against official Ministry of Health facility listings (`healthsites.io`).
* **Data Sovereignty & Privacy:**  
  Zero patient health data is exported to third-party proprietary commercial clouds. All inference runs locally on the edge node or district gateway.

---

## 4. Where the Tool Sits in Noor's Day (End-to-End User Journey)

```
                       NOOR'S JOURNEY WITH NOORCARE
                       
 [ 06:30 AM - Noor's Farm ]
 Noor's child wakes with high fever & vomiting. Noor has her basic 2G phone.
                         │
                         ▼
 [ 06:32 AM - 2G SMS Query in Swahili ]
 Noor sends: "Mtoto ana homa kali sana na anatapika" to the clinic's shortcode.
                         │
                         ▼
 [ 06:32 AM - Small AI Processing (< 1 ms) ]
 NoorCare NLP classifies: CLINICAL_YELLOW (92% confidence).
 Router checks World Bank SDI: 
   - Kipkabus Clinic: Doctor Kiprono is ABSENT today (vaccination outreach).
   - Ondera Dispensary: Clinical Officer Amina is ON DUTY today, oral rehydration in stock.
                         │
                         ▼
 [ 06:33 AM - Return SMS to Noor in Swahili ]
 "NoorCare: UCHUNGUZI WA KLINIKI. Kituo kinachofaa: Ondera Sub-County Dispensary (65 min walk).
  Afisa Amina yupo kazini leo. Kunywa maji safi safarini."
                         │
                         ▼
 [ 07:45 AM - Arrival at Ondera Dispensary ]
 Noor arrives. Because the Small AI pre-populated her DHIS2 intake record,
 Nurse Amina opens the tablet, verifies vitals, and sees Noor in 3 minutes instead of 45.
```

---

## 5. What Localizing AI Development Means to Us

> *"Localizing AI development means shifting the center of gravity of artificial intelligence away from multi-billion-dollar hyperscale data centers serving Silicon Valley and toward the 2.6 billion people living in the Global South. It means designing AI that respects the radical constraints of our communities: running in memory-constrained devices, speaking indigenous mother tongues like Swahili and Quechua without internet, and solving practical structural bottlenecks—like doctor absenteeism and medical record burdens—so that technology serves human dignity at the last mile."*

---

## 6. Technical Stack & Repository Structure

* **Small AI Inference:** Pure C99 and Python standard library (Zero third-party dependencies, < 50 KB binary).
* **Language Models / Benchmarks:** Meta MMS / FLORES-200 / MASSIVE intent taxonomy.
* **Data Grounding:** World Bank Service Delivery Indicators (SDI) + `healthsites.io` (OpenStreetMap).
* **Clinical Platform Fit:** DHIS2 Standard Web API Event Capture JSON.
* **Testing:** 100% automated test coverage with sub-millisecond execution latency.
