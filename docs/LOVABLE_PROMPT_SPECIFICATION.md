# TinyCardio Dispatch: Specification and Master Prompt for Lovable.dev
**World Bank Group & Korea MSIT/MOFE Global AI Summit Hackathon 2026**  
**Rural Medical Triage & Ambulance Dispatch Platform Powered by 2G SMS Telemetry**

---

## 1. Platform Vision & Healthcare Purpose

**TinyCardio Dispatch** is a mission-critical emergency command dashboard engineered for rural health dispensaries, district triage hospitals, and Community Health Worker (CHW) networks across low- and middle-income countries (LMICs).

The platform bridges the critical **"Last Mile"** gap: when an ultra-low-cost TinyCardio edge monitor (\$4.30 USD) detects a life-threatening ventricular arrhythmia or pre-arrest autonomic collapse on a rural patient, it autonomously broadcasts a compact 2G SMS telemetry payload (<60 bytes). **TinyCardio Dispatch** ingests this transmission, geolocates the patient's homestead on an interactive GIS map, displays their pre-registered clinical history, and empowers healthcare dispatchers to **mobilize an ambulance, equipped mototaxi, or community health worker in under 60 seconds**, while automatically generating a reassurance SMS confirmation back to the patient.

---

## 2. Copy-and-Paste Master Prompt for Lovable.dev

> [!TIP]
> **Instructions for Lovable:**  
> 1. Navigate to [Lovable.dev](https://lovable.dev) and start a new project.
> 2. Copy and paste the prompt below directly into the Lovable generation input.
> 3. Lovable will immediately scaffold the complete application with React, Tailwind CSS, Lucide Icons, and interactive mapping.

```markdown
Create a high-impact, mission-critical medical emergency dashboard called "TinyCardio Dispatch".
This platform is built for rural health centers, district hospitals, and Community Health Worker networks in low- and middle-income countries (World Bank Group & Korea MSIT Hackathon 2026).
It processes ultra-low-bandwidth 2G SMS emergency alerts (<60 bytes) sent by $4.30 USD TinyML wearable ECG monitors worn by vulnerable rural cardiac patients.

Key Features & UI Structure:

1. Top Navigation Bar:
   - Left: Heartbeat icon with pulsing red dot and title "TinyCardio Dispatch" (subtitle: "Rural Rapid Triage & Emergency Response System").
   - Center: Location badge: "Centro de Salud Rural San Gabriel (District 4 - Rural Health Network)".
   - Right: Real-time telemetry status: "2G GSM Network: ACTIVE (94% Signal)", "Ambulances Ready: 3/4", live digital clock, and Dark/Light mode toggle.

2. Impact & Triage KPI Banner (Top Grid - 5 Cards):
   - "Active Critical Emergencies": 2 (Pulsing Red badge, lethal arrhythmias in Golden Hour).
   - "Moderate Ischemia Alerts": 1 (Amber badge, ST elevation / angina).
   - "Average Response Time": "12.8 min" (World Bank Target: < 20 min).
   - "Rural Population Protected": "1,420 patients" (Bottom 40% Initiative).
   - "Monitoring Cost per Patient": "$4.30 USD (BOM) / $0.08 month".

3. Split-Screen Operations Center (Main Grid):
   A. Left Column (42% width) - "Real-Time Emergency Triage Feed":
      - Filter tabs: [All (3)], [Critical Red (2)], [Moderate Amber (1)], [En Route (1)].
      - Toggle for audible siren alert sound.
      - Patient Alert Cards sorted by medical urgency:
        * Card Header: Severity badge ("CRITICAL: Ventricular Fibrillation" or "ALERT: Acute Ischemia") + Golden Hour countdown timer (e.g., "56:18 remaining").
        * Patient details: Name (e.g. "María Quispe Huamán", 62 yrs), Device ID (#TC-1024), Battery level (89%).
        * Vital signs badges: Heart Rate (182 bpm), AI Risk Score (96% - TinyML INT8).
        * Location: "Vereda El Roble, Sector Alto (Km 14)".
        * Primary Button: "Evaluate Rhythm & Dispatch Aid".

   B. Right Column (58% width) - "Geospatial Rescue & Resource Dispatch Map":
      - Interactive map view (using Leaflet / Mapbox or clean SVG mock map):
        * Pin for "Hospital San Gabriel" (Central Base).
        * Pulsing Red pins for patients with active critical cardiac arrest alerts.
        * Amber pin for patients with acute ischemia.
        * Moving / Static markers for emergency units:
          - Ambulance 01 (Type II - Central Base)
          - Rescue Mototaxi 02 (Equipped with portable AED)
          - Community Health Worker Rosa Medina (CHW-04 on motorcycle)
        * When a patient is selected, draw a route line from the nearest resource to the patient's home coordinates with ETA (e.g., "Rural Road 4 - 12 min").

4. Patient Emergency Details & Dispatch Modal (Opens on card click):
   - Patient Profile: Full Name, Age, Blood Type (e.g., O+), Chronic Conditions ("Prior MI 2024, Hypertension"), Current Medications ("Enalapril 10mg, Aspirin"), Emergency Contact ("Son: Juan Quispe +51 984 551 203").
   - Raw 2G SMS Payload Inspector: Shows the actual received 37-character SMS:
     `TC:D=1024;E=VF;R=96%;HR=182;TS=3412s` with chips breaking down: Device ID, Event Code (VFDB), Risk Probability, Heart Rate, and CRC8 Validated.
   - Dynamic Animated ECG Oscilloscope: An animated HTML5 canvas drawing the lethal ECG waveform (rapid chaotic ventricular fibrillation waves or ST-elevation segment) with grid background.
   - Medical Protocol Recommendation: "Protocol 01: Ventricular Fibrillation. Immediate dispatch of unit equipped with Automated External Defibrillator (AED)."
   - Emergency Resource Dispatch Selector:
     * Radio cards to choose vehicle:
       [Ambulance 01 - 4x4 All-Terrain (ETA 18 min)]
       [Rescue Mototaxi 02 - With Portable AED (ETA 11 min)] (Recommended)
       [Community Health Worker CHW-04 on Motorcycle (ETA 07 min)]
     * Notes input for driver/paramedic.
   - Big Action Button: "CONFIRM EMERGENCY DISPATCH".
   - Automated Reassurance SMS Action: Upon dispatch, show toast notification:
     "Reassurance SMS transmitted to patient: 'TinyCardio: Emergency assistance dispatched. Rescue Mototaxi 02 en route to your location. ETA: 11 min. Please remain lying down.'"
   - Updates patient card badge to "EN ROUTE".

5. Floating Live Simulation Drawer (For Hackathon Judges & Demos):
   - A discreet button at bottom-right "Hackathon Jury Demonstration Toolbar":
     * Button: "Simulate SMS Alert: Ventricular Fibrillation (María Quispe - TC-1024)"
     * Button: "Simulate SMS Alert: Acute Ischemia (Carlos Mamani - TC-2048)"
     * Button: "Simulate SMS Alert: Cardiac Arrest Precursor (Esperanza Flores - TC-3072)"
     * Button: "Reset Demo Data"
   - Clicking these buttons injects real-time alerts into the feed with sound and animations to showcase the live 2G reception capability.

Design System:
- Professional, mission-critical emergency healthcare aesthetic.
- Color palette: Deep medical navy (#0F172A), Slate grey (#1E293B), Alert Crimson (#EF4444), Warning Amber (#F59E0B), Vital Green (#10B981), Cyan accents (#06B6D4).
- Clean typography, high contrast, smooth transitions, and mobile/tablet responsive layout for rural healthcare workers.
```

---

## 3. Data Model & API Contract (TypeScript Interfaces)

For configuring or manually adjusting types within your Lovable project:

```typescript
export type AlertSeverity = 'CRITICO_ROJO' | 'ALERTA_AMARILLO' | 'OBSERVACION_AZUL';
export type AlertStatus = 'PENDIENTE_DESPACHO' | 'EN_CAMINO' | 'RESUELTO';
export type EventCode = 'VF' | 'ISCH' | 'ARR';

export interface Coordinates {
  lat: number;
  lng: number;
}

export interface PatientRecord {
  device_id: number;
  name: string;
  age: number;
  gender: string;
  blood_type: string;
  location_name: string;
  coordinates: Coordinates;
  medical_history: string;
  current_medication: string;
  emergency_contact: string;
  assigned_chw: string;
  device_battery_pct: number;
}

export interface TelemetryData {
  device_id: number;
  event_code: EventCode;
  event_label: string;
  risk_score_pct: number;
  heart_rate_bpm: number;
  battery_pct: number;
  raw_sms: string;
  clinical_protocol: string;
}

export interface DispatchInfo {
  dispatched_at_iso: string;
  resource_id: string;
  resource_name: string;
  estimated_arrival_minutes: number;
  operator_notes: string;
  return_sms_sent: {
    destination_number: string;
    sms_content: string;
  };
}

export interface EmergencyAlert {
  alert_id: string;
  created_at_epoch: number;
  created_at_iso: string;
  status: AlertStatus;
  severity: AlertSeverity;
  golden_hour_seconds_left: number;
  patient: PatientRecord;
  telemetry: TelemetryData;
  dispatch_info?: DispatchInfo | null;
}
```

---

## 4. Connecting Lovable with the Local Python Backend

To link the Lovable frontend directly to your local TinyCardio telemetry hub:

1. **Launch the Telemetry Server in your terminal:**
   ```bash
   python3 02_model_cpu/sms_telemetry_webhook.py --port 8090
   ```
2. **Expose locally via tunnel (Cloudflare Tunnels or ngrok):**
   ```bash
   npx localtunnel --port 8090
   # Or using ngrok:
   ngrok http 8090
   ```
3. **Configure the API endpoint in Lovable:**  
   Set the public URL (e.g. `https://tinycardio-hub.loca.lt`) in your Lovable configuration:
   - `GET /api/alerts`: Fetches the active emergency queue.
   - `POST /api/sms/incoming`: Ingests incoming 2G SMS payloads.
   - `POST /api/dispatch/ambulance`: Confirms resource dispatch and queues reassurance SMS.
   - `GET /api/alerts/stream`: Real-time Server-Sent Events (SSE) push channel.

4. **Simulate live events during presentations:**
   ```bash
   # Send Ventricular Fibrillation alert
   python3 02_model_cpu/simulate_telemetry_event.py --device 1024 --event VF --risk 97 --hr 185

   # Send Cardiac Arrest Precursor alert
   python3 02_model_cpu/simulate_telemetry_event.py --device 3072 --event ARR --risk 99 --hr 195
   ```
