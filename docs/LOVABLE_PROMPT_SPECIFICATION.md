# TinyCardio Dispatch: Especificación y Prompt Maestro para Lovable.dev
**World Bank Group & Korea MSIT/MOFE Global AI Summit Hackathon 2026**  
**Plataforma de Despacho y Triaje Médico Rural impulsada por Telemetría 2G SMS**

---

## 1. Visión y Propósito de la Plataforma

**TinyCardio Dispatch** es el centro de comando web de respuesta rápida para centros de salud rurales, hospitales distritales y redes de Promotores de Salud Comunitarios (*Community Health Workers - CHWs*).

La plataforma resuelve el eslabón crítico de la **"Última Milla"**: cuando el dispositivo de borde TinyCardio (\$4.30 USD) detecta una arritmia maligna o infarto en el pecho de un paciente rural, transmite un paquete SMS 2G ultracomprimido (<60 bytes). **TinyCardio Dispatch** recibe esta señal, la geolocaliza en un mapa rural interactivo, presenta el perfil clínico del paciente, y permite al personal médico **despachar una ambulancia o mototaxi con desfibrilador en menos de 60 segundos**, enviando automáticamente un SMS de confirmación al paciente.

---

## 2. Prompt Maestro Copiar y Pegar para Lovable.dev

> [!TIP]
> **Instrucciones para Lovable:**  
> 1. Ve a [Lovable.dev](https://lovable.dev) e inicia un nuevo proyecto.
> 2. Copia y pega el texto del recuadro inferior en el campo de creación de Lovable.
> 3. Lovable generará automáticamente la aplicación completa con diseño React, Tailwind CSS, Lucide Icons y mapa interactivo.

```markdown
Create a high-impact, mission-critical medical emergency dashboard called "TinyCardio Dispatch".
This platform is built for rural health centers, district hospitals, and Community Health Worker networks in low- and middle-income countries (World Bank Group & Korea MSIT Hackathon 2026).
It processes ultra-low-bandwidth 2G SMS emergency alerts (<60 bytes) sent by $4.30 USD TinyML wearable ECG monitors worn by vulnerable rural cardiac patients.

Key Features & UI Structure:

1. Top Navigation Bar:
   - Left: Heartbeat icon with pulsing red dot and title "TinyCardio Dispatch" (subtitle: "Sistema Central de Triaje y Respuesta Rápida Rural").
   - Center: Location badge: "Centro de Salud Rural San Gabriel (Distrito 4 - Red Veredal)".
   - Right: Real-time telemetry status: "Red 2G GSM: ACTIVA (94% Señal)", "Ambulancias Listas: 3/4", live digital clock, and Dark/Light mode toggle.

2. Impact & Triage KPI Banner (Top Grid - 5 Cards):
   - "Emergencias Críticas Activas": 2 (Pulsing Red badge, lethal arrhythmias in Golden Hour).
   - "Alertas Isquémicas Moderadas": 1 (Amber badge, ST elevation / angina).
   - "Tiempo Promedio de Respuesta": "12.8 min" (World Bank Target: < 20 min).
   - "Población Rural Protegida": "1,420 pacientes" (Iniciativa Bottom 40%).
   - "Costo de Monitoreo / Paciente": "$4.30 USD (BOM) / $0.08 mes".

3. Split-Screen Operations Center (Main Grid):
   A. Left Column (42% width) - "Bandeja de Triage en Tiempo Real":
      - Filter tabs: [Todas (3)], [Críticas Rojas (2)], [Moderadas (1)], [En Camino (1)].
      - Toggle for audible siren alert sound.
      - Patient Alert Cards sorted by medical urgency:
        * Card Header: Severity badge ("CRÍTICO: Fibrilación Ventricular" or "ALERTA: Isquemia Aguda") + Golden Hour countdown timer (e.g., "56:18 restantes").
        * Patient details: Name (e.g. "María Quispe Huamán", 62 años), Device ID (#TC-1024), Battery level (89%).
        * Vital signs badges: Heart Rate (182 lpm), AI Risk Score (96% - TinyML INT8).
        * Location: "Vereda El Roble, Sector Alto (Km 14)".
        * Primary Button: "Evaluar Trazado & Despachar Auxilio".

   B. Right Column (58% width) - "Mapa Geoespacial de Rescate y Recursos":
      - Interactive map view (using Leaflet / Mapbox or clean SVG mock map):
        * Pin for "Hospital San Gabriel" (Central Base).
        * Pulsing Red pins for patients with active critical cardiac arrest alerts.
        * Amber pin for patients with acute ischemia.
        * Moving / Static markers for emergency units:
          - Ambulancia 01 (Tipo II - Base Central)
          - Mototaxi Médica 02 (Equipada con DEA portátil)
          - Promotora de Salud Comunitaria Rosa Medina (CHW-04 en motocicleta)
        * When a patient is selected, draw a route line from the nearest resource to the patient's home coordinates with ETA (e.g., "Ruta Rural 4 - 12 min").

4. Patient Emergency Details & Dispatch Modal (Opens on card click):
   - Patient Profile: Full Name, Age, Blood Type (e.g., O+), Chronic Conditions ("Infarto previo 2024, Hipertensa"), Current Medications ("Enalapril 10mg, Aspirina"), Emergency Contact ("Hijo: Juan Quispe +51 984 551 203").
   - Raw 2G SMS Payload Inspector: Shows the actual received 37-character SMS:
     `TC:D=1024;E=VF;R=96%;HR=182;TS=3412s` with chips breaking down: Device ID, Event Code (VFDB), Risk Probability, Heart Rate, and CRC8 Validated.
   - Dynamic Animated ECG Oscilloscope: An animated HTML5 canvas drawing the lethal ECG waveform (rapid chaotic ventricular fibrillation waves or ST-elevation segment) with grid background.
   - Medical Protocol Recommendation: "Protocolo 01: Fibrilación Ventricular. Despacho inmediato de unidad con Desfibrilador Externo Automático (DEA)."
   - Emergency Resource Dispatch Selector:
     * Radio cards to choose vehicle:
       [Ambulancia 01 - 4x4 Todo Terreno (ETA 18 min)]
       [Mototaxi Médica 02 - Con DEA portátil (ETA 11 min)] (Recomendada)
       [Promotor Comunitario CHW-04 en Moto (ETA 07 min)]
     * Notes input for driver/paramedic.
   - Big Action Button: "CONFIRMAR DESPACHO DE EMERGENCIA".
   - Automated Reassurance SMS Action: Upon dispatch, show toast notification:
     "SMS de reaseguramiento transmitido al paciente: 'TinyCardio: Auxilio despachado. Mototaxi Médica 02 en camino a su ubicación. ETA: 11 min. Manténgase recostado.'"
   - Updates patient card badge to "EN CAMINO".

5. Floating Live Simulation Drawer (For Hackathon Judges & Demos):
   - A discreet button at bottom-right "Demostración para Jurado Hackathon":
     * Button: "Simular Alerta SMS: Fibrilación Ventricular (María Quispe - TC-1024)"
     * Button: "Simular Alerta SMS: Isquemia Aguda (Carlos Mamani - TC-2048)"
     * Button: "Simular Alerta SMS: Pre-Paro Cardíaco (Esperanza Flores - TC-3072)"
     * Button: "Restablecer Datos de Demostración"
   - Clicking these buttons injects real-time alerts into the feed with sound and animations to showcase the live 2G reception capability.

Design System:
- Professional, mission-critical emergency healthcare aesthetic.
- Color palette: Deep medical navy (#0F172A), Slate grey (#1E293B), Alert Crimson (#EF4444), Warning Amber (#F59E0B), Vital Green (#10B981), Cyan accents (#06B6D4).
- Clean typography, high contrast, smooth transitions, and mobile/tablet responsive layout for rural healthcare workers.
```

---

## 3. Modelo de Datos y Contrato de API (TypeScript Interfaces)

Si deseas configurar o editar manualmente los tipos en el proyecto Lovable, utiliza la siguiente estructura tipada:

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

## 4. Conexión Opcional con el Backend Local en Python

Si deseas conectar la plataforma en Lovable directamente con tu servidor local TinyCardio:

1. **Inicia el Servidor de Telemetría en tu terminal:**
   ```bash
   python3 02_model_cpu/sms_telemetry_webhook.py --port 8090
   ```
2. **Exponer localmente vía túnel (ngrok o Cloudflare Tunnels):**
   ```bash
   npx localtunnel --port 8090
   # O con ngrok:
   ngrok http 8090
   ```
3. **Configurar la URL en Lovable:**  
   Pega la URL pública generada (ej. `https://tinycardio-hub.loca.lt`) en la variable de entorno o configuración de fetch de Lovable:
   - `GET /api/alerts`: Obtiene la lista activa de emergencias.
   - `POST /api/sms/incoming`: Recibe nuevos mensajes SMS.
   - `POST /api/dispatch/ambulance`: Confirma el despacho y emite el SMS de retorno.
   - `GET /api/alerts/stream`: Canal en vivo SSE (*Server-Sent Events*).

4. **Simular eventos en tiempo real durante la presentación:**
   ```bash
   # Enviar alerta de Fibrilación Ventricular
   python3 02_model_cpu/simulate_telemetry_event.py --device 1024 --event VF --risk 97 --hr 185

   # Enviar alerta de Pre-Paro Cardíaco
   python3 02_model_cpu/simulate_telemetry_event.py --device 3072 --event ARR --risk 99 --hr 195
   ```
