# Análisis Clínico y Técnico de las Bases de Datos PhysioNet

Este documento presenta el análisis detallado, fisiopatológico y técnico de los cuatro conjuntos de datos de electrocardiografía (ECG) organizados en este espacio de trabajo: **`sddb`**, **`vfdb`**, **`edb`** y **`nsrdb`**.

---

## 1. Resumen Global del Repositorio

| Base de Datos | Rol en el Proyecto | Registros | Frecuencia de Muestreo | Canales | Duración Total | Datos Crudos (.dat) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`nsrdb`** | **Grupo de Control** (Ritmo Sinusal Normal) | 18 | **128 Hz** | 2 (ECG1, ECG2) | 437.5 horas | 576.8 MB |
| **`edb`** | **Isquemia Miocárdica** (Alteraciones ST-T) | 90 | **250 Hz** | 2 (V1–V5, MLI–III) | 180.0 horas | 463.5 MB |
| **`vfdb`** | **Arritmias Letales** (Ectopia Ventricular Maligna) | 22 | **250 Hz** | 2 (ECG, ECG) | 12.8 horas | 33.0 MB |
| **`sddb`** | **Paro Cardíaco / Muerte Súbita** (Holter SDDB) | 23 | **250 Hz** | 2 (ECG, ECG) | 446.6 horas | 1,150.0 MB |
| **TOTAL** | — | **153** | — | — | **1,076.9 horas** | **~2,223 MB** |

---

## 2. Análisis Individual de las Bases de Datos

### 2.1. NSRDB — MIT-BIH Normal Sinus Rhythm Database (Grupo de Control)
* **Objetivo Clínico:** Establecer la línea base de normalidad electrofisiológica sin cardiopatías ni arritmias complejas.
* **Población:** 18 pacientes (5 hombres entre 26 y 45 años; 13 mujeres entre 20 y 50 años) evaluados en el Laboratorio de Arritmias del Hospital Beth Israel de Boston.
* **Características de la Señal:**
  * Frecuencia de muestreo: **128 Hz**.
  * Formato: WFDB Format 212 (resolución de 12 bits).
  * Canales: 2 derivaciones bipolares modificadas (`ECG1`, `ECG2`).
  * Duración: Grabaciones continuas de ~24 horas por sujeto (total: 437.5 horas).
* **Anotaciones:**
  * Archivos `.atr` con marcación beat-by-beat (latido a latido). Predominan latidos sinusales normales (`N` o `code 1`), con escasos latidos ectópicos aislados o artefactos de movimiento propios de la vida ambulatoria.

---

### 2.2. EDB — European ST-T Database (Isquemia Miocárdica)
* **Objetivo Clínico:** Detección y cuantificación de episodios isquémicos transitorios miocárdicos mediante variaciones dinámicas del segmento ST y la onda T.
* **Población:** 90 registros de 79 pacientes (70 hombres y 9 mujeres, edad media de 55.3 años) con angina de esfuerzo o mixta, infarto previo de miocardio y enfermedad coronaria confirmada (1, 2 o 3 vasos arteriales: LAD, LCX, RCA).
* **Características de la Señal:**
  * Frecuencia de muestreo: **250 Hz**.
  * Canales: 2 derivaciones simultáneas elegidas para maximizar la visibilidad de la pared cardíaca en riesgo (derivaciones precordiales modificadas V1 a V5 y bipolares de extremidades MLI, MLIII, D3).
  * Duración: Exactamente 2 horas por registro (total: 180 horas).
* **Criterios de Anotación de Isquemia:**
  * **Episodio ST:** Desplazamiento absoluto del segmento ST $\ge 0.1\text{ mV}$ ($1\text{ mm}$ a calibración estándar) con una duración mínima de 30 segundos. Las anotaciones marcan el inicio, el pico de máxima desviación y el final del episodio.
  * **Episodio T:** Cambios en la amplitud de la onda T $\ge 0.12\text{ mV}$ con duración sostenida.

---

### 2.3. VFDB — MIT-BIH Malignant Ventricular Ectopy Database (Arritmias Ventriculares Letales)
* **Objetivo Clínico:** Identificación y discriminación automática de taquicardias y fibrilaciones ventriculares potencialmente mortales frente a otros ritmos.
* **Población:** 22 registros de aproximadamente 35 minutos de duración cada uno (total: 12.8 horas), seleccionados a partir de grabaciones Holter ambulatorias de pacientes que sufrieron eventos arrítmicos mayores.
* **Características de la Señal:**
  * Frecuencia de muestreo: **250 Hz**.
  * Canales: 2 canales analógicos de ECG.
* **Anotaciones de Ritmo (`.atr`):**
  * Indican el instante exacto del cambio de ritmo mediante etiquetas `AUX`:
    * `(VF` o `(VFIB`: Fibrilación ventricular (actividad eléctrica caótica, colapso hemodinámico inmediato).
    * `(VFL`: Flutter ventricular (ritmo sinusoidal rápido, frecuentemente > 200 bpm).
    * `(VT`: Taquicardia ventricular sostenida (complejos QRS anchos organizados).
    * `(AFIB`: Fibrilación auricular.
    * `(ASYS`: Asistolia.
    * `(N`: Ritmo sinusal normal recuperado o basal.

---

### 2.4. SDDB — Sudden Cardiac Death Holter Database (Paros Cardíacos y Muerte Súbita)
* **Objetivo Clínico:** Análisis predictivo de la progresión electrofisiológica que culmina en paro cardíaco y muerte súbita (MSC).
* **Relación Clave con VFDB:** Los 22 extractos de 35 minutos de `vfdb` fueron extraídos originalmente de estas mismas 23 cintas Holter completas recopiladas por Scott Greenwald en el MIT. Mientras `vfdb` solo contiene la ventana crítica del evento, `sddb` ofrece la grabación continua completa (entre 4 y 25 horas antes y durante el paro).
* **Características de la Señal:**
  * Frecuencia de muestreo: **250 Hz**.
  * Canales: 2 canales de ECG.
  * Duración: Grabaciones completas (total: 446.6 horas).
* **Metadatos Críticos:**
  * En los encabezados (`.hea`) se incluye la anotación `# vfon: HH:MM:SS`, que especifica la hora exacta en la que se inicia la fibrilación/taquicardia ventricular que lleva al paro.
* **Anotaciones:**
  * Archivos `.ari`: Anotaciones automatizadas latido a latido para todos los registros.
  * Archivos `.atr`: Anotaciones revisadas y curadas por especialistas para 12 registros de la serie.

---

## 3. Consideraciones Metodológicas y Técnicas para IA / Machine Learning

Al entrenar algoritmos de Aprendizaje Profundo (CNN, Transformers, LSTM) o de Procesamiento de Señales Biomédicas, deben tenerse en cuenta las siguientes particularidades:

```
                                      ┌───────────────┐
                                      │  Preprocesado │
                                      └───────┬───────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
         Frecuencia de Muestreo (Fs)                     Amplitud y Calibración
   ┌─────────────────────────────────────┐         ┌─────────────────────────────────────┐
   │ • nsrdb: 128 Hz                     │         │ • nsrdb, edb, vfdb: 200 ADC units/mV│
   │ • edb, vfdb, sddb: 250 Hz           │         │ • sddb: 800 ADC units/mV            │
   │ ➔ Re-muestrear a Fs común (ej. 250) │         │ ➔ Convertir a mV: (raw - base) /gain│
   └─────────────────────────────────────┘         └─────────────────────────────────────┘
```

### 3.1. Discrepancia en la Frecuencia de Muestreo ($F_s$)
* `nsrdb` está muestreado a **128 Hz**, mientras que `edb`, `vfdb` y `sddb` están a **250 Hz**.
* **Acción requerida:** Aplicar re-muestreo polifásico (anti-aliasing) a las señales de `nsrdb` (de 128 Hz a 250 Hz) o estandarizar todas las señales a una frecuencia unificada (p. ej. 250 Hz o 200 Hz).

### 3.2. Calibración de Amplitud (Unidades Físicas)
* Los factores de ganancia varían:
  * `nsrdb`: 200 unidades ADC por mV.
  * `edb`: 200 unidades ADC por mV.
  * `vfdb`: 200 unidades ADC por mV.
  * `sddb`: 800 unidades ADC por mV (y 200 en registros puntuales).
* **Acción requerida:** Utilizar la conversión física a milivoltios:
  $$V(\text{mV}) = \frac{\text{ADC}_{\text{raw}} - \text{baseline}}{\text{gain}}$$
  El módulo `utils/ecg_loader.py` realiza esta conversión automáticamente si se indica `physical=True`.

### 3.3. Prevención de Fuga de Datos (Data Leakage)
* **Entre VFDB y SDDB:** Debido a que los registros de `vfdb` son ventanas extraídas directamente de `sddb`, **nunca** se debe colocar el registro `vfdb/418` en el conjunto de entrenamiento y `sddb/30` en el conjunto de prueba (o viceversa), ya que corresponden al mismo paciente y evento.
* **Separación por Paciente:** La división en entrenamiento / validación / prueba debe hacerse **a nivel de paciente/registro**, no mezclando ventanas del mismo registro entre sets.

### 3.4. Desbalance Extremo de Clases
* Las arritmias letales y los paros cardíacos representan una fracción temporal muy reducida en comparación con las cientos de horas de ritmo sinusal. Se recomienda el uso de funciones de pérdida ponderadas (Focal Loss, Weighted Cross-Entropy) o técnicas de submuestreo de ventanas sinusales.
