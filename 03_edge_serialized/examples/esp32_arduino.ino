/**
 * TinyCardio ESP32 / ESP32-C3 Arduino Firmware Sketch
 * Hardware: ESP32-C3 (RISC-V @ 160 MHz) or ESP32 (Xtensa Dual Core)
 * Sensor: AD8232 on GPIO 34 (ADC1_CH6)
 * Modem / Radios: Hardware Serial2 for SIM800L / LoRa SX1278
 */

#include <Arduino.h>
#include "../src/tinyml_infer.h"
#include "../src/sms_alert_encoder.h"

#define ECG_PIN                34
#define SAMPLING_RATE_HZ       250
#define WINDOW_SIZE            250
#define ALERT_THRESHOLD        0.50f
#define EMERGENCY_NUMBER       "+18005550199"

static int16_t ecg_raw_buffer[WINDOW_SIZE];
static int8_t ecg_quant_buffer[WINDOW_SIZE];
static int sample_ptr = 0;
static hw_timer_t* sampling_timer = NULL;
static volatile bool new_sample_ready = false;
static volatile int16_t latest_adc_val = 0;

void IRAM_ATTR onSampleTimer() {
    latest_adc_val = (int16_t)analogRead(ECG_PIN);
    new_sample_ready = true;
}

void setup() {
    Serial.begin(115200);
    Serial.println("=================================================");
    Serial.println(" TinyCardio ESP32 Edge Telemetry Node Starting   ");
    Serial.println("=================================================");

    analogReadResolution(12); // 12-bit ADC (0 - 4095)
    tinyml_init();
    tinyml_set_threshold(ALERT_THRESHOLD);

    // Setup 250 Hz hardware timer interrupt (1,000,000 / 250 = 4000 microseconds)
    sampling_timer = timerBegin(0, 80, true);
    timerAttachInterrupt(sampling_timer, &onSampleTimer, true);
    timerAlarmWrite(sampling_timer, 4000, true);
    timerAlarmEnable(sampling_timer);

    Serial.println("Timer configured: 250 Hz. Monitoring patient...");
}

void loop() {
    if (new_sample_ready) {
        new_sample_ready = false;
        ecg_raw_buffer[sample_ptr++] = latest_adc_val;

        if (sample_ptr >= WINDOW_SIZE) {
            sample_ptr = 0;

            // Zero-malloc preprocessing
            tinyml_preprocess_window(ecg_raw_buffer, ecg_quant_buffer);

            // Execute INT8 1D-CNN inference
            TinyCardioResult res;
            tinyml_predict(ecg_quant_buffer, &res);

            Serial.printf("[INFER] Risk: %u%% | Status: %s\n",
                          res.risk_percent, res.is_alert ? "ALERT" : "NORMAL");

            if (res.is_alert) {
                TinyCardioAlert alert;
                alert.device_id = 0x3E12;
                alert.timestamp_sec = millis() / 1000;
                alert.event_type = ALERT_EVENT_MALIGNANT_ARRHYTHMIA;
                alert.risk_score_pct = res.risk_percent;
                alert.heart_rate_bpm = 185;

                char sms[64];
                encode_sms_human_text(&alert, sms, sizeof(sms));
                Serial.printf("[EMERGENCY-SMS] Dispatching: %s\n", sms);

                char hex_payload[32];
                encode_sms_binary_hex(&alert, hex_payload, sizeof(hex_payload));
                Serial.printf("[LORA-BROADCAST] Packet: %s (%d bytes)\n", hex_payload, strlen(hex_payload));
            }
        }
    }
}
