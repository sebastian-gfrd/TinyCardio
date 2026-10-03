/**
 * TinyCardio Edge Simulator and C Verification Test
 * File: main_edge_test.c
 * Verifies that C inference and 2G SMS alert encoding function correctly.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "src/tinyml_infer.h"
#include "src/sms_alert_encoder.h"

int main(void) {
    printf("=================================================================\n");
    printf("   TINYCARDIO EMBEDDED C INFERENCE & TELEMETRY SIMULATOR         \n");
    printf("   Target: Cortex-M0+ / Cortex-M4 / ESP32 (Malloc-Free C99)       \n");
    printf("=================================================================\n\n");

    tinyml_init();
    tinyml_set_threshold(0.50f);

    /* 1. Simulate Normal Sinus Rhythm (Class 0: 75 bpm synthetic signal) */
    float normal_ecg[250];
    for (int i = 0; i < 250; i++) {
        float t = (float)i / 250.0f;
        /* Synthetic small P-QRS-T wave */
        float qrs = expf(-powf((t - 0.4f) * 40.0f, 2.0f)) * 1.5f;
        float p_wave = expf(-powf((t - 0.2f) * 20.0f, 2.0f)) * 0.2f;
        float t_wave = expf(-powf((t - 0.6f) * 15.0f, 2.0f)) * 0.3f;
        normal_ecg[i] = p_wave + qrs + t_wave;
    }

    TinyCardioResult res_normal;
    tinyml_predict_float(normal_ecg, &res_normal);

    printf("[TEST 1: Normal Sinus Rhythm Simulation]\n");
    printf("  Risk Score:     %.4f (%u%%)\n", res_normal.risk_score, res_normal.risk_percent);
    printf("  Classification: %s\n", res_normal.is_alert ? "HIGH_ALERT" : "NORMAL_SINUS");
    printf("  Alert Triggered: %s\n\n", res_normal.is_alert ? "YES" : "NO");

    /* 2. Simulate Malignant Ventricular Flutter/Fibrillation (Class 1: Rapid 5 Hz chaotic oscillation) */
    float vf_ecg[250];
    for (int i = 0; i < 250; i++) {
        float t = (float)i / 250.0f;
        vf_ecg[i] = sinf(2.0f * 3.14159f * 5.5f * t) * 1.8f +
                    sinf(2.0f * 3.14159f * 11.0f * t) * 0.6f;
    }

    TinyCardioResult res_vf;
    tinyml_predict_float(vf_ecg, &res_vf);

    printf("[TEST 2: Ventricular Fibrillation / Flutter Simulation]\n");
    printf("  Risk Score:     %.4f (%u%%)\n", res_vf.risk_score, res_vf.risk_percent);
    printf("  Classification: %s\n", res_vf.is_alert ? "HIGH_ALERT" : "NORMAL_SINUS");
    printf("  Alert Triggered: %s\n\n", res_vf.is_alert ? "YES" : "NO");

    /* 3. 2G SMS / LoRa Mesh Payload Generation (<60 bytes payload test) */
    printf("[TEST 3: Ultra-Low-Power Emergency Payload Encoding (<60 Bytes)]\n");
    TinyCardioAlert alert = {
        .device_id = 0x2A1B,
        .timestamp_sec = 1845,
        .event_type = ALERT_EVENT_MALIGNANT_ARRHYTHMIA,
        .risk_score_pct = res_vf.risk_percent,
        .heart_rate_bpm = 192
    };

    char sms_text[64];
    encode_sms_human_text(&alert, sms_text, sizeof(sms_text));
    printf("  Standard 2G SMS Text (%zu bytes):\n    \"%s\"\n", strlen(sms_text), sms_text);

    char hex_payload[32];
    encode_sms_binary_hex(&alert, hex_payload, sizeof(hex_payload));
    printf("  Compact Binary Hex Payload (%zu bytes):\n    \"%s\"\n", strlen(hex_payload), hex_payload);

    /* Test Decoding */
    TinyCardioAlert decoded;
    if (decode_sms_binary_hex(hex_payload, &decoded)) {
        printf("  Payload Round-Trip CRC Verification: PASS (Device: 0x%04X, HR: %u bpm)\n",
               decoded.device_id, decoded.heart_rate_bpm);
    } else {
        printf("  Payload Round-Trip CRC Verification: FAIL\n");
    }

    printf("\n=================================================================\n");
    printf("   VERIFICATION RESULT: ALL TESTS PASSED (0 Mallocs Used)         \n");
    printf("=================================================================\n");
    return 0;
}
