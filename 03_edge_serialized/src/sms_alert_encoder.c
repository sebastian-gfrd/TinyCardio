/**
 * TinyCardio 2G SMS / LoRa Mesh Alert Payload Encoder
 * File: sms_alert_encoder.c
 */

#include "sms_alert_encoder.h"
#include <stdio.h>
#include <string.h>

/* Simple CRC8 for data integrity */
static uint8_t compute_crc8(const uint8_t* data, size_t len) {
    uint8_t crc = 0x00;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int b = 0; b < 8; b++) {
            if (crc & 0x80) {
                crc = (crc << 1) ^ 0x07;
            } else {
                crc <<= 1;
            }
        }
    }
    return crc;
}

int encode_sms_human_text(const TinyCardioAlert* alert, char* out_buffer, size_t max_len) {
    if (!alert || !out_buffer || max_len < 40) return -1;

    const char* code_str = "NORM";
    if (alert->event_type == ALERT_EVENT_MALIGNANT_ARRHYTHMIA) code_str = "VF/VT";
    else if (alert->event_type == ALERT_EVENT_ACUTE_ISCHEMIA) code_str = "ISCH";
    else if (alert->event_type == ALERT_EVENT_ARREST_PRECURSOR) code_str = "ARREST";

    int written = snprintf(out_buffer, max_len,
                           "TC:D=%04X;E=%s;R=%u%%;HR=%u;T=%us",
                           alert->device_id,
                           code_str,
                           alert->risk_score_pct,
                           alert->heart_rate_bpm,
                           alert->timestamp_sec);
    return written;
}

int encode_sms_binary_hex(const TinyCardioAlert* alert, char* out_hex_buffer, size_t max_len) {
    if (!alert || !out_hex_buffer || max_len < 25) return -1;

    /* 10-byte binary packet structure */
    uint8_t raw[10];
    raw[0] = 0x54; /* 'T' Magic */
    raw[1] = 0x43; /* 'C' Magic */
    raw[2] = (uint8_t)(alert->device_id >> 8);
    raw[3] = (uint8_t)(alert->device_id & 0xFF);
    raw[4] = (uint8_t)(alert->timestamp_sec >> 24);
    raw[5] = (uint8_t)(alert->timestamp_sec >> 16);
    raw[6] = (uint8_t)(alert->timestamp_sec >> 8);
    raw[7] = (uint8_t)(alert->timestamp_sec & 0xFF);
    raw[8] = (uint8_t)(((alert->event_type & 0x07) << 5) | ((alert->risk_score_pct / 4) & 0x1F));
    raw[9] = alert->heart_rate_bpm;

    uint8_t crc = compute_crc8(raw, 10);

    /* Format to 22-char hex string */
    for (int i = 0; i < 10; i++) {
        sprintf(&out_hex_buffer[i * 2], "%02X", raw[i]);
    }
    sprintf(&out_hex_buffer[20], "%02X", crc);
    out_hex_buffer[22] = '\0';

    return 22;
}

bool decode_sms_binary_hex(const char* hex_buffer, TinyCardioAlert* out_alert) {
    if (!hex_buffer || !out_alert || strlen(hex_buffer) < 22) return false;

    uint8_t raw[11];
    for (int i = 0; i < 11; i++) {
        unsigned int val;
        if (sscanf(&hex_buffer[i * 2], "%02X", &val) != 1) return false;
        raw[i] = (uint8_t)val;
    }

    if (raw[0] != 0x54 || raw[1] != 0x43) return false;
    uint8_t expected_crc = compute_crc8(raw, 10);
    if (raw[10] != expected_crc) return false;

    out_alert->device_id = ((uint16_t)raw[2] << 8) | raw[3];
    out_alert->timestamp_sec = ((uint32_t)raw[4] << 24) |
                               ((uint32_t)raw[5] << 16) |
                               ((uint32_t)raw[6] << 8) |
                               (uint32_t)raw[7];
    out_alert->event_type = (AlertEventType)((raw[8] >> 5) & 0x07);
    out_alert->risk_score_pct = (raw[8] & 0x1F) * 4;
    out_alert->heart_rate_bpm = raw[9];

    return true;
}
