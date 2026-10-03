/**
 * TinyCardio 2G SMS / LoRa Mesh Alert Payload Encoder
 * File: sms_alert_encoder.h
 * Generates ultra-compressed telemetry payloads (< 60 bytes) for emergency broadcast.
 */

#ifndef SMS_ALERT_ENCODER_H
#define SMS_ALERT_ENCODER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ALERT_EVENT_NORMAL_RESTORED     = 0x00,
    ALERT_EVENT_MALIGNANT_ARRHYTHMIA = 0x01, /* VF / VT */
    ALERT_EVENT_ACUTE_ISCHEMIA      = 0x02, /* Severe ST deviation */
    ALERT_EVENT_ARREST_PRECURSOR    = 0x03  /* Autonomic collapse */
} AlertEventType;

typedef struct {
    uint16_t device_id;         /* Unique ID of edge node */
    uint32_t timestamp_sec;     /* Time since boot or epoch */
    AlertEventType event_type;  /* Clinical alert code */
    uint8_t risk_score_pct;     /* Calculated risk 0-100% */
    uint8_t heart_rate_bpm;     /* Heart rate in beats per minute */
} TinyCardioAlert;

/**
 * Encodes an alert into a human-readable, single-message compact SMS string (< 60 bytes).
 * Example: "TC:D=1024;E=VF;R=96%;HR=182;TS=3412s"
 */
int encode_sms_human_text(const TinyCardioAlert* alert, char* out_buffer, size_t max_len);

/**
 * Encodes an alert into a compact binary hex string (24 characters, 12 bytes raw payload).
 */
int encode_sms_binary_hex(const TinyCardioAlert* alert, char* out_hex_buffer, size_t max_len);

/**
 * Decodes a binary hex string back into a TinyCardioAlert struct.
 */
bool decode_sms_binary_hex(const char* hex_buffer, TinyCardioAlert* out_alert);

#ifdef __cplusplus
}
#endif

#endif /* SMS_ALERT_ENCODER_H */
