/**
 * TinyCardio 100% Pure Integer Fixed-Point Inference Engine
 * File: pure_fixed_point_infer.h
 * Zero floating-point instructions (no float, no double, no FPU required).
 * Optimized for ARM Cortex-M0+ (RP2040), basic RISC-V (ESP32-C3), and low-power MCUs.
 */

#ifndef PURE_FIXED_POINT_INFER_H
#define PURE_FIXED_POINT_INFER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int8_t risk_q7;        /* Risk score in Q7 format: 0 to 127 (0% to 100%) */
    uint8_t risk_percent;  /* Risk score as integer percentage: 0 to 100% */
    bool is_alert;         /* True if risk >= alert threshold */
} FixedPointResult;

/**
 * Initialize fixed-point engine state and static buffers.
 */
void fixed_point_init(void);

/**
 * Set the integer alert threshold in Q7 (e.g. 64 for 50%).
 */
void fixed_point_set_threshold_q7(int8_t threshold_q7);

/**
 * Executes 100% integer fixed-point forward pass on 250 int8_t ECG samples.
 * Uses 32-bit accumulators with arithmetic right-shifts for fractional scaling.
 *
 * @param input_q7 Pointer to 250 int8_t samples.
 * @param result   Pointer to FixedPointResult structure.
 */
void fixed_point_predict(const int8_t* input_q7, FixedPointResult* result);

#ifdef __cplusplus
}
#endif

#endif /* PURE_FIXED_POINT_INFER_H */
