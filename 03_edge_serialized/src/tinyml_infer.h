/**
 * TinyCardio Embedded Inference Engine
 * File: tinyml_infer.h
 * Zero dynamic memory allocation (malloc-free).
 * Suitable for bare-metal Cortex-M0+, Cortex-M4, ESP32, and RISC-V.
 */

#ifndef TINYML_INFER_H
#define TINYML_INFER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Result structure for MCU inference */
typedef struct {
    float risk_score;       /* Risk probability in [0.0f, 1.0f] */
    uint8_t risk_percent;   /* Risk score as integer 0 - 100 % */
    bool is_alert;          /* True if risk_score >= threshold */
    uint32_t cycles_taken;  /* Benchmark cycles or timer ticks */
} TinyCardioResult;

/**
 * Initialize TinyCardio internal state and static buffers.
 */
void tinyml_init(void);

/**
 * Set the alert threshold for triage (default: 0.50f).
 */
void tinyml_set_threshold(float threshold);

/**
 * Normalize raw ADC samples into normalized 8-bit integer window [-128, 127].
 * Zero dynamic memory allocation.
 *
 * @param raw_input    Pointer to 250 raw ECG integer samples.
 * @param quant_output Pointer to destination 250 int8_t samples.
 */
void tinyml_preprocess_window(const int16_t* raw_input, int8_t* quant_output);

/**
 * Executes full deterministic forward pass on 250 quantized samples.
 *
 * @param input_window Pointer to 250 int8_t normalized ECG samples.
 * @param result       Pointer to TinyCardioResult output structure.
 */
void tinyml_predict(const int8_t* input_window, TinyCardioResult* result);

/**
 * Convenience function executing forward pass directly on normalized floats.
 *
 * @param float_window Pointer to 250 normalized float samples.
 * @param result       Pointer to TinyCardioResult output structure.
 */
void tinyml_predict_float(const float* float_window, TinyCardioResult* result);

#ifdef __cplusplus
}
#endif

#endif /* TINYML_INFER_H */
