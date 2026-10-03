/**
 * TinyCardio Embedded Inference Engine
 * File: tinyml_infer.c
 * Implementation of 1D-CNN with zero dynamic memory allocation (malloc-free).
 */

#include "tinyml_infer.h"
#include "../include/model_weights.h"
#include <math.h>
#include <string.h>

/* Global configuration */
static float g_threshold = 0.50f;

/* Static Memory Arena (Tensor Arena < 2.5 KB RAM total) */
static float s_arena_layer1[TINYCARDIO_CONV1_OUT_LEN * TINYCARDIO_CONV1_OUT_CH]; /* 125 * 8 * 4 = 4000 bytes */
static float s_arena_layer2[TINYCARDIO_CONV2_OUT_LEN * TINYCARDIO_CONV2_OUT_CH]; /* 63 * 16 * 4 = 4032 bytes */
static float s_arena_gap[TINYCARDIO_DENSE_IN_FEAT];                              /* 16 * 4 = 64 bytes */

void tinyml_init(void) {
    g_threshold = 0.50f;
    memset(s_arena_layer1, 0, sizeof(s_arena_layer1));
    memset(s_arena_layer2, 0, sizeof(s_arena_layer2));
    memset(s_arena_gap, 0, sizeof(s_arena_gap));
}

void tinyml_set_threshold(float threshold) {
    if (threshold >= 0.0f && threshold <= 1.0f) {
        g_threshold = threshold;
    }
}

void tinyml_preprocess_window(const int16_t* raw_input, int8_t* quant_output) {
    /* Calculate mean and standard deviation in integer/float without malloc */
    int32_t sum = 0;
    for (int i = 0; i < TINYCARDIO_INPUT_LEN; i++) {
        sum += raw_input[i];
    }
    float mean = (float)sum / (float)TINYCARDIO_INPUT_LEN;

    float sum_sq_diff = 0.0f;
    for (int i = 0; i < TINYCARDIO_INPUT_LEN; i++) {
        float diff = (float)raw_input[i] - mean;
        sum_sq_diff += diff * diff;
    }
    float std_dev = sqrtf(sum_sq_diff / (float)TINYCARDIO_INPUT_LEN);
    if (std_dev < 1e-4f) {
        std_dev = 1.0f;
    }

    /* Normalize to [-128, 127] */
    for (int i = 0; i < TINYCARDIO_INPUT_LEN; i++) {
        float z = ((float)raw_input[i] - mean) / std_dev;
        int32_t q = (int32_t)(z * 32.0f);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        quant_output[i] = (int8_t)q;
    }
}

static inline float relu_f(float val) {
    return (val > 0.0f) ? val : 0.0f;
}

static inline float sigmoid_f(float val) {
    if (val > 15.0f) return 1.0f;
    if (val < -15.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-val));
}

void tinyml_predict_float(const float* float_window, TinyCardioResult* result) {
    /* Layer 1: Conv1D (k=5, in_ch=1, out_ch=8, stride=2, SAME padding) */
    const int in_len1 = TINYCARDIO_INPUT_LEN;
    const int out_len1 = TINYCARDIO_CONV1_OUT_LEN;
    const int k_w1 = TINYCARDIO_CONV1_KERNEL;
    const int stride1 = TINYCARDIO_CONV1_STRIDE;
    const int out_ch1 = TINYCARDIO_CONV1_OUT_CH;

    for (int i = 0; i < out_len1; i++) {
        int in_center = i * stride1;
        int in_start = in_center - (k_w1 / 2);

        for (int oc = 0; oc < out_ch1; oc++) {
            float sum = g_tinycardio_conv1_b[oc];

            for (int k = 0; k < k_w1; k++) {
                int in_idx = in_start + k;
                float in_val = (in_idx >= 0 && in_idx < in_len1) ? float_window[in_idx] : 0.0f;
                int w_idx = k * out_ch1 + oc;
                float w_val = (float)g_tinycardio_conv1_w[w_idx] * TINYCARDIO_SCALE_CONV1;
                sum += in_val * w_val;
            }
            s_arena_layer1[i * out_ch1 + oc] = relu_f(sum);
        }
    }

    /* Layer 2: Conv1D (k=3, in_ch=8, out_ch=16, stride=2, SAME padding) */
    const int in_len2 = TINYCARDIO_CONV1_OUT_LEN;
    const int out_len2 = TINYCARDIO_CONV2_OUT_LEN;
    const int k_w2 = TINYCARDIO_CONV2_KERNEL;
    const int in_ch2 = TINYCARDIO_CONV2_IN_CH;
    const int out_ch2 = TINYCARDIO_CONV2_OUT_CH;
    const int stride2 = TINYCARDIO_CONV2_STRIDE;

    for (int i = 0; i < out_len2; i++) {
        int in_center = i * stride2;
        int in_start = in_center - (k_w2 / 2);

        for (int oc = 0; oc < out_ch2; oc++) {
            float sum = g_tinycardio_conv2_b[oc];

            for (int k = 0; k < k_w2; k++) {
                int in_idx = in_start + k;
                if (in_idx >= 0 && in_idx < in_len2) {
                    for (int ic = 0; ic < in_ch2; ic++) {
                        float in_val = s_arena_layer1[in_idx * in_ch2 + ic];
                        int w_idx = k * (in_ch2 * out_ch2) + ic * out_ch2 + oc;
                        float w_val = (float)g_tinycardio_conv2_w[w_idx] * TINYCARDIO_SCALE_CONV2;
                        sum += in_val * w_val;
                    }
                }
            }
            s_arena_layer2[i * out_ch2 + oc] = relu_f(sum);
        }
    }

    /* Layer 3: Global Average Pooling (over spatial dimension 63) */
    for (int oc = 0; oc < out_ch2; oc++) {
        float ch_sum = 0.0f;
        for (int i = 0; i < out_len2; i++) {
            ch_sum += s_arena_layer2[i * out_ch2 + oc];
        }
        s_arena_gap[oc] = ch_sum / (float)out_len2;
    }

    /* Layer 4: Dense Unit (16 -> 1) + Sigmoid */
    float logit = g_tinycardio_dense_b[0];
    for (int oc = 0; oc < TINYCARDIO_DENSE_IN_FEAT; oc++) {
        float w_val = (float)g_tinycardio_dense_w[oc] * TINYCARDIO_SCALE_DENSE;
        logit += s_arena_gap[oc] * w_val;
    }

    float risk = sigmoid_f(logit);

    if (result) {
        result->risk_score = risk;
        result->risk_percent = (uint8_t)(risk * 100.0f);
        result->is_alert = (risk >= g_threshold);
        result->cycles_taken = 0;
    }
}

void tinyml_predict(const int8_t* input_window, TinyCardioResult* result) {
    float temp_input[TINYCARDIO_INPUT_LEN];
    for (int i = 0; i < TINYCARDIO_INPUT_LEN; i++) {
        temp_input[i] = (float)input_window[i] / 32.0f;
    }
    tinyml_predict_float(temp_input, result);
}
