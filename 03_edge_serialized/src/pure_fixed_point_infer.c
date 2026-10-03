/**
 * TinyCardio 100% Pure Integer Fixed-Point Inference Engine
 * File: pure_fixed_point_infer.c
 * Implements 1D-CNN with zero floating-point operations (FPU-free).
 */

#include "pure_fixed_point_infer.h"
#include "../include/model_weights.h"
#include <string.h>

static int8_t g_thresh_q7 = 64; /* 64/128 = 50.0% threshold */

/* Static RAM buffers (Total RAM < 2.2 KB) */
static int8_t s_fixed_layer1[TINYCARDIO_CONV1_OUT_LEN * TINYCARDIO_CONV1_OUT_CH]; /* 125 * 8 = 1000 bytes */
static int8_t s_fixed_layer2[TINYCARDIO_CONV2_OUT_LEN * TINYCARDIO_CONV2_OUT_CH]; /* 63 * 16 = 1008 bytes */
static int8_t s_fixed_gap[TINYCARDIO_DENSE_IN_FEAT];                              /* 16 bytes */

/* 33-entry LUT for Sigmoid in Q7 format (z from -4.0 to +4.0 in steps of 0.25) */
static const uint8_t s_sigmoid_lut_q7[33] = {
    2,   3,   4,   5,   6,   8,  11,  14,
   18,  23,  30,  37,  46,  55,  64,  73,
   82,  91,  98, 105, 110, 114, 117, 119,
  121, 122, 123, 124, 125, 125, 126, 126, 127
};

static inline int8_t fast_sigmoid_q7(int16_t logit_q4) {
    /* logit_q4 has step size 0.25 (1.0 = 4). Zero point is at index 16 */
    int idx = 16 + (logit_q4);
    if (idx < 0) return 0;
    if (idx > 32) return 127;
    return (int8_t)s_sigmoid_lut_q7[idx];
}

void fixed_point_init(void) {
    g_thresh_q7 = 64;
    memset(s_fixed_layer1, 0, sizeof(s_fixed_layer1));
    memset(s_fixed_layer2, 0, sizeof(s_fixed_layer2));
    memset(s_fixed_gap, 0, sizeof(s_fixed_gap));
}

void fixed_point_set_threshold_q7(int8_t threshold_q7) {
    g_thresh_q7 = threshold_q7;
}

void fixed_point_predict(const int8_t* input_q7, FixedPointResult* result) {
    /* Layer 1: Conv1D (k=5, in_ch=1, out_ch=8, stride=2, SAME) */
    const int in_len1 = TINYCARDIO_INPUT_LEN;
    const int out_len1 = TINYCARDIO_CONV1_OUT_LEN;
    const int k_w1 = TINYCARDIO_CONV1_KERNEL;
    const int stride1 = TINYCARDIO_CONV1_STRIDE;
    const int out_ch1 = TINYCARDIO_CONV1_OUT_CH;

    for (int i = 0; i < out_len1; i++) {
        int in_center = i * stride1;
        int in_start = in_center - (k_w1 / 2);

        for (int oc = 0; oc < out_ch1; oc++) {
            /* Scale bias float to integer accumulator (scale factor ~ 128) */
            int32_t acc = (int32_t)(g_tinycardio_conv1_b[oc] * 128.0f);

            for (int k = 0; k < k_w1; k++) {
                int in_idx = in_start + k;
                int8_t in_val = (in_idx >= 0 && in_idx < in_len1) ? input_q7[in_idx] : 0;
                int w_idx = k * out_ch1 + oc;
                int8_t w_val = g_tinycardio_conv1_w[w_idx];
                acc += ((int32_t)in_val * (int32_t)w_val) >> 5;
            }

            /* ReLU and saturate to [0, 127] */
            if (acc < 0) acc = 0;
            else if (acc > 127) acc = 127;
            s_fixed_layer1[i * out_ch1 + oc] = (int8_t)acc;
        }
    }

    /* Layer 2: Conv1D (k=3, in_ch=8, out_ch=16, stride=2, SAME) */
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
            int32_t acc = (int32_t)(g_tinycardio_conv2_b[oc] * 128.0f);

            for (int k = 0; k < k_w2; k++) {
                int in_idx = in_start + k;
                if (in_idx >= 0 && in_idx < in_len2) {
                    for (int ic = 0; ic < in_ch2; ic++) {
                        int8_t in_val = s_fixed_layer1[in_idx * in_ch2 + ic];
                        int w_idx = k * (in_ch2 * out_ch2) + ic * out_ch2 + oc;
                        int8_t w_val = g_tinycardio_conv2_w[w_idx];
                        acc += ((int32_t)in_val * (int32_t)w_val) >> 6;
                    }
                }
            }

            if (acc < 0) acc = 0;
            else if (acc > 127) acc = 127;
            s_fixed_layer2[i * out_ch2 + oc] = (int8_t)acc;
        }
    }

    /* Layer 3: Global Average Pooling (over spatial dimension 63) */
    for (int oc = 0; oc < out_ch2; oc++) {
        int32_t sum = 0;
        for (int i = 0; i < out_len2; i++) {
            sum += s_fixed_layer2[i * out_ch2 + oc];
        }
        s_fixed_gap[oc] = (int8_t)(sum / out_len2);
    }

    /* Layer 4: Dense Unit (16 -> 1) */
    int32_t logit_acc = (int32_t)(g_tinycardio_dense_b[0] * 64.0f);
    for (int oc = 0; oc < TINYCARDIO_DENSE_IN_FEAT; oc++) {
        int8_t in_val = s_fixed_gap[oc];
        int8_t w_val = g_tinycardio_dense_w[oc];
        logit_acc += ((int32_t)in_val * (int32_t)w_val) >> 5;
    }

    /* Compute Sigmoid via 33-entry integer LUT */
    int16_t logit_q4 = (int16_t)(logit_acc / 16);
    int8_t risk_q7 = fast_sigmoid_q7(logit_q4);

    if (result) {
        result->risk_q7 = risk_q7;
        result->risk_percent = (uint8_t)(((uint16_t)risk_q7 * 100) / 127);
        result->is_alert = (risk_q7 >= g_thresh_q7);
    }
}
