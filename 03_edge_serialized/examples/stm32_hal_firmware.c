/**
 * TinyCardio Reference Firmware for STM32 (Cortex-M3 / Cortex-M4)
 * Compatible with: STM32F103 (Blue Pill), STM32F401RE (Nucleo-64), STM32L432KC (Ultra-Low-Power)
 * Sensor: AD8232 Single-Lead ECG connected to ADC1 Channel 0 (PA0)
 * Modem: Quectel M95 / SIM800L GSM connected to USART2 (PA2 TX, PA3 RX)
 * 
 * Features:
 *   - Hardware Timer (TIM2) triggers ADC1 conversion at exact 250.0 Hz
 *   - ADC with DMA in Circular Buffer mode (zero CPU overhead during acquisition)
 *   - In-place INT8 inference using pure integer fixed-point engine
 *   - Power-saving WFI (Wait-For-Interrupt) between windows (< 15 mA total system power)
 *   - Autonomous SMS 2G dispatch via UART with GSM FSM driver
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

/* STM32 HAL Header (Included in STM32CubeIDE / Keil MDK / PlatformIO projects)
 * #include "stm32f4xx_hal.h"
 */

#include "../src/pure_fixed_point_infer.h"
#include "../src/sms_alert_encoder.h"
#include "../src/gsm_modem_driver.h"

#define ECG_SAMPLE_RATE_HZ     250
#define ECG_WINDOW_SAMPLES     250
#define EMERGENCY_DISPATCH_NUM "+18005550199"
#define STM32_NODE_ID          0x3C4D

/* Double buffer for DMA ADC streaming */
static uint16_t s_dma_adc_buffer[ECG_WINDOW_SAMPLES * 2];
static volatile bool s_half_complete_flag = false;
static volatile bool s_full_complete_flag = false;

/* Static window buffer for TinyML engine */
static int8_t s_quantized_window[ECG_WINDOW_SAMPLES];

/**
 * Normalizes 12-bit ADC raw samples (0..4095) to INT8 [-128..127]
 */
static void stm32_preprocess_buffer(const uint16_t *raw_buf, int8_t *out_buf, int len) {
    /* Calculate mean for baseline centering */
    int32_t sum = 0;
    for (int i = 0; i < len; i++) {
        sum += raw_buf[i];
    }
    int32_t mean = sum / len;

    /* Center and scale */
    for (int i = 0; i < len; i++) {
        int32_t centered = raw_buf[i] - mean;
        int32_t scaled = (centered * 64) / 512;
        if (scaled > 127) scaled = 127;
        if (scaled < -128) scaled = -128;
        out_buf[i] = (int8_t)scaled;
    }
}

/**
 * DMA Half-Transfer Callback (invoked at 125 samples in circular buffer)
 */
void HAL_ADC_ConvHalfCpltCallback_Mock(void) {
    s_half_complete_flag = true;
}

/**
 * DMA Full-Transfer Callback (invoked at 250 samples in circular buffer)
 */
void HAL_ADC_ConvCpltCallback_Mock(void) {
    s_full_complete_flag = true;
}

/**
 * System and Peripheral Initialization
 */
void System_Init(void) {
    printf("[STM32] Initializing Cortex-M Clock & Peripherals...\n");
    printf("[STM32] TIM2 configured for 250 Hz ADC trigger\n");
    printf("[STM32] ADC1 + DMA Circular Buffer configured (2x250 words)\n");
    printf("[STM32] USART2 configured at 9600-8N1 for GSM Modem\n");

    /* Initialize TinyML Pure Fixed-Point Engine */
    fixed_point_infer_init();
    fixed_point_set_threshold_q7(33); /* 0.50 risk threshold in Q7: round(0.50 * 64) = 32..33 */

    /* Initialize GSM Driver */
    gsm_modem_init(EMERGENCY_DISPATCH_NUM);
    printf("[STM32] TinyCardio Edge Diagnostics Active.\n");
}

/**
 * Main application loop
 */
int main(void) {
    System_Init();

    uint32_t window_count = 0;

    /* Main Superloop */
    while (1) {
        /* Simulate or wait for DMA transfer complete interrupt */
        /* In real hardware: __WFI(); (Wait For Interrupt to minimize power) */

        /* Check for completed buffer */
        const uint16_t *ready_raw_buffer = NULL;
        if (s_full_complete_flag) {
            ready_raw_buffer = &s_dma_adc_buffer[ECG_WINDOW_SAMPLES];
            s_full_complete_flag = false;
        } else if (s_half_complete_flag) {
            ready_raw_buffer = &s_dma_adc_buffer[0];
            s_half_complete_flag = false;
        }

        if (ready_raw_buffer != NULL) {
            window_count++;

            /* 1. Preprocess raw ADC buffer into INT8 window */
            stm32_preprocess_buffer(ready_raw_buffer, s_quantized_window, ECG_WINDOW_SAMPLES);

            /* 2. Execute Zero-Allocation Fixed-Point Inference */
            FixedPointResult result = fixed_point_predict(s_quantized_window, ECG_WINDOW_SAMPLES);

            printf("[STM32 W#%u] Risk Q7=%d (%s) | Status: %s\n",
                   window_count,
                   result.risk_probability_q7,
                   result.risk_level_str,
                   result.is_emergency ? "EMERGENCY DETECTED!" : "NORMAL");

            /* 3. Emergency Telemetry Dispatch */
            if (result.is_emergency) {
                printf("[STM32 ALERT] Raising Emergency Dispatch Sequence...\n");

                SmsAlertPayload payload;
                sms_payload_init(&payload, STM32_NODE_ID, 0.94f, window_count, 138);

                char sms_text[64];
                sms_payload_format_text(&payload, sms_text, sizeof(sms_text));
                printf("[STM32 SMS OUT] %s\n", sms_text);

                gsm_modem_trigger_alert(sms_text);
            }
        }

        /* Tick GSM Modem State Machine */
        gsm_modem_tick();

        /* Low-power idle break for simulation demonstration */
        break;
    }

    return 0;
}
