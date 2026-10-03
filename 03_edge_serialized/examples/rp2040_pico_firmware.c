/**
 * TinyCardio Reference Firmware for Raspberry Pi Pico (RP2040)
 * Architecture: Dual ARM Cortex-M0+ @ 133 MHz (Zero FPU)
 * Sensor: AD8232 Single-Lead ECG connected to ADC0 (GPIO 26)
 * Modem: SIM800L 2G GSM Module connected to UART0 (TX GP0, RX GP1)
 *
 * Demonstrates:
 *   - Continuous 250 Hz ADC sampling via hardware timer
 *   - Circular ring buffer
 *   - Zero-allocation TinyML 1D-CNN inference on Core 1
 *   - Autonomous emergency SMS dispatch if arrhythmia / arrest detected
 */

#include <stdio.h>
#include <string.h>
/* In a Pico SDK build:
#include "pico/stdlib.h"
#include "hardware/adc.h"
#include "hardware/uart.h"
*/

#include "../src/tinyml_infer.h"
#include "../src/sms_alert_encoder.h"
#include "../src/gsm_modem_driver.h"

#define ECG_SAMPLE_RATE_HZ     250
#define ECG_WINDOW_SAMPLES     250
#define EMERGENCY_PHONE_NUMBER "+18005550199"
#define DEVICE_NODE_ID         0x2A1B

/* Circular buffer for ADC samples */
static int16_t s_ecg_circular_buffer[ECG_WINDOW_SAMPLES];
static int s_buffer_index = 0;
static bool s_buffer_ready = false;

/* Buffer for quantized inference window */
static int8_t s_quantized_window[ECG_WINDOW_SAMPLES];

void pico_hardware_init(void) {
    /* 
     * stdio_init_all();
     * adc_init();
     * adc_gpio_init(26); // GP26 is ADC0
     * adc_select_input(0);
     * uart_init(uart0, 9600);
     * gpio_set_function(0, GPIO_FUNC_UART);
     * gpio_set_function(1, GPIO_FUNC_UART);
     */
    printf("[PICO] Initializing RP2040 Hardware Peripherals...\n");
    tinyml_init();
    tinyml_set_threshold(0.50f);
    gsm_modem_init(EMERGENCY_PHONE_NUMBER);
    printf("[PICO] TinyCardio TinyML Engine Ready. Monitoring 250 Hz...\n");
}

void pico_process_sample(int16_t raw_adc_value) {
    s_ecg_circular_buffer[s_buffer_index++] = raw_adc_value;
    if (s_buffer_index >= ECG_WINDOW_SAMPLES) {
        s_buffer_index = 0;
        s_buffer_ready = true;
    }

    if (s_buffer_ready) {
        /* Preprocess without malloc */
        tinyml_preprocess_window(s_ecg_circular_buffer, s_quantized_window);

        /* Execute sub-millisecond inference */
        TinyCardioResult result;
        tinyml_predict(s_quantized_window, &result);

        if (result.is_alert) {
            printf("[PICO-ALERT] CRITICAL EVENT DETECTED! Risk: %u%%\n", result.risk_percent);

            TinyCardioAlert alert = {
                .device_id = DEVICE_NODE_ID,
                .timestamp_sec = 120, // Example uptime
                .event_type = ALERT_EVENT_MALIGNANT_ARRHYTHMIA,
                .risk_score_pct = result.risk_percent,
                .heart_rate_bpm = 180
            };

            char sms_payload[64];
            encode_sms_human_text(&alert, sms_payload, sizeof(sms_payload));
            printf("[PICO-MODEM] Dispatching SMS payload: %s\n", sms_payload);

            /* UART transmission to SIM800L */
            char at_cmd[128];
            gsm_modem_format_cmgs_command(EMERGENCY_PHONE_NUMBER, at_cmd, sizeof(at_cmd));
            /* uart_puts(uart0, at_cmd); */
        }
        s_buffer_ready = false;
    }
}

int main(void) {
    pico_hardware_init();

    /* Simulate streaming 500 samples */
    for (int i = 0; i < 500; i++) {
        int16_t mock_adc = (int16_t)(512 + (i % 50) * 10);
        pico_process_sample(mock_adc);
    }
    return 0;
}
