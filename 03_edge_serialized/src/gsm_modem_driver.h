/**
 * TinyCardio 2G Cellular GSM Modem Driver (AT Commands)
 * File: gsm_modem_driver.h
 * Finite State Machine (FSM) for SIM800L / Quectel M95 / A6 modules.
 * Transmits emergency cardiac arrest alerts (<60 bytes) via UART.
 */

#ifndef GSM_MODEM_DRIVER_H
#define GSM_MODEM_DRIVER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GSM_STATE_IDLE = 0,
    GSM_STATE_CHECK_AT,
    GSM_STATE_CHECK_CREG,
    GSM_STATE_SET_TEXT_MODE,
    GSM_STATE_SEND_CMGS,
    GSM_STATE_WAIT_PROMPT,
    GSM_STATE_SEND_PAYLOAD,
    GSM_STATE_WAIT_OK,
    GSM_STATE_SUCCESS,
    GSM_STATE_ERROR
} GsmModemState;

/**
 * Initializes the modem driver with an emergency phone number.
 */
void gsm_modem_init(const char* emergency_phone_number);

/**
 * Returns current state of the modem finite state machine.
 */
GsmModemState gsm_modem_get_state(void);

/**
 * Formats standard AT+CMGS command for single-message SMS delivery.
 */
int gsm_modem_format_cmgs_command(const char* phone_number, char* out_cmd, size_t max_len);

/**
 * Formats full SMS payload with Ctrl+Z terminator (0x1A).
 */
int gsm_modem_format_sms_body(const char* alert_text, char* out_body, size_t max_len);

/**
 * Processes incoming UART lines from modem and generates the next AT command.
 *
 * @param rx_line  Incoming null-terminated line from modem UART (or NULL on timeout).
 * @param tx_cmd   Buffer where driver writes next command to send to modem.
 * @param tx_max   Maximum capacity of tx_cmd buffer.
 * @return true if there is a command to transmit.
 */
bool gsm_modem_process_uart(const char* rx_line, char* tx_cmd, size_t tx_max);

#ifdef __cplusplus
}
#endif

#endif /* GSM_MODEM_DRIVER_H */
