/**
 * TinyCardio 2G Cellular GSM Modem Driver (AT Commands)
 * File: gsm_modem_driver.c
 */

#include "gsm_modem_driver.h"
#include <stdio.h>
#include <string.h>

static GsmModemState s_state = GSM_STATE_IDLE;
static char s_phone_number[32] = "+10000000000";
static char s_pending_alert[64] = "";

void gsm_modem_init(const char* emergency_phone_number) {
    s_state = GSM_STATE_IDLE;
    if (emergency_phone_number && strlen(emergency_phone_number) < sizeof(s_phone_number)) {
        strncpy(s_phone_number, emergency_phone_number, sizeof(s_phone_number) - 1);
    }
}

GsmModemState gsm_modem_get_state(void) {
    return s_state;
}

int gsm_modem_format_cmgs_command(const char* phone_number, char* out_cmd, size_t max_len) {
    return snprintf(out_cmd, max_len, "AT+CMGS=\"%s\"\r\n", phone_number);
}

int gsm_modem_format_sms_body(const char* alert_text, char* out_body, size_t max_len) {
    /* Append ASCII 0x1A (Ctrl+Z) to trigger transmission on cellular modems */
    return snprintf(out_body, max_len, "%s\x1A", alert_text);
}

bool gsm_modem_process_uart(const char* rx_line, char* tx_cmd, size_t tx_max) {
    if (!tx_cmd || tx_max < 32) return false;
    tx_cmd[0] = '\0';

    switch (s_state) {
        case GSM_STATE_IDLE:
            /* Start initialization handshake */
            snprintf(tx_cmd, tx_max, "AT\r\n");
            s_state = GSM_STATE_CHECK_AT;
            return true;

        case GSM_STATE_CHECK_AT:
            if (rx_line && strstr(rx_line, "OK")) {
                snprintf(tx_cmd, tx_max, "AT+CREG?\r\n");
                s_state = GSM_STATE_CHECK_CREG;
                return true;
            }
            break;

        case GSM_STATE_CHECK_CREG:
            /* +CREG: 0,1 (registered home) or +CREG: 0,5 (registered roaming) */
            if (rx_line && (strstr(rx_line, ",1") || strstr(rx_line, ",5") || strstr(rx_line, "OK"))) {
                snprintf(tx_cmd, tx_max, "AT+CMGF=1\r\n"); /* SMS Text Mode */
                s_state = GSM_STATE_SET_TEXT_MODE;
                return true;
            }
            break;

        case GSM_STATE_SET_TEXT_MODE:
            if (rx_line && strstr(rx_line, "OK")) {
                gsm_modem_format_cmgs_command(s_phone_number, tx_cmd, tx_max);
                s_state = GSM_STATE_WAIT_PROMPT;
                return true;
            }
            break;

        case GSM_STATE_WAIT_PROMPT:
            /* Modem sends '>' prompt when ready for text payload */
            if (rx_line && strchr(rx_line, '>')) {
                gsm_modem_format_sms_body(s_pending_alert, tx_cmd, tx_max);
                s_state = GSM_STATE_WAIT_OK;
                return true;
            }
            break;

        case GSM_STATE_WAIT_OK:
            if (rx_line && strstr(rx_line, "OK")) {
                s_state = GSM_STATE_SUCCESS;
                return false;
            }
            if (rx_line && strstr(rx_line, "ERROR")) {
                s_state = GSM_STATE_ERROR;
                return false;
            }
            break;

        case GSM_STATE_SUCCESS:
        case GSM_STATE_ERROR:
        default:
            break;
    }
    return false;
}
