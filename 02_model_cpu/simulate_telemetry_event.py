#!/usr/bin/env python3
"""
TinyCardio 2G SMS Telemetry Simulator (Pilar 2)
==============================================
Simulates the transmission of an emergency 2G SMS payload from an edge node
($4.30 USD) into the Rural Triage & SMS Telemetry Hub.

Usage:
  python3 02_model_cpu/simulate_telemetry_event.py --device 3072 --event ARR --risk 98 --hr 195
  python3 02_model_cpu/simulate_telemetry_event.py --interactive
"""

import sys
import os
import time
import argparse
import urllib.request
import json


def send_sms_event(hub_url: str, raw_sms: str):
    print("=" * 65)
    print("SIMULATING 2G SMS TRANSMISSION FROM TINYCARDIO EDGE NODE")
    print("=" * 65)
    print(f"Edge Node Radio: SIM800L 2G GSM Quad-Band (900/1800 MHz)")
    print(f"Raw Payload:     '{raw_sms}' ({len(raw_sms)} characters / bytes)")
    print(f"Target Hub:      {hub_url}")
    print("-" * 65)

    data = json.dumps({"raw_sms": raw_sms}).encode('utf-8')
    req = urllib.request.Request(
        hub_url,
        data=data,
        headers={"Content-Type": "application/json"}
    )

    try:
        start_time = time.time()
        with urllib.request.urlopen(req, timeout=5) as resp:
            elapsed_ms = (time.time() - start_time) * 1000
            res_code = resp.getcode()
            body = resp.read().decode('utf-8')
            res_json = json.loads(body)

            print(f"[OK] Telemetry delivered in {elapsed_ms:.1f} ms (HTTP {res_code})")
            alert = res_json.get("alert", {})
            patient = alert.get("patient", {})
            print(f"\n[HUB DISPATCH STATE UPDATED]:")
            print(f"  Alert ID:     {alert.get('alert_id')}")
            print(f"  Patient:      {patient.get('name')} ({patient.get('age')}a, {patient.get('gender')})")
            print(f"  Location:     {patient.get('location_name')}")
            print(f"  Coordinates:  Lat {patient.get('coordinates', {}).get('lat')}, Lng {patient.get('coordinates', {}).get('lng')}")
            print(f"  Severity:     {alert.get('severity')}")
            print(f"  Protocol:     {alert.get('telemetry', {}).get('clinical_protocol')}")
            print("=" * 65)
            return True

    except Exception as e:
        print(f"[ERROR] Failed to send alert to {hub_url}: {e}")
        print("Ensure the hub is running: python3 02_model_cpu/sms_telemetry_webhook.py --port 8090")
        return False


def main():
    parser = argparse.ArgumentParser(description="TinyCardio Edge SMS Telemetry Event Simulator")
    parser.add_argument("--hub", type=str, default="http://localhost:8090/api/sms/incoming", help="Hub incoming SMS URL")
    parser.add_argument("--device", type=int, default=1024, help="Device ID (1024, 2048, 3072, 4096)")
    parser.add_argument("--event", type=str, default="VF", choices=["VF", "ISCH", "ARR"], help="Event code: VF (Fibrillation), ISCH (Ischemia), ARR (Pre-Arrest)")
    parser.add_argument("--risk", type=int, default=96, help="Risk score percentage (0-100)")
    parser.add_argument("--hr", type=int, default=182, help="Heart rate in BPM")
    parser.add_argument("--raw", type=str, default=None, help="Custom raw SMS string")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive menu")

    args = parser.parse_args()

    if args.interactive:
        print("\n--- TINYCARDIO INTERACTIVE EMERGENCY DISPATCH SIMULATOR ---")
        print("Select clinical scenario:")
        print("  1) Ventricular Fibrillation (VFDB 418) - María Quispe (ID 1024)")
        print("  2) Acute Myocardial Ischemia (EDB 105) - Carlos Mamani (ID 2048)")
        print("  3) Autonomic Cardiac Arrest Precursor (SDDB 30) - Esperanza Flores (ID 3072)")
        print("  4) Severe Tachycardia & Ectopy - José David Morales (ID 4096)")
        choice = input("Enter option [1-4, default 1]: ").strip() or "1"

        if choice == "1":
            dev, ev, risk, hr = 1024, "VF", 97, 185
        elif choice == "2":
            dev, ev, risk, hr = 2048, "ISCH", 82, 118
        elif choice == "3":
            dev, ev, risk, hr = 3072, "ARR", 99, 192
        else:
            dev, ev, risk, hr = 4096, "VF", 91, 168

        ts = int(time.time())
        raw_sms = f"TC:D={dev};E={ev};R={risk}%;HR={hr};TS={ts}s"
    elif args.raw:
        raw_sms = args.raw
    else:
        ts = int(time.time())
        raw_sms = f"TC:D={args.device};E={args.event};R={args.risk}%;HR={args.hr};TS={ts}s"

    send_sms_event(args.hub, raw_sms)


if __name__ == "__main__":
    main()
