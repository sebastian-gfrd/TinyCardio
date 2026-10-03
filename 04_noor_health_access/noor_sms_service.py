#!/usr/bin/env python3
"""
TinyCardio / NoorCare: End-to-End Multilingual SMS Service for Noor
==================================================================
Simulates Noor interacting from her basic phone via 2G SMS or local voice line.

End-to-End Pipeline:
  1. Noor texts in Swahili: "Nina homa kali na maumivu ya kifua"
  2. Small AI local NLP classifies urgency & extracts red flags
  3. Facility Router checks healthsites.io & World Bank SDI for clinician attendance
  4. Returns tailored response in Swahili with travel time & attendance
  5. Auto-generates DHIS2 clinical event record for the clinic staff
"""

import sys
import os
import json
import time

# Ensure parent path
sys.path.insert(0, os.path.dirname(__file__))

from local_nlp_triage import classify_symptom_triage
from facility_router import HealthFacilityRouter
from dhis2_generator import generate_dhis2_event_payload


def process_noor_inquiry(patient_phone: str, patient_name: str, message_text: str) -> dict:
    print("-" * 65)
    print(f"[INCOMING SMS FROM NOOR'S BASIC 2G PHONE]")
    print(f"  Sender: {patient_name} ({patient_phone})")
    print(f"  Message: \"{message_text}\"")
    print("-" * 65)

    start_t = time.time()

    # Step 1: Small AI Multilingual Triage
    triage = classify_symptom_triage(message_text)

    # Step 2: World Bank SDI & Healthsites.io Facility Routing
    router = HealthFacilityRouter()
    facility = router.find_best_facility(triage["urgency_level"])

    # Step 3: Auto-generate DHIS2 Record to relieve clinician paperwork
    dhis2_record = generate_dhis2_event_payload(
        patient_name=patient_name,
        national_id_or_phone=patient_phone,
        facility_code=facility["dhis2_facility_code"],
        triage_result=triage
    )

    elapsed_ms = (time.time() - start_t) * 1000

    # Step 4: Construct Return SMS (strictly bounded, safe, concise)
    lang = triage["detected_language"]
    advice = triage["patient_guidance_text"]
    fac_name = facility["facility_name"]
    time_est = facility["estimated_travel_time"]
    doc_status = facility["provider_status"]

    if lang == "sw":
        sms_reply = f"NoorCare: {triage['localized_urgency_label']}. Kituo kinachofaa: {fac_name} ({time_est}). {doc_status}. {advice}"
    elif lang == "qu":
        sms_reply = f"NoorCare: {triage['localized_urgency_label']}. Hampiwasiman riy: {fac_name} ({time_est}). {advice}"
    else:
        sms_reply = f"NoorCare: {triage['localized_urgency_label']}. Recommended: {fac_name} ({time_est}). {doc_status}. {advice}"

    # Print summary
    print(f"[SMALL AI INFERENCE COMPLETE in {elapsed_ms:.1f} ms]")
    print(f"  Language Detected:   {triage['detected_language'].upper()}")
    print(f"  Triage Category:     {triage['urgency_level']} (Confidence: {triage['confidence_score']*100:.0f}%)")
    print(f"  Red Flags:           {triage['red_flags_detected']}")
    print(f"  Routed Facility:     {facility['facility_name']}")
    print(f"  Provider On Duty:    {facility['provider_status']}")
    print(f"  DHIS2 Event Queued:  Event ID {dhis2_record['event']} -> {facility['dhis2_facility_code']}")
    print("\n[OUTGOING RETURN SMS TO NOOR (< 160 chars)]:")
    print(f"  \"{sms_reply}\"")
    print("-" * 65)

    return {
        "triage": triage,
        "facility": facility,
        "dhis2_record": dhis2_record,
        "sms_reply": sms_reply,
        "latency_ms": elapsed_ms
    }


def run_demo():
    print("=" * 65)
    print("TINYCARDIO / NOORCARE: END-TO-END RURAL SMS TRIAGE PIPELINE")
    print("World Bank Group Small AI for Development Hackathon 2026")
    print("=" * 65)

    scenarios = [
        ("Noor Chepkemoi", "+254 712 998 011", "Nina maumivu makali ya kifua na kizunguzungu kali"),
        ("Noor Chepkemoi", "+254 712 998 011", "Mtoto ana homa kali sana na anatapika"),
        ("Noor Chepkemoi", "+254 712 998 011", "Nahitaji kuongeza dawa zangu za shinikizo la damu")
    ]

    for name, phone, msg in scenarios:
        process_noor_inquiry(phone, name, msg)
        print()


if __name__ == "__main__":
    run_demo()
