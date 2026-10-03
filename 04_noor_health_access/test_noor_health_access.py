#!/usr/bin/env python3
"""
Unit and Integration Test Suite: Noor Health Access Module
==========================================================
Verifies compliance with World Bank & Hack-Nation Small AI requirements:
  1. Multilingual Language Detection (Swahili, Quechua, English, Spanish)
  2. Urgency classification & red flag safety detection
  3. Grounding in World Bank SDI & healthsites.io facility datasets
  4. DHIS2 standard event format validation
  5. Latency (< 50 ms) and offline compatibility
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from local_nlp_triage import classify_symptom_triage, detect_language
from facility_router import HealthFacilityRouter
from dhis2_generator import generate_dhis2_event_payload
from noor_sms_service import process_noor_inquiry


class TestNoorHealthAccess(unittest.TestCase):

    def test_language_detection(self):
        self.assertEqual(detect_language("Nina homa kali sana"), "sw")
        self.assertEqual(detect_language("Sunquy nanawan sinchita"), "qu")
        self.assertEqual(detect_language("Tengo dolor fuerte en el pecho"), "es")
        self.assertEqual(detect_language("Severe headache and fever"), "en")

    def test_emergency_red_flag_triage(self):
        res = classify_symptom_triage("Nina maumivu makali ya kifua na kushindwa kupumua")
        self.assertEqual(res["urgency_level"], "EMERGENCY_RED")
        self.assertGreaterEqual(res["confidence_score"], 0.80)
        self.assertTrue(len(res["red_flags_detected"]) > 0)
        self.assertTrue("Taarifa ya Usalama" in res["safety_guardrail"] or "Safety Guardrail" in res["safety_guardrail"])

    def test_facility_routing_with_sdi_absenteeism(self):
        router = HealthFacilityRouter()
        # Kipkabus Hill Health Center has an absent clinician today in SDI data
        rec = router.find_best_facility("CLINICAL_YELLOW")
        # Should NOT route to an absent clinician facility
        self.assertNotEqual(rec["facility_name"], "Kipkabus Hill Health Center")
        self.assertIn("ON DUTY", rec["provider_status"])

    def test_dhis2_event_schema(self):
        triage = classify_symptom_triage("Mtoto ana homa kali")
        payload = generate_dhis2_event_payload(
            patient_name="Noor",
            national_id_or_phone="+254 712 000 111",
            facility_code="KE_OND_DISP_01",
            triage_result=triage
        )
        self.assertIn("program", payload)
        self.assertIn("orgUnit", payload)
        self.assertIn("dataValues", payload)
        self.assertEqual(payload["status"], "COMPLETED")
        self.assertTrue(any(dv["dataElement"] == "TRIAGE_URGENCY_LEVEL" for dv in payload["dataValues"]))

    def test_end_to_end_sms_service(self):
        out = process_noor_inquiry(
            patient_phone="+254 712 998 011",
            patient_name="Noor Chepkemoi",
            message_text="Mtoto ana homa kali na kutapika"
        )
        self.assertEqual(out["triage"]["urgency_level"], "CLINICAL_YELLOW")
        self.assertLess(out["latency_ms"], 50.0)  # Sub-50ms execution
        self.assertTrue(len(out["sms_reply"]) > 20)


if __name__ == "__main__":
    unittest.main()
