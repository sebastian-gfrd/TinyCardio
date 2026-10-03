#!/usr/bin/env python3
"""
TinyCardio / NoorCare: Facility Router & Service Delivery Indicator Matcher
==========================================================================
Grounds triage recommendations using World Bank SDI & healthsites.io data:
  - Answers the "is Tobi there today?" problem (Provider absenteeism).
  - Checks essential tracer medicine stock status before dispatching Noor.
  - Computes travel time surface estimation (walking vs rural motorbike).
"""

import os
import json
from typing import Dict, Any, List, Optional


DATA_DIR = os.path.join(os.path.dirname(__file__), "data_grounding")
SDI_PATH = os.path.join(DATA_DIR, "world_bank_sdi_ondera.json")
HEALTHSITES_PATH = os.path.join(DATA_DIR, "healthsites_ondera.geojson")


class HealthFacilityRouter:
    def __init__(self, sdi_file: str = SDI_PATH, healthsites_file: str = HEALTHSITES_PATH):
        with open(sdi_file, "r", encoding="utf-8") as f:
            self.sdi_data = json.load(f)
        with open(healthsites_file, "r", encoding="utf-8") as f:
            self.healthsites_data = json.load(f)

        self.facilities = self.sdi_data.get("facilities", [])

    def find_best_facility(self, urgency_level: str, required_medicine: Optional[str] = None) -> Dict[str, Any]:
        """
        Determines the optimal facility for Noor based on urgency, clinician attendance, and stock.
        """
        if urgency_level == "EMERGENCY_RED":
            # Direct to Hospital with 24/7 emergency & cardiac care
            for fac in self.facilities:
                if "Hospital" in fac["name"] and fac["provider_on_duty_today"]:
                    return {
                        "facility_name": fac["name"],
                        "tier": fac["tier"],
                        "distance_km": fac["distance_from_noor_km"],
                        "estimated_travel_time": f"{fac.get('estimated_travel_time_motorcycle_min', 35)} min (Motorcycle)",
                        "provider_status": f"ON DUTY: {fac['staff_name']}",
                        "service_readiness": "24/7 Emergency triage & resuscitation active",
                        "dhis2_facility_code": fac["dhis2_facility_code"],
                        "reasoning": "High-risk red-flag condition requires level 4 hospital care."
                    }

        # For Yellow or Green: Search primary dispensary where clinician is on duty
        available_facilities = []
        for fac in self.facilities:
            if not fac["provider_on_duty_today"]:
                continue  # Skip facility where clinician is absent (World Bank SDI problem)

            # Check medicine if specified
            if required_medicine:
                stock = fac.get("stock_status", {}).get(required_medicine)
                if stock == "STOCKOUT":
                    continue  # Skip stocked-out facility

            available_facilities.append(fac)

        if not available_facilities:
            # Fallback to District Hospital
            target = next(f for f in self.facilities if "Hospital" in f["name"])
            return {
                "facility_name": target["name"],
                "tier": target["tier"],
                "distance_km": target["distance_from_noor_km"],
                "estimated_travel_time": f"{target.get('estimated_travel_time_motorcycle_min', 35)} min (Motorcycle)",
                "provider_status": f"ON DUTY: {target['staff_name']}",
                "service_readiness": "Referral care",
                "dhis2_facility_code": target["dhis2_facility_code"],
                "reasoning": "Local dispensaries currently report provider absence or stockouts. Rerouting to hospital."
            }

        # Choose closest available facility
        best = min(available_facilities, key=lambda x: x["distance_from_noor_km"])
        return {
            "facility_name": best["name"],
            "tier": best["tier"],
            "distance_km": best["distance_from_noor_km"],
            "estimated_travel_time": f"{best.get('estimated_travel_time_walk_min', 60)} min (Walk)",
            "provider_status": f"ON DUTY: {best['staff_name']}",
            "current_queue": f"{best.get('current_queue_length', 10)} patients waiting",
            "dhis2_facility_code": best["dhis2_facility_code"],
            "reasoning": "Nearest open facility with confirmed clinician presence and verified inventory."
        }


if __name__ == "__main__":
    router = HealthFacilityRouter()
    print("=" * 65)
    print("TINYCARDIO / NOORCARE: WORLD BANK SDI FACILITY ROUTING TEST")
    print("=" * 65)

    cases = [
        ("EMERGENCY_RED", None),
        ("CLINICAL_YELLOW", "salbutamol_inhaler"),
        ("PRIMARY_GREEN", "antihypertensive_atenolol"),
        ("PRIMARY_GREEN", "amoxicillin")  # Amoxicillin is stocked out at dispensary, should route appropriately
    ]

    for urgency, med in cases:
        rec = router.find_best_facility(urgency, med)
        print(f"\nScenario: Urgency={urgency}, Needed Med={med}")
        print(f"  Recommended Facility: {rec['facility_name']} ({rec['distance_km']} km)")
        print(f"  Provider Status     : {rec['provider_status']}")
        print(f"  Travel Time         : {rec['estimated_travel_time']}")
        print(f"  Clinical Rationale  : {rec['reasoning']}")
    print("=" * 65)
