import os
import joblib
import numpy as np
import pandas as pd

VALID_PARTS = {
    "back_bumper", "back_door", "back_glass", "back_light",
    "front_bumper", "front_door", "front_glass", "front_light",
    "hood"
}

class Stage5CostEstimator:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        self.model_path_xgb = os.path.join(
            base_dir, "../../../Middleware/models/stage5/models/stage5_cost_model_xgb.pkl"
        )
    
        if not os.path.exists(self.model_path_xgb):
            raise FileNotFoundError(f"Missing XGBoost model at {self.model_path_xgb}")
        
        self.model_xgb = joblib.load(self.model_path_xgb)

    def estimate_cost(self, stage4_output: dict):
        roi_images = stage4_output.get("roi_images", [])
        severity = stage4_output.get("damage_severity") or stage4_output.get("severity")
        damage_type = stage4_output.get("damage_type")
        coverage = float(stage4_output.get("damage_coverage_percent") or stage4_output.get("coverage_percent", 0))

        valid_parts = []
        ignored = 0
        for d in roi_images:
            part_detected = d.get("part_detected")
            if part_detected is None:
                ignored += 1
                continue
            
            part_name = part_detected.get("class_name", "").lower().strip()
            if part_name in VALID_PARTS:
                valid_parts.append(part_name)
            else:
                ignored += 1

        if len(valid_parts) == 0:
            return {
                "valid_regions": 0,
                "ignored_regions": ignored,
                "total_estimated_cost": 0,
                "details": [],
                "note": "All detected parts were invalid or unknown."
            }

        region_outputs = []

        for part in valid_parts:
            row = pd.DataFrame([{
                "part": part,
                "severity": severity,
                "damage_type": damage_type,
                "coverage_percent": coverage
            }])

            pred_xgb = float(self.model_xgb.predict(row)[0])

            region_outputs.append({
                "part": part,
                "severity": severity,
                "damage_type": damage_type,
                "coverage_percent": coverage,
                "final_cost": pred_xgb
            })

        total_cost = round(sum(r["final_cost"] for r in region_outputs), 2)

        return {
            "valid_regions": len(region_outputs),
            "ignored_regions": ignored,
            "total_estimated_cost": total_cost,
            "details": region_outputs
        }
