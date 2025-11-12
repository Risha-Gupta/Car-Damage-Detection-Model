from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from app.services.prediction import PredictionService, Stage4DamageClassifier, DetectronModelService
from app.services.locate import LocationService
from app.services.segmentation import SegmentationService
import numpy as np
from PIL import Image
import io
import cv2
import os
import tempfile

router = APIRouter()

# Service instances
pred_svc = PredictionService()
loc_svc = LocationService()
seg_svc = SegmentationService()
classifier_svc = Stage4DamageClassifier()
detectron_svc = DetectronModelService()

# Dependency getters
def get_pred_svc():
    return pred_svc

def get_loc_svc():
    return loc_svc

def get_seg_svc():
    return seg_svc

@router.post("/predict-detectron")
async def predict_with_detectron(file: UploadFile = File(...)):
    """
    Detectron2 model inference endpoint.
    
    Accepts image file upload and returns bounding box predictions
    from trained Detectron2 checkpoint.
    
    Returns:
    - boxes: List of [x1, y1, x2, y2] coordinates
    - scores: Confidence scores (filtered by 0.5 threshold)
    - classes: Predicted class IDs
    - num_detections: Total number of detections
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    temp_file_path = None
    try:
        # Save uploaded file temporarily for Detectron2 processing
        img_bytes = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(img_bytes)
            temp_file_path = tmp_file.name
        
        # Run Detectron2 inference
        predictions = detectron_svc.predict_damage_detectron(temp_file_path)
        
        return {
            "success": True,
            "filename": file.filename,
            "model": "detectron2",
            "predictions": predictions,
            "confidence_threshold": 0.5
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image with Detectron2: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@router.post("/predict")
async def predict_damage_pipeline(
    file: UploadFile = File(...)
):
    """
    Full damage detection pipeline: Stage 1 -> Stage 2 -> Stage 3 -> Stage 4
    
    Stage 1 (Prediction): Binary classification - is the image damaged?
    Stage 2 (Localization): Detect bounding boxes of damage regions
    Stage 3 (Segmentation): Create detailed masks of damaged areas
    Stage 4 (Classification): Classify damage type, severity, and repair priority
    
    Input: Single image file
    Output: Complete damage report with all stages' results
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        # Read image bytes once for all stages
        img_bytes = await file.read()
        
        # ========== STAGE 1: Binary Damage Classification ==========
        stage1_result = pred_svc.predict(img_bytes)
        
        # If no damage detected, short-circuit and return early
        if not stage1_result["is_damaged"]:
            return {
                "success": True,
                "filename": file.filename,
                "pipeline_status": "no_damage_detected",
                "stage_1_classification": stage1_result,
                "stages_2_3_4": None
            }
        
        # ========== STAGE 2: Damage Localization ==========
        stage2_result = loc_svc.predict(img_bytes)
        
        # ========== STAGE 3: Segmentation ==========
        stage3_result = seg_svc.predict(img_bytes)
        
        # ========== STAGE 4: Damage Classification & Severity ==========
        # Prepare image array for metrics calculation
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        stage4_result = classifier_svc.classify_damage(
            segmentation_data=stage3_result,
            bbox_data=stage2_result,
            image_array=img_bgr
        )
        
        # ========== Compile Full Pipeline Response ==========
        return {
            "success": True,
            "filename": file.filename,
            "pipeline_status": "completed",
            "stage_1_classification": {
                "is_damaged": stage1_result["is_damaged"],
                "confidence": stage1_result["confidence"],
                "status": stage1_result["status"]
            },
            "stage_2_localization": {
                "detected_damage_bboxes": stage2_result.get("detected_damage_bboxes", []),
                "damage_regions_count": stage2_result.get("damage_regions_count", 0),
                "annotated_image_path": stage2_result.get("annotated_image_path", None)
            },
            "stage_3_segmentation": {
                "detections": stage3_result.get("detections", []),
                "detection_count": stage3_result.get("detection_count", 0),
                "has_masks": stage3_result.get("has_segmentation_masks", False),
                "annotated_image": stage3_result.get("annotated_image", None)
            },
            "stage_4_classification": stage4_result,
            "summary": {
                "overall_damage_status": "damaged" if stage1_result["is_damaged"] else "not_damaged",
                "severity": stage4_result.get("damage_severity"),
                "repair_priority": stage4_result.get("repair_priority"),
                "damage_type": stage4_result.get("damage_type"),
                "coverage_percent": stage4_result.get("damage_coverage_percent")
            }
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image through pipeline: {str(e)}"
        )


# Existing individual stage endpoints (kept for testing/debugging)
@router.post("/predict-stage1")
async def predict_damage(
    file: UploadFile = File(...),
    svc: PredictionService = Depends(get_pred_svc)
):
    """Stage 1 only: Binary damage classification"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        img_bytes = await file.read()
        result = svc.predict(img_bytes)
        
        return {
            "success": True,
            "filename": file.filename,
            "stage": 1,
            "result": result
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

@router.post("/locate")
async def locate_damage(
    file: UploadFile = File(...),
    svc: LocationService = Depends(get_loc_svc)
):
    """Stage 2 only: Damage localization with bounding boxes"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        img_bytes = await file.read()
        result = svc.predict(img_bytes)

        return {
            "success": True,
            "filename": file.filename,
            "stage": 2,
            "result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during localization: {str(e)}")

@router.post("/segment")
async def segment_damage(
    file: UploadFile = File(...),
    svc: SegmentationService = Depends(get_seg_svc)
    ):
    """Stage 3 only: Segmentation and mask detection"""
    try:
        img_bytes = await file.read()
        result = svc.predict(img_bytes)

        return {
            "success": True,
            "filename": file.filename,
            "stage": 3,
            "result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during segmentation: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "pipeline": "stages_1_2_3_4_available", "detectron2": "ready"}
