from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from app.services.prediction import PredictionService, DetectronModelService
from app.services.stage4_classifier import Stage4DamageClassifier
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

@router.post("/classify-damage")
async def classify_damage(
    segmentation_data: dict,
    bbox_data: dict,
    image_file: UploadFile = File(...)
):
    """
    Stage 4 only: Classify damage severity, type, and repair priority.
    
    Expects:
    - segmentation_data: Output from /segment endpoint (Stage 3)
    - bbox_data: Output from /locate endpoint (Stage 2)
    - image_file: Original image file for coverage metrics
    
    Returns:
    - damage_severity: "minor", "moderate", or "severe"
    - damage_type: "scratch", "crack", "dent", or "no_damage"
    - damage_coverage_percent: Percentage of image covered by damage
    - repair_priority: "high", "medium", or "low"
    - detection_count: Number of damage detections
    """
    if not image_file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        img_bytes = await image_file.read()
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        result = classifier_svc.classify_damage(
            segmentation_data=segmentation_data,
            bbox_data=bbox_data,
            image_array=img_bgr
        )
        
        return {
            "success": True,
            "filename": image_file.filename,
            "stage": 4,
            "result": result
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during Stage 4 classification: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "pipeline": "stages_1_2_3_4_available", "detectron2": "ready"}
