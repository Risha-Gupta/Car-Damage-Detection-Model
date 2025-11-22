from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from app.services.serverity import Stage4DamageClassifier
from app.services.prediction import PredictionService
from app.services.locate import LocationService
from app.services.segmentation import SegmentationService
from app.services.estimation import Stage5CostEstimator
from app.services.report_generator import InsuranceReportGenerator
import numpy as np
from fastapi.responses import FileResponse
from PIL import Image
import io
import cv2
import os
import tempfile
from app.services.detectron_model import DetectronModelService
router = APIRouter()

pred_svc = PredictionService()
loc_svc = LocationService()
seg_svc = SegmentationService()
classifier_svc = Stage4DamageClassifier()
detectron_svc = DetectronModelService()
estimation_svc = Stage5CostEstimator()
report_gen = InsuranceReportGenerator()

def get_pred_svc():
    return pred_svc

def get_loc_svc():
    return loc_svc

def get_seg_svc():
    return seg_svc

def get_classifier_svc():
    return classifier_svc

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

@router.post("/classify")
async def classify_damage(
    file: UploadFile = File(...),
    svc: Stage4DamageClassifier = Depends(get_classifier_svc)
):
    """
    Stage 4 only: Classify damage severity and type.
    
    This endpoint runs Stage 3 (segmentation) internally to get the required data,
    then classifies damage severity, type, coverage, and repair priority.
    
    Returns:
    - damage_severity: minor, moderate, or severe
    - damage_type: scratch, crack, dent, or no_damage
    - damage_coverage_percent: percentage of image covered by damage
    - repair_priority: low, medium, or high
    - detection_count: number of damage regions detected
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        img_bytes = await file.read()
        
        # Run Stage 3 (segmentation) to get detection data
        print("Running Stage 3 segmentation...")
        stage3_result = seg_svc.predict(img_bytes)
        print(f"Stage 3 result: {stage3_result}")
        
        # Prepare image array for Stage 4 classification
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Create bbox_data from segmentation results
        # Handle different bbox formats
        bboxes = []
        detections = stage3_result.get("detections", [])
        
        if not detections:
            print("No detections found in Stage 3")
        
        for detection in detections:
            if "bbox" in detection:
                bbox = detection["bbox"]
                
                # Handle dict format: {"x_min": ..., "y_min": ..., "width": ..., "height": ...}
                if isinstance(bbox, dict):
                    x_min = bbox.get("x_min", 0)
                    y_min = bbox.get("y_min", 0)
                    width = bbox.get("width", 0)
                    height = bbox.get("height", 0)
                    area = width * height
                    bbox_list = [x_min, y_min, x_min + width, y_min + height]
                
                # Handle list format: [x1, y1, x2, y2]
                elif isinstance(bbox, list) and len(bbox) >= 4:
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    area = width * height
                    bbox_list = bbox
                else:
                    print(f"Unknown bbox format: {bbox}")
                    continue
                
                bboxes.append({
                    "bbox": bbox_list,
                    "bbox_area": area
                })
        
        bbox_data = {"detected_damage_bboxes": bboxes}
        print(f"Created bbox_data with {len(bboxes)} bboxes")
        
        # Run Stage 4 classification
        result = svc.classify_damage(
            segmentation_data=stage3_result,
            bbox_data=bbox_data,
            image_array=img_bgr
        )
        
        return {
            "success": True,
            "filename": file.filename,
            "stage": 4,
            "result": result,
            "segmentation_summary": {
                "detection_count": stage3_result.get("detection_count", 0),
                "has_masks": stage3_result.get("has_segmentation_masks", False)
            }
        }
    
    except Exception as e:
        import traceback
        error_detail = f"Error during damage classification: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(
            status_code=500,
            detail=error_detail
        )
@router.post("/estimate")
async def estimate_damage_cost(
    file: UploadFile = File(...),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        img_bytes = await file.read()

        # -----------------------------
        # Stage 3: Segmentation
        # -----------------------------
        stage3_result = seg_svc.predict(img_bytes)

        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # -----------------------------
        # Convert Stage 3 → bbox_data
        # -----------------------------
        detections = stage3_result.get("detections", [])
        bboxes = []

        for det in detections:
            bbox = det.get("bbox")

            if isinstance(bbox, dict):
                x = bbox.get("x_min", 0)
                y = bbox.get("y_min", 0)
                w = bbox.get("width", 0)
                h = bbox.get("height", 0)
                area = w * h
                bbox_list = [x, y, x + w, y + h]

            elif isinstance(bbox, list) and len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                area = (x2 - x1) * (y2 - y1)
                bbox_list = [x1, y1, x2, y2]

            else:
                continue

            bboxes.append({"bbox": bbox_list, "bbox_area": area})

        bbox_data = {"detected_damage_bboxes": bboxes}

        # -----------------------------
        # Stage 4: Severity + Type
        # -----------------------------
        stage4_output = classifier_svc.classify_damage(
            segmentation_data=stage3_result,
            bbox_data=bbox_data,
            image_array=img_bgr
        )

        # -----------------------------
        # Stage 5: Cost Estimation
        # -----------------------------
        result = estimation_svc.estimate_cost(stage4_output)
        pdf_path = report_gen.generate(stage4_output, result, meta={})
        
        filename = os.path.basename(pdf_path)
        download_url = f"/api/download-report/{filename}"
        # -----------------------------
        # Final Response
        # -----------------------------
        return {
            "success": True,
            "filename": file.filename,
            "stage":5,
            "result": result,
            "report": {
                "download_url": download_url
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stage 5 error: {str(e)}")

@router.get("/download-report/{filename}")
async def download_report(filename: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(base_dir)
    report_dir = os.path.join(base_dir, "../../../outputs/reports")
    file_path = os.path.join(report_dir, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Report not found")

    return FileResponse(
        file_path,
        media_type="application/pdf",
        filename=filename
    )

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "pipeline": "stages_1_2_3_4_available", "detectron2": "ready"}
