from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from app.services.prediction import PredictionService
from app.services.locate import LocationService
router = APIRouter()

prediction_service = PredictionService()

def get_prediction_service():
    return prediction_service

@router.post("/predict")
async def predict_damage(
    file: UploadFile = File(...),
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Predict if an image shows damage
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Make prediction
        result = service.predict(image_bytes)
        
        return {
            "success": True,
            "filename": file.filename,
            "result": result
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

damage_service = LocationService()

def get_damage_service():
    return damage_service


@router.post("/damage/locate")
async def locate_damage(
    file: UploadFile = File(...),
    service: LocationService = Depends(get_damage_service)
):
    """
    Stage 2: Detect damage regions (bbox + mask + roi + polygons)
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        result = service.predict(image_bytes)

        return {
            "success": True,
            "filename": file.filename,
            "stage": 2,
            "result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during segmentation: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
