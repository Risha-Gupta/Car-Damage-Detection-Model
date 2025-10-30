from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from app.services.prediction import PredictionService

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

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
