from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from app.services.prediction import PredictionService
from app.services.locate import LocationService

router = APIRouter()

pred_svc = PredictionService()

def get_pred_svc():
    return pred_svc

@router.post("/predict")
async def predict_damage(
    file: UploadFile = File(...),
    svc: PredictionService = Depends(get_pred_svc)
):
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
            "result": result
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

loc_svc = LocationService()

def get_loc_svc():
    return loc_svc

@router.post("locate")
async def locate_damage(
    file: UploadFile = File(...),
    svc: LocationService = Depends(get_loc_svc)
):
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

@router.get("/health")
async def health_check():
    return {"status": "healthy"}
