from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI(
    title="Car Damage Detection",
    description="API for detecting damage in images",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api", tags=["predictions"])

@app.get("/")
async def root():
    return {
        "message": "Image Damage Detection API",
        "docs": "/docs",
        "health": "/api/v1/health"
    }