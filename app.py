import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pipeline import AnimalDetectionPipeline
from utils import extract_gps_info, get_species_info

app = FastAPI(title="Animal Detection Demo")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
pipeline = AnimalDetectionPipeline()

# Mount static files for frontend
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # 1. Extract GPS (if any)
        gps_data = extract_gps_info(image_bytes)
        
        # 2. Run Pipeline (MegaDetector -> ConvNeXt)
        results = pipeline.predict(image_bytes)
        
        # 3. Get Species Info for Top-1 (if detection successful)
        species_info = None
        if results.get("top_predictions"):
            top1_label = results["top_predictions"][0]["label"]
            species_info = get_species_info(top1_label)
            
        return JSONResponse(content={
            "success": True,
            "boxes": results.get("boxes", []),
            "top_predictions": results.get("top_predictions", []),
            "gps": gps_data,
            "species_info": species_info
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
