from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn
import os
import tempfile
from typing import Optional
from contextlib import asynccontextmanager
from zina import DEFAULT_MODEL_ID, ZinaModel

zina_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global zina_model
    model_path = os.environ.get("MODEL_PATH", DEFAULT_MODEL_ID)
    zina_model = ZinaModel(model_path=model_path)
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    cand: str = Form(...),
    refs: str = Form(...),
    verbose: Optional[bool] = Form(False)
):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[1]) as tmp_file:
            content = await image.read()
            tmp_file.write(content)
            image_path = tmp_file.name
        
        pred_with_tags, pred = zina_model.run(
            cand=cand,
            refs=refs,
            image_path=image_path,
            verbose=verbose
        )
        
        os.unlink(image_path)
        
        return JSONResponse({
            "status": "success",
            "pred_with_tags": pred_with_tags,
            "pred": pred
        })
        
    except Exception as e:
        if 'image_path' in locals() and os.path.exists(image_path):
            os.unlink(image_path)
        
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=3000)
