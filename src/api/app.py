# api/main.py
import torch
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import sys

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logger import get_logger
from src.utils import load_config
from src.inference import load_model_for_inference, preprocess_image, predict
from src.exceptions import InferenceError

logger = get_logger(__name__)

# Constants
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
MODEL_PATH = Path(__file__).resolve().parent.parent / "artefacts" / "models" / "best_model.pth"

# Load configuration
config = load_config(str(CONFIG_PATH))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# We need label mapping; since training created it, we'll mimic a lightweight loader.
# For production you'd save mapping.json, but for now let's load from data.
from src.get_and_process_data.dataset import build_data_loader
_, _, num_classes, idx_to_label = build_data_loader(config)
config['model']['num_classes'] = num_classes

# Load model at startup
if not MODEL_PATH.exists():
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Run training first.")
model = load_model_for_inference(config, str(MODEL_PATH), device)
logger.info("Model loaded successfully. API ready.")

app = FastAPI(title="Art Style ViT", description="Vision Transformer for art style classification", version="1.0")

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Accept an image file and return the predicted art style with confidence scores.
    """
    # Validate image format
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    try:
        # Save temporary file
        img_bytes = await file.read()
        tmp_path = Path(f"/tmp/{file.filename}")
        tmp_path.write_bytes(img_bytes)

        # Preprocess and predict
        input_tensor = preprocess_image(str(tmp_path), config['data']['image_size'])
        result = predict(model, input_tensor, device, idx_to_label, config['evaluation']['top_k'])

        # Clean up temp file
        tmp_path.unlink(missing_ok=True)

        return JSONResponse(content={
            "predicted_style": result['predicted_label'],
            "confidence": round(result['confidence'], 4),
            "top_predictions": [
                {"style": label, "probability": round(prob, 4)} for label, prob in result['top_k']
            ]
        })
    except InferenceError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Unhandled error during inference")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok"}