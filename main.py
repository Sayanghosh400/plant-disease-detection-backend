from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
import function

app = FastAPI()

# Allow all origins for development (update as needed in production)
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/process_input")
async def process_input(file: UploadFile = File(...)):
    contents = await file.read()
    image_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(image_path, "wb") as f:
        f.write(contents)
    a = function.evaluate(image_path)
    if a:
        return {"result": f"Detected disease: {a}"}
    else:
        return {"result": "Failed to evaluate the image."}
