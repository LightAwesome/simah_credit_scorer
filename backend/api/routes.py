from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from extraction.extractor import extract
import tempfile
import os
import shutil




app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["https://your-frontend.vercel.app"]
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}



@app.post("/upload/")
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    temp_dir = tempfile.gettempdir()
    safe_name = os.path.basename(file.filename).replace(" ", "_")
    temp_path = os.path.join(temp_dir, f"temp_{safe_name}")

    print(f"Saving to {temp_path}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = extract(temp_path)
    finally:
        os.remove(temp_path)

    return JSONResponse(content=result)
# def read_item(file: int, q: Union[str, None] = None):
#     return {"File": file, "q": q}
