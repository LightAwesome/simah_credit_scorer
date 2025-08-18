from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from calculations.routes import router as calculations_router
from api.routes import router as api_router
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include calculation routes
app.include_router(calculations_router, prefix="/api/calculations", tags=["calculations"])

# Include API routes
app.include_router(api_router, prefix="/api", tags=["api"])

@app.get("/")
def read_root():
    return {"message": "Simah Credit Scorer API"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_includes=["*.py", "*.json"]
    )