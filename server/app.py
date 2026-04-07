import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.post("/reset")
async def reset_environment():
    """
    Required endpoint for Hugging Face Space validation.
    The validator sends a POST request here to ensure the container is alive.
    """
    return {"status": "success", "message": "Environment reset"}

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()