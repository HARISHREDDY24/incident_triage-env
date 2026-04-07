import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI(
    title="Incident Triage Environment",
    version="1.0.0",
    description="SRE Simulator",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

class HealthResponse(BaseModel):
    status: str = "healthy"

@app.get("/health")
async def health():
    return HealthResponse()

@app.get("/metadata")
async def metadata():
    return {"name": "incident_triage", "description": "SRE cascading failure simulation"}

@app.get("/schema")
async def get_schema():
    return {
        "action": {"type": "object", "properties": {"command": {"type": "string"}, "args": {"type": "string"}}},
        "observation": {"type": "object", "properties": {"disk_usage_percent": {"type": "number"}}},
        "state": {"type": "object", "properties": {"step_count": {"type": "integer"}}}
    }

@app.post("/reset")
async def reset():
    return {"status": "success", "message": "Environment reset"}

@app.post("/step")
async def step(action: Dict[str, Any]):
    return {"observation": {"disk_usage_percent": 45.0}, "reward": 0.0, "done": False, "info": {}}

@app.get("/state")
async def get_state():
    return {"step_count": 0, "status": "active"}

@app.post("/mcp")
async def mcp_endpoint(payload: Dict[str, Any]):
    return {"jsonrpc": "2.0", "result": "ready", "id": payload.get("id", 1)}

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()