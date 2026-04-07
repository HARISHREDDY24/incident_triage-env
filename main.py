from fastapi import FastAPI
from src.environment import AgriMarketEnv

app = FastAPI()
env = AgriMarketEnv()


@app.post("/reset")
async def reset():
    obs = await env.reset()
    return obs.model_dump()


@app.get("/state")
async def state():
    return await env.state()