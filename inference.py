import os
import sys
import asyncio
from typing import List, Optional
from openai import OpenAI  

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.environment import IncidentEnv
from src.models import Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")  # No default allowed per checklist!

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
TASK_NAME = os.getenv("TASK_ID", "cascading_failure_hard")

MAX_STEPS = 10
SUCCESS_THRESHOLD = 0.7

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    done_val = str(done).lower()
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

class LLMAgent:
    def __init__(self):
        self.client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy-key")
        self.history: List[str] = []

    def select_action(self, obs) -> Action:
        prompt = f"""
        You are an SRE agent. 
        Your GOAL to complete the task: Disk usage must be safely under 90%, AND all services must be 'running'.
        
        Current State:
        - Disk Usage: {obs.disk_usage_percent}%
        - Services: {obs.services_status}
        
        Previous actions taken: {self.history[-3:] if self.history else 'None'}

        Rules for choosing a command:
        1. If Disk Usage is dangerously high (e.g., > 80%), you MUST clear space first. Reply with: rm -rf /tmp/*
        2. If you already tried 'rm' and Disk is still high, reply with: truncate -s 0 /var/log/*.log
        3. If Disk is safe (< 90%) but a service is NOT running, pick EXACTLY ONE stopped service and reply with: systemctl restart <service_name>
        4. If everything looks completely fine, reply with: df -h

        CRITICAL: Output ONLY a SINGLE command on a SINGLE line. No quotes, no markdown, no multiple commands.
        """
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=30
            )
            
            raw_text = response.choices[0].message.content.strip()
            text = raw_text.splitlines()[0].strip()
            
            parts = text.split(" ", 1)
            cmd = parts[0]
            args = parts[1] if len(parts) > 1 else ""
            return Action(command=cmd, args=args)
            
        except Exception as e:
            print(f"[DEBUG] LLM Call Failed: {e}. Falling back to safe action.", flush=True)
            return Action(command="df", args="-h")

async def main():
    env = IncidentEnv()
    agent = LLMAgent()
    
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    
    log_start(task=TASK_NAME, env="incident_triage", model=MODEL_NAME)

    try:
        obs = await env.reset(TASK_NAME)
        
        for step in range(1, MAX_STEPS + 1):
            action_obj = agent.select_action(obs)
            action_full_str = f"{action_obj.command} {action_obj.args}".strip()

            obs, reward, done, _ = await env.step(action_obj)
            agent.history.append(action_full_str)
            
            rewards.append(reward)
            steps_taken = step
            error_msg = obs.stderr if obs.exit_code != 0 else None
            
            log_step(step=step, action=action_full_str, reward=reward, done=done, error=error_msg)

            if done:
                break

        score = max(0.0, min(1.0, sum(rewards)))
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Runtime Error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())