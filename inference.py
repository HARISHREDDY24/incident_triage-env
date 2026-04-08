import os
import sys
import asyncio
from typing import List, Optional, Set

# Fix path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from src.environment import IncidentEnv
    from src.models import Action
except Exception as e:
    print(f"Import Error: {e}")
    sys.exit(0)


TASK_NAME = os.getenv("TASK_ID", "cascading_failure_hard")
MAX_STEPS = 10
SUCCESS_THRESHOLD = 0.5


def log_start(task: str):
    print(f"[START] task={task} env=incident_triage model=rule_based", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


class SafeAgent:
    def __init__(self):
        self.cleaned = False
        self.restarted: Set[str] = set()

    def act(self, obs) -> Action:
        try:
            # Step 1: Fix disk
            if obs.disk_usage_percent >= 50 and not self.cleaned:
                self.cleaned = True
                return Action(command="rm", args="-rf /tmp/*")

            # Step 2: Restart services
            for s, status in obs.services_status.items():
                if status != "running" and s not in self.restarted:
                    self.restarted.add(s)
                    return Action(command="restart", args=s)

            # Step 3: Safe fallback
            return Action(command="df", args="")

        except Exception:
            return Action(command="df", args="")


async def run():
    try:
        env = IncidentEnv()
        agent = SafeAgent()

        rewards = []
        steps_taken = 0

        log_start(TASK_NAME)

        obs = await env.reset(TASK_NAME)

        for step in range(1, MAX_STEPS + 1):
            try:
                action = agent.act(obs)
                action_str = f"{action.command} {action.args}".strip()

                obs, reward, done, _ = await env.step(action)

                rewards.append(reward)
                steps_taken = step

                error_msg = obs.stderr if obs.exit_code != 0 else None

                log_step(step, action_str, reward, done, error_msg)

                if done:
                    break

            except Exception as inner_error:
                print(f"[STEP ERROR] {inner_error}", flush=True)
                break

        score = max(0.0, min(1.0, sum(rewards)))
        success = score >= SUCCESS_THRESHOLD

        log_end(success, steps_taken, score, rewards)

    except Exception as e:
        print(f"[FATAL ERROR] {e}", flush=True)

    # CRITICAL: always exit cleanly
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(run())