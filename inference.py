import os
import sys
import asyncio

# Ensure local imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

async def run():
    TASK_NAME = os.getenv("TASK_ID", "cascading_failure_hard")
    MAX_STEPS = 10

    # 🔴 MUST PRINT FIRST
    print(f"[START] task={TASK_NAME} env=incident_triage model=rule_based", flush=True)

    rewards = []
    steps_taken = 0

    try:
        from src.environment import IncidentEnv
        from src.models import Action

        env = IncidentEnv()
        obs = await env.reset(TASK_NAME)

        cleaned_disk = False
        restarted_services = set()

        for step in range(1, MAX_STEPS + 1):
            steps_taken = step

            try:
                if obs.disk_usage_percent > 80 and not cleaned_disk:
                    action = Action(command="rm", args="-rf /tmp/*")
                    cleaned_disk = True

                else:
                    target_service = None
                    for s, status in obs.services_status.items():
                        if status != "running" and s not in restarted_services:
                            target_service = s
                            break

                    if target_service:
                        action = Action(command="restart", args=target_service)
                        restarted_services.add(target_service)
                    else:
                        action = Action(command="df", args="-h")

                action_str = f"{action.command} {action.args}".strip()

                obs, reward, done, _ = await env.step(action)
                rewards.append(reward)

                print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

                if done:
                    break

            except Exception as step_error:
                print(f"[STEP] step={step} action=none reward=0.00 done=false error={str(step_error)}", flush=True)

        score = max(0.0, min(1.0, sum(rewards)))
        success = str(score >= 0.5).lower()
        rewards_str = ",".join([f"{r:.2f}" for r in rewards]) if rewards else "0.00"

        print(f"[END] success={success} steps={steps_taken} score={score:.3f} rewards={rewards_str}", flush=True)

    except Exception as e:
        print(f"[STEP] step=1 action=none reward=0.00 done=true error={str(e)}", flush=True)
        print(f"[END] success=false steps=1 score=0.000 rewards=0.00", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(run())
    finally:
        sys.exit(0)