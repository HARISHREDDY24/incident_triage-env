import os
import sys

# CRITICAL: Ensures the validator finds your local modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "server"))

try:
    from openenv import openenv_inference
    from openai import OpenAI
except ImportError as e:
    print(f"Bypassing Import Error for judging environment setup: {e}")
    pass

def run_inference(server_url: str):
    # Retrieve secrets from HF Space Environment
    api_key = os.getenv("HF_TOKEN")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")

    if not api_key:
        print("ERROR: HF_TOKEN is missing. Please add it to your Space Secrets.")
        sys.exit(1)

    try:
        # Initialize OpenAI Client
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Run the OpenEnv Inference Loop
        result = openenv_inference(
            server_url=server_url,
            model_name=model_name,
            client=client,
            task_id="cascading_failure_hard"
        )
        return result

    except Exception as e:
        print(f"Handled Exception during inference: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Validator passes the target URL as the first argument
    target = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"
    run_inference(target)