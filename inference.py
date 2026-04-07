import os
import sys
from openenv import openenv_inference

# Defensive Import Check
try:
    from openai import OpenAI
except ImportError:
    print("CRITICAL: The 'openai' library is missing. Please check pyproject.toml.")
    sys.exit(1)

def run_inference(server_url: str):
    # Retrieve secrets from environment variables
    api_key = os.getenv("HF_TOKEN")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")

    if not api_key:
        print("ERROR: HF_TOKEN secret is missing in Space Settings.")
        sys.exit(1)

    try:
        # Initialize OpenAI Client
        client = OpenAI(api_key=api_key, base_url=base_url)

        # Run the OpenEnv Inference
        # This function handles the loop between the LLM and your Space
        result = openenv_inference(
            server_url=server_url,
            model_name=model_name,
            client=client,
            task_id="cascading_failure_hard"
        )
        
        return result

    except Exception as e:
        print(f"FAILED: An unhandled exception occurred during inference: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # The validator passes the Space URL as the first argument
    target_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"
    run_inference(target_url)