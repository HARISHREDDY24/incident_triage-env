import os
import sys

# Standardized Error Handling for Imports
try:
    from openenv import openenv_inference
    from openai import OpenAI
except ImportError as e:
    print(f"CRITICAL: Missing dependency: {e}")
    print("Ensure 'openenv-core' and 'openai' are in pyproject.toml.")
    sys.exit(1)

def run_inference(server_url: str):
    # Your secrets (must be set in HF Space Settings)
    api_key = os.getenv("HF_TOKEN")
    
    if not api_key:
        print("ERROR: HF_TOKEN secret not found.")
        sys.exit(1)

    try:
        client = OpenAI(
            api_key=api_key, 
            base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        )

        result = openenv_inference(
            server_url=server_url,
            model_name=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"),
            client=client,
            task_id="cascading_failure_hard"
        )
        return result

    except Exception as e:
        print(f"Deep Validation Failure: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Use the URL passed by the validator, or default to local
    target = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"
    run_inference(target)