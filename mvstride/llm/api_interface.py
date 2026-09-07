import base64
import os
import openai
import time
import requests
import httpx
from openai import OpenAI

# Credentials and proxy are read from the environment; do not hard-code secrets.
username = os.environ.get("API_USERNAME")
password = os.environ.get("API_PASSWORD")
proxy_url = os.environ.get("PROXY_URL")
if proxy_url:
    os.environ["http_proxy"] = f"http://{username}:{password}@{proxy_url}:8080"
    os.environ["https_proxy"] = f"http://{username}:{password}@{proxy_url}:8080"

# Retryable error types (network / rate-limit related).
def is_retryable_error(e):
    return isinstance(e, (
        openai.APIConnectionError,   # official OpenAI network error
        openai.APIError,             # requires further status-code inspection
        requests.ConnectionError,    # low-level network error
        requests.Timeout             # request timeout
    ))

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def gpt4o_image_text_inference(idx, base64_images, user_prompt, system_prompt=None, model_name="gpt-4o", MAX_TRY_TIMES=5, max_tokens=12000, temperature=0.2):
    # Accepts a list of base64-encoded images for multi-image input.
    client = OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        http_client=httpx.Client(verify=False)
    )

    messages = []

    if system_prompt is not None:
        messages.append({
            'role': 'system',
            'content': system_prompt
        })

    input = [
        {
            "type": "text",
            "text": user_prompt,
        }
    ]
    if not isinstance(base64_images, list):
        base64_images = [base64_images]
    images = [{
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
    } for base64_image in base64_images]
    input.extend(images)
    messages.append({
        "role": "user",
        "content": input,
    })

    try_times = 0
    response_lst = []
    success = False
    while not success:
        try:
            try_times += 1
            if try_times > MAX_TRY_TIMES:
                return None
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            response_lst.append(response.to_dict())
            response_content = response.choices[0].message.content
            success = True
            return response_content, response_lst
        except Exception as e:
            if is_retryable_error(e):
                print(f"Request error for {idx}: {e}")
                time.sleep(0.1)
            else:
                print(f"Error generating question for {idx}: {e}")
                return None


def gpt4o_text_inference(idx, user_prompt, system_prompt=None, model_name="gpt-4o", MAX_TRY_TIMES=5, temperature=0.2):
    client = OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        http_client=httpx.Client(verify=False)
    )

    messages = []

    # Construct the system prompt.
    if system_prompt is not None:
        messages.append({
            'role': 'system',
            'content': system_prompt
        })

    # Construct the user prompt (plain text).
    messages.append({
        "role": "user",
        "content": user_prompt,
    })

    try_times = 0
    response_lst = []

    while try_times < MAX_TRY_TIMES:
        try:
            try_times += 1
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=2048,
                temperature=temperature
            )

            response_lst.append(response.model_dump())
            response_content = response.choices[0].message.content

            return response_content, response_lst

        except Exception as e:
            if is_retryable_error(e):
                print(f"Text Request error for {idx} (Attempt {try_times}): {e}")
                time.sleep(1.0 * try_times)
            else:
                print(f"Fatal Text error for {idx}: {e}")
                return None

    # Exceeded the maximum number of retries.
    print(f"Text Request for {idx} failed after {MAX_TRY_TIMES} attempts.")
    return None
