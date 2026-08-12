import base64
import os
import openai
import time
import requests
import httpx
from openai import OpenAI

# username = os.environ.get('USERNAME', 'z00770277')
username = 'z00770277'
# password = os.environ.get('PASSWORD', 'bjzt=2022')
password = 'bjzt=2022'
proxy_url = "proxynj.huawei.com"
os.environ["http_proxy"] = f"http://{username}:{password}@{proxy_url}:8080"
os.environ["https_proxy"] = f"http://{username}:{password}@{proxy_url}:8080"

# client = OpenAI(api_key='<KEY>')
# client.api_key = 'sk-iqtQZh0Oxb0qoerlDeepD6c6OC483j7E3tYHcIXSel2PS9iZ'
# client.api_key = sk-2gHJWhnV9hAMz98t4255F84c86F04aC9BcE38643E63a912f
# client.base_url = 'http://az.gptplus5.com'
# client.base_url = 'http://rerverseapi.workergpt.cn/v1'

# 定义需要重试的异常类型
def is_retryable_error(e):
    return isinstance(e, (
        openai.APIConnectionError,   # OpenAI 官方网络错误
        openai.APIError,             # 需要进一步检查状态码
        requests.ConnectionError,    # 底层网络错误
        requests.Timeout             # 请求超时
    ))

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def gpt4o_image_text_inference(idx, base64_images, user_prompt, system_prompt = None, model_name = "gpt-4o-2024-11-20", MAX_TRY_TIMES = 5, max_tokens = 12000, temperature = 0.2):
    # image_path = os.path.join(image_folder, image_filename)
    # base64_image = encode_image(image_path)
    # base64_images应该是一个list，方便多图像输入操作
    # 输入此函数的图像必须为base64格式
    client = OpenAI(
        base_url='http://az.gptplus5.com/v1',
        # api_key='sk-iqtQZh0Oxb0qoerlDeepD6c6OC483j7E3tYHcIXSel2PS9iZ',
        api_key='sk-gqful3d2J7Sy6spvWPZ1euazuJUVKsyAw0n9y36x1I2JtXlf',
        http_client=httpx.Client(verify=False)
    )

    messages = []

    if system_prompt is not None:
        system_message = {
            'role': 'system',
            'content': system_prompt
        }
        messages.append(system_message)

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
    user_message = {
        "role": "user",
        "content": input,
    }
    messages.append(user_message)

    try_times = 0
    response_lst = []
    success = False
    while not success:
        try:
            try_times += 1
            if try_times > MAX_TRY_TIMES:
                return None
            # 调用GPT-4 API生成空间推理问题
            response = client.chat.completions.create(
                # model="gpt-4o-2024-08-06",
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            # print(response)
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


def gpt4o_text_inference(idx, user_prompt, system_prompt=None, model_name="gpt-4o-2024-11-20", MAX_TRY_TIMES=5, temperature = 0.2):
    # 初始化客户端，注意 base_url 补全 /v1
    client = OpenAI(
        base_url='http://az.gptplus5.com/v1',
        api_key='sk-iqtQZh0Oxb0qoerlDeepD6c6OC483j7E3tYHcIXSel2PS9iZ',
        http_client=httpx.Client(verify=False)
    )

    messages = []

    # 构造 System Prompt
    if system_prompt is not None:
        messages.append({
            'role': 'system',
            'content': system_prompt
        })

    # 构造 User Prompt (纯文本格式)
    messages.append({
        "role": "user",
        "content": user_prompt,
    })

    try_times = 0
    response_lst = []

    while try_times < MAX_TRY_TIMES:
        try:
            try_times += 1
            # 调用 API
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=2048,
                temperature=temperature
            )

            # 使用 model_dump() 替代 to_dict()
            response_lst.append(response.model_dump())
            response_content = response.choices[0].message.content

            return response_content, response_lst

        except Exception as e:
            # 这里的 is_retryable_error 需确保已在外部定义
            if is_retryable_error(e):
                print(f"Text Request error for {idx} (Attempt {try_times}): {e}")
                # 稍微增加延迟，避免连续触发 Rate Limit
                time.sleep(1.0 * try_times)
            else:
                print(f"Fatal Text error for {idx}: {e}")
                return None

    # 超过最大重试次数
    print(f"Text Request for {idx} failed after {MAX_TRY_TIMES} attempts.")
    return None
