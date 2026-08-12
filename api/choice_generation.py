# -*- coding = utf-8 -*-
import json
import base64
import jsonlines
import cv2
import concurrent.futures
from api_interface import gpt4o_image_text_inference, gpt4o_text_inference
from datasets import Dataset
import os
import json
import copy
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
import time
import threading
import random
import logging
import re

def read_json(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def read_jsonl(jsonl_file):
    annotations = []
    with jsonlines.open(jsonl_file) as reader:
        for obj in reader:
            annotations.append(obj)
    return annotations

# 创建新的jsonl文件，包含图像文件名、caption和生成的问题
def save_to_jsonl(output_jsonl_file, results):
    with open(output_jsonl_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

# 读取图像并使用base64编码
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class APIDataProcesser:
    def __init__(self, config):
        """
                初始化配置
                config 需包含: base_path, model_name, thread_num, batch_size, sample_fps 等
                """
        self.config = config
        self.thread_num = config.get('thread_num', 5)
        self.batch_size = config.get('batch_size', 10)
        self.input_file_path = Path(config.get('input_file_path', './'))
        self.output_file_dir = Path(config.get('output_file_dir', './'))
        self.model_name = config.get('model_name', 'gpt-4o-2024-11-20')
        self.output_file_name = f"{self.input_file_path.stem}_{self.model_name}_MCA.jsonl"
        self.image_base_path = config.get('image_base_path', './')
        self.output_file_path = self.output_file_dir / self.output_file_name
        self.checkpoints_path = self.output_file_dir / f'checkpoints_{self.output_file_name.split(".")[0]}_json'
        self.stats_path = self.output_file_dir / 'cost_statistics'
        self.temperature = 0.2

        # prompt template for SAT
        # self.SYSTEM_PROMPT = """You are a Spatial Reasoning Expert specializing in 3D and 2D visual analysis. Your expertise covers spatial orientation, relative distance, object counting, viewpoint transformation, and movement consequence.
        #
        # Your task is to refactor an open-ended Visual QA pair into a professional multiple-choice format (MCQ).
        #
        # **Strict Guidelines:**
        # 1. **Core Content Preservation**: DO NOT change the original intent of the question. Ensure the correct answer is derived directly and accurately from the "Original Answer".
        # 2. **Flexible Options**:
        #    - For standard questions (Counting, Orientation, Distance, etc.), provide **4 options (A, B, C, D)**.
        #    - For Yes/No or True/False questions, provide **2 options (A, B)**.
        # 3. **Plausible Distractors**: Create distractors that are logically related to the spatial context (e.g., if the answer is "Left", distractors should be other directions like "Right" or "Behind"). For numerical values, provide numbers in a similar range.
        # 4. **Output Format**: You must output a valid JSON object with EXACTLY the following structure:
        #    - "question_with_options": The original question followed by "\nOptions: A: ..., B: ..., C: ..."
        #    - "answer": The correct option in the format "Letter: Content" (e.g., "A: Yes" or "C: 1.3 meters").
        #
        # **Example 1 (Spatial Relation):**
        # Input Question: "Is the sofa to the left of the table?"
        # Input Answer: "No"
        # Output JSON:
        # {
        #     "question_with_options": "Is the sofa to the left of the table?\nOptions: A: Yes, B: No",
        #     "answer": "B: No"
        # }
        #
        # **Example 2 (Numerical/3D):**
        # Input Question: "What is the distance between the red point and blue point?"
        # Input Answer: "1.3 meters"
        # Output JSON:
        # {
        #     "question_with_options": "What is the distance between the red point and blue point?\nOptions: A: 0.5 meters, B: 1.3 meters, C: 2.1 meters, D: 3.0 meters",
        #     "answer": "B: 1.3 meters"
        # }"""

        # for SPAR
        self.SYSTEM_PROMPT = """
        You are a Spatial Reasoning Expert specializing in 3D and 2D visual analysis. Your expertise covers spatial orientation, relative distance, object counting, viewpoint transformation, and movement consequence.

        Your task is to refactor an open-ended Visual QA pair into a professional multiple-choice format (MCQ).

        **Strict Guidelines:**
        1. **Core Answer Extraction**: Since the "Original Answer" might be a full sentence or paragraph, you must first extract the core spatial factual conclusion as the "Correct Option". Keep it concise but accurate.
        2. **Core Content Preservation**: DO NOT change the original intent of the question. The choice you create must be directly supported by the factual details in the "Original Answer".
        3. **Flexible Options**:
           - For standard spatial reasoning (Position, Movement, Count), provide **4 options (A, B, C, D)**.
           - For simple judgment (Yes/No, True/False), provide **2 options (A, B)**.
        4. **Plausible Distractors**: Create distractors that represent logical spatial alternatives (e.g., swapping "Left/Right", "Closer/Farther", or "In front/Behind"). For complex movement consequences, distractors should reflect incorrect spatial transformations.
        5. **Output Format**: You must output a valid JSON object with EXACTLY the following structure:
           - "question_with_options": The original question followed by "\nOptions: A: ..., B: ..., C: ..."
           - "answer": The correct option in the format "Letter: Content" (e.g., "A: Left and below, closer").

        **Example (Complex Spatial/SPAR):**
        Input Question: "How are light switch and paper spatially related? What changes after observer relocating?"
        Input Answer: "Before relocating, light switch is to the left and below with respect to paper. After reaching printer, it is observed as closer."
        Output JSON:
        {
            "question_with_options": "How are light switch and paper spatially related? What changes after observer relocating?\nOptions: A: Right and above, farther, B: Left and below, closer, C: Left and above, same distance, D: Right and below, closer",
            "answer": "B: Left and below, closer"
        }
        """

        os.makedirs(self.output_file_dir, exist_ok=True)
        self.input_data = self.read_input_data()
        # random.seed(42)
        # random.shuffle(self.input_data)
        # self.input_data = self.input_data[:25]
        # 新增：统计信息汇总
        self.stats = {
            "total_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "failed_calls": 0,
            "total_cost": 0
        }

        # 线程锁，确保多线程写入文件时不会乱序
        self.write_lock = threading.Lock()
        self.stats_lock = threading.Lock()

        # 价格系数 (根据你提供的代码：Input $1.25/M, Output $10/M)
        self.PRICE_INPUT = 1.25 / 1000000
        self.PRICE_OUTPUT = 10.0 / 1000000

        self._setup_logger()
        self.checkpoints = self._read_checkpoints()
        self.all_data_num = len(self.input_data)
        self.processed_data_num = len(self.checkpoints)

    def _read_checkpoints(self):
        # 建立一个 Set 存储所有已成功处理的 idx
        successful_ids = set()
        if os.path.exists(self.checkpoints_path):
            data = read_json(self.checkpoints_path)
            # 假设 checkpoint 存的是 [{'idx': 0, 'success': True}, ...]
            successful_ids = {item['idx'] for item in data if item.get('success')}
        return successful_ids

    def _setup_logger(self):
        log_dir = self.output_file_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"process_{time.strftime('%Y%m%d_%H%M%S')}.log"

        self.logger = logging.getLogger(self.model_name)
        self.logger.setLevel(logging.INFO)

        # 格式：时间 - 线程名 - 级别 - 信息
        formatter = logging.Formatter('%(asctime)s - [%(threadName)s] - %(levelname)s - %(message)s')

        # 文件输出
        fh = logging.FileHandler(log_path, encoding='utf-8')
        fh.setFormatter(formatter)

        # 控制台输出
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def read_input_data(self):
        return read_json(self.input_file_path)

    def process_data(self):
        try:
            self._run_parallel_tasks(self.input_data, self.output_file_path)
        except (KeyboardInterrupt, SystemExit):
            self.logger.warning("收到中断请求，正在安全关闭...")
        except Exception as e:
            self.logger.error(f"运行时发生致命错误: {e}")
        finally:
            # 无论正常结束还是中途退出，均保存一次报告
            # 注意：save_stats_report 内部需要处理 total_calls 为 0 的除零错误
            self.save_stats_report()
            self.logger.info("程序已退出，统计数据已持久化。")

    def worker_task(self, idx, item):
        """单个线程执行的任务"""
        try:
            # 1. 准备输入
            system_prompt = self.SYSTEM_PROMPT
            user_prompt = self._format_user_prompt(item)
            # base64_images = self._process_image_input(item)

            # 2. 调用推理接口 (假设 gpt4o_image_text_inference 已在外部定义)
            # results = gpt4o_image_text_inference(
            #     idx,
            #     base64_images,
            #     user_prompt,
            #     system_prompt=system_prompt,
            #     model_name=self.model_name,
            #     temperature=self.temperature
            # )
            results = gpt4o_text_inference(
                idx,
                user_prompt,
                system_prompt=system_prompt,
                model_name=self.model_name,
                temperature=self.temperature
            )
            if results is not None:
                output, raw_response = results
                print(output)
                # 3. 统计 Token (从 response_lst 提取)
                self._update_token_stats(raw_response)

                # 4. 构造返回格式
                # 应当保存的是训练数据格式，只保留level3问题和生成的cot回答作为messages，原数据可以改名为其他key保存，同时也要保存调用api过程参数
                output_piece = self._process_api_response(output, item)
                output_piece['response_lst'] = raw_response
                if output == '':
                    return False, output_piece
                return True, output_piece
            else:
                self.logger.warning(f"第 {idx} 条数据请求失败，跳过...")
                return False, ''

            # output, raw_response = gpt4o_text_inference(
            #     idx,
            #     user_prompt,
            #     system_prompt=system_prompt,
            #     model_name=self.model_name
            # )
            # output, raw_response = gpt4o_text_inference(
            #     1,
            #     "who are you?",
            #     model_name=self.model_name
            # )


        except Exception as e:
            self.logger.warning(f"Error in worker_task at index {idx}: {e}")
            return False, ''

    def _format_user_prompt(self, item):
        """提取单轮对话，剥离图像标签"""
        user_msg = item['messages'][0]['content']
        assistant_msg = item['messages'][1]['content']

        # 彻底去掉 <image> 和 <video> 标签及其前后的换行
        clean_question = user_msg.replace("<image>", "").replace("<video>", "").strip()
        clean_answer = assistant_msg.strip()

        user_prompt = f"Original Question: {clean_question}\nOriginal Answer: {clean_answer}"
        return user_prompt

    def _process_api_response(self, api_output_str, original_item):
        """
        api_output_str: 可能包含 Markdown 标签或杂质的字符串
        original_item: 原始数据
        """
        try:
            # 1. 使用正则表达式提取最外层的大括号内容
            #
            json_match = re.search(r'(\{.*\}|\[.*\])', api_output_str, re.DOTALL)

            if json_match:
                clean_json_str = json_match.group(1)
            else:
                # 如果没找到大括号，记录并跳过
                self.logger.error(f"无法在输出中找到 JSON 结构: {api_output_str[:100]}...")
                return None

            # 2. 解析清洗后的字符串
            res = json.loads(clean_json_str)

            # 3. 提取字段
            new_q = res.get("question_with_options", "")
            new_a = res.get("answer", "")

            if not new_q or not new_a:
                return None

            # 4. 组装数据逻辑（保持不变）
            images = original_item.get("images", [])
            image_prefix = "<image>" * len(images)

            final_item = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"{image_prefix}{new_q}\n"
                    },
                    {
                        "role": "assistant",
                        "content": new_a
                    }
                ],
                "images": images,
                'old_messages': original_item['messages']
            }
            # print(final_item)
            return final_item

        except Exception as e:
            self.logger.error(f"解析失败: {e} | 原始输出: {api_output_str[:100]}...")
            return None

    def _process_image_input(self, item):
        base64_images = []
        image_paths = item['images']
        for image_path in image_paths:
            local_image_path = os.path.join(self.image_base_path, *image_path.split('/')[-3:])
            base64_image = encode_image(local_image_path)
            base64_images.append(base64_image)
        return base64_images

    def _update_token_stats(self, raw_response):
        """解析并累加多平台 API 的 Token 消耗"""
        if not raw_response:
            with self.stats_lock:
                self.stats["failed_calls"] += 1
            return

        with self.stats_lock:
            # 记录成功的调用次数
            self.stats["total_calls"] += len(raw_response)

            for elem in raw_response:
                usage = elem.get("usage", {})

                i_tokens, o_tokens = 0, 0
                # 情况 1: OpenAI 格式
                if "prompt_tokens" in usage:
                    i_tokens = usage['prompt_tokens']
                    o_tokens = usage['completion_tokens']
                # 情况 2: Claude/Anthropic 格式
                elif "input_tokens" in usage:
                    i_tokens = usage['input_tokens']
                    o_tokens = usage['output_tokens']
                else:
                    continue

                # 累加到字典中
                self.stats["prompt_tokens"] += i_tokens
                self.stats["completion_tokens"] += o_tokens
                self.stats["total_tokens"] += (i_tokens + o_tokens)

                # 累加成本
                self.stats["total_cost"] += (i_tokens * self.PRICE_INPUT + o_tokens * self.PRICE_OUTPUT)

    def _run_parallel_tasks(self, annotations, output_path):
        """并行执行并实时保存"""
        results_buffer = []
        already_done_count = len(self.checkpoints)
        self.logger.info(f"Total: {len(annotations)}, Already processed: {already_done_count}")

        with open(output_path, 'a', encoding='utf-8') as f:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_num) as executor:
                # 提交所有任务
                future_to_idx = {}
                for i, item in enumerate(annotations):
                    # 关键：跳过已成功的 idx
                    if i in self.checkpoints:
                        continue
                    future_to_idx[executor.submit(self.worker_task, i, item)] = i

                if not future_to_idx:
                    self.logger.info("No new data to process.")
                    return

                pbar = tqdm(concurrent.futures.as_completed(future_to_idx), total=len(future_to_idx))
                for future in pbar:
                    idx = future_to_idx[future]
                    success, output = future.result()
                    if success:
                        results_buffer.append(output)
                        with self.stats_lock:
                            self.checkpoints.add(idx)  # 更新内存中的状态

                    # 达到 batch_size 写入文件
                    if len(results_buffer) >= self.batch_size:
                        self._flush_to_file(f, results_buffer)
                        results_buffer = []  # 清空缓冲

                # 处理剩余数据
                if results_buffer:
                    self._flush_to_file(f, results_buffer)

    def _flush_to_file(self, file_handle, data_list):
        """确保线程安全的文件写入"""
        with self.write_lock:
            # 1. 写入结果 JSONL
            lines = [json.dumps(item, ensure_ascii=False) + '\n' for item in data_list]
            file_handle.writelines(lines)
            file_handle.flush()

            # 2. 写入进度 JSON (持久化)
            # 将 set 转回列表格式保存
            checkpoint_data = [{"idx": i, "success": True} for i in self.checkpoints]
            with open(self.checkpoints_path, 'w', encoding='utf-8') as f_cp:
                json.dump(checkpoint_data, f_cp, ensure_ascii=False, indent=2)

    def save_stats_report(self):
        """生成并保存最终统计报告"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_name = f"stats_{self.output_file_path.stem}_{timestamp}.txt"
        report_path = self.stats_path / report_name

        # 从字典取值
        total_calls = self.stats["total_calls"]
        avg_output = self.stats["completion_tokens"] / total_calls if total_calls > 0 else 0

        report_content = [
            f"Report Generated at: {timestamp}",
            f"Input File: {self.config.get('input_file_path')}",
            f"Model Name: {self.model_name}",
            f"-" * 20,
            f"Task Progress Statistics:",
            f"  - Total Dataset Size: {self.all_data_num}",
            f"  - Previously Processed: {self.processed_data_num}",
            f"  - Newly Succeeded (This Run): {len(self.checkpoints)-self.processed_data_num}",
            f"  - Newly Failed (This Run): {self.stats['failed_calls']}",
            f"-" * 20,
            f"Token & Cost Statistics (This Run):",
            f"  - Total API Calls: {total_calls}",
            f"  - Total Input Tokens: {self.stats['prompt_tokens']}",
            f"  - Total Output Tokens: {self.stats['completion_tokens']}",
            f"  - Total Tokens consumed: {self.stats['total_tokens']}",
            f"  - Average Output Tokens/Call: {avg_output:.2f}",
            f"  - Estimated Cost (This Run): ${self.stats['total_cost']:.4f}"
        ]

        report_str = "\n".join(report_content)
        self.logger.info("\n" + "=" * 40 + "\n" + report_str + "\n" + "=" * 40)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_str)
        self.logger.info(f"Stats report saved to: {report_path}")

    def analyze_token_stats_from_file(self, price_input=1.25 / 1000000, price_output=10.0 / 1000000):
        """
        独立函数：从已经生成的 jsonl 文件中重新统计 token 消耗和成本
        """

        path = self.output_file_path
        if not path.exists():
            print(f"错误：找不到文件 {self.output_file_path}")
            return

        total_stats = {
            "total_items": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0
        }

        print(f"开始分析文件: {path.name} ...")
        n = 40

        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    # 从你保存的 response_lst 字段提取
                    raw_responses = item.get('response_lst', [])

                    total_stats["total_items"] += 1

                    for resp in raw_responses:
                        usage = resp.get("usage", {})

                        # 兼容 OpenAI 和 Anthropic/Claude 的 Token 字段名
                        i_tokens = usage.get('prompt_tokens') or usage.get('input_tokens') or 0
                        o_tokens = usage.get('completion_tokens') or usage.get('output_tokens') or 0

                        total_stats["prompt_tokens"] += i_tokens
                        total_stats["completion_tokens"] += o_tokens
                        total_stats["total_tokens"] += (i_tokens + o_tokens)
                        total_stats["total_cost"] += (i_tokens * price_input + o_tokens * price_output)

                except Exception as e:
                    print(f"解析行失败: {e}")

        # 计算平均值
        avg_output = total_stats["completion_tokens"] / total_stats["total_items"] if total_stats["total_items"] > 0 else 0

        # 生成报告字符串
        report = [
            f"{'=' * 40}",
            f"OFFLINE TOKEN ANALYSIS REPORT",
            f"{'=' * 40}",
            f"File Analyzed     : {path.name}",
            f"Total Valid Items : {total_stats['total_items']}",
            f"-" * 20,
            f"Total Input Tokens      : {total_stats['prompt_tokens']:,}",
            f"Total Output Tokens     : {total_stats['completion_tokens']:,}",
            f"Total Tokens Consumed   : {total_stats['total_tokens']:,}",
            f"Avg Output Per Item     : {avg_output:.2f}",
            f"-" * 20,
            f"Estimated Total Cost    : ${total_stats['total_cost']:.4f}",
            f"{'=' * 40}"
        ]

        report_str = "\n".join(report)
        print(report_str)

        # 保存报告到同级目录
        report_path = path.parent / f"offline_stats_{path.stem}.txt"
        with open(report_path, 'w', encoding='utf-8') as f_out:
            f_out.write(report_str)

        print(f"离线统计报告已保存至: {report_path}")
        return total_stats

    def aggregate_all_stats(self):
        """
        增强版：遍历目录下的所有txt文件，解析并汇总：
        Token、成本、总调用次数、以及全局加权平均输出长度。
        """
        total_metrics = {
            "files_processed": 0,
            "total_calls": 0,         # 新增
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0
        }

        path_list = list(self.stats_path.glob("*.txt"))
        print(f"正在扫描目录: {self.stats_path}，共发现 {len(path_list)} 个统计文件...\n")

        for file_path in path_list:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

                file_has_data = False
                file_input = 0
                file_output = 0
                file_cost = 0.0
                file_calls = 0        # 新增

                # 1. 解析调用次数 (兼容 "Total API Calls", "Total Valid Items", "Total API Responses Processed")
                calls_match = re.search(r'(?:Total API Calls|Total Valid Items|Total API Responses Processed)\s*[:\s]\s*([\d,]+)', content)
                if calls_match:
                    file_calls = int(calls_match.group(1).replace(',', ''))
                    # 即使 tokens 为 0 (比如报错的报告)，只要有调用次数也算有数据
                    file_has_data = True

                # 2. 解析 Input Tokens
                input_match = re.search(r'Total Input Tokens\s*[:\s]\s*([\d,]+)', content)
                if input_match:
                    file_input = int(input_match.group(1).replace(',', ''))
                    file_has_data = True

                # 3. 解析 Output Tokens
                output_match = re.search(r'Total Output Tokens\s*[:\s]\s*([\d,]+)', content)
                if output_match:
                    file_output = int(output_match.group(1).replace(',', ''))
                    file_has_data = True

                # 4. 解析 Cost
                cost_match = re.search(r'(?:Cost|Cost \(This Run\))\s*[:\s]\s*\$?\s*([\d,]+\.\d+)', content)
                if cost_match:
                    file_cost = float(cost_match.group(1).replace(',', ''))

                # 如果解析到了有效数据，累加到总计
                if file_has_data:
                    total_metrics["files_processed"] += 1
                    total_metrics["total_calls"] += file_calls
                    total_metrics["input_tokens"] += file_input
                    total_metrics["output_tokens"] += file_output
                    total_metrics["total_tokens"] += (file_input + file_output)
                    total_metrics["cost"] += file_cost
                    print(f"  [解析成功] {file_path.name}: Calls={file_calls}, In={file_input}, Out={file_output}, Cost=${file_cost:.4f}")
                else:
                    print(f"  [跳过空白] {file_path.name}: 未提取到有效数据")

        # 计算全局加权平均输出长度
        avg_output_len = 0.0
        if total_metrics["total_calls"] > 0:
            avg_output_len = total_metrics["output_tokens"] / total_metrics["total_calls"]

        # 打印最终汇总报告
        print("\n" + "=" * 60)
        print("                全实验运行记录汇总审计报告")
        print("=" * 60)
        print(f"处理文件总数:      {total_metrics['files_processed']}")
        print(f"累计总调用次数:    {total_metrics['total_calls']:,}")
        print("-" * 40)
        print(f"累计输入 Tokens:    {total_metrics['input_tokens']:,}")
        print(f"累计输出 Tokens:    {total_metrics['output_tokens']:,}")
        print(f"累计消耗 Tokens:    {total_metrics['total_tokens']:,}")
        print("-" * 40)
        print(f"全局平均输出长度:   {avg_output_len:.2f} tokens/call")
        print(f"累计总花费 (USD):   ${total_metrics['cost']:.4f}")
        print("=" * 60)

        return total_metrics

if __name__ == "__main__":
    # username = os.environ.get('USERNAME', 'z00770277')
    username = 'z00770277'
    # password = os.environ.get('PASSWORD', 'bjzt=2022')
    password = 'bjzt=2022'
    proxy_url = "proxynj.huawei.com"
    os.environ["http_proxy"] = f"http://{username}:{password}@{proxy_url}:8080"
    os.environ["https_proxy"] = f"http://{username}:{password}@{proxy_url}:8080"

    my_config = {
        # 'input_file_path': r'D:\Data\SAT\SAT_stage3_grpo_sampled.json',
        'input_file_path': r'D:\Data\SPAR\SPAR_stage3_grpo_sampled.json',
        # 'input_file_path': 'D:/Data/infinigen_20251031/QA_jsons_ScannetppIphone_MultilevelCategories_20260121_sampled_MCA_Multistage/conversation/level_3/Positional Relationship(Obj.-Obj.).json',
        'output_file_dir': './output',
        'image_base_path': r"D:\Data\scannetpp\scannetpp_sampled_modified",
        # 'model_name': 'gpt-5',
        'model_name': 'gemini-3-flash-preview',
        # 'model_name': 'gemini-3-pro-preview',
        'thread_num': 20,
        'batch_size': 20
    }

    processer = APIDataProcesser(my_config)
    processer.process_data()
    # processer.analyze_token_stats_from_file()
    # processer.aggregate_all_stats()
