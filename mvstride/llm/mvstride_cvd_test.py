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

# Create a new jsonl file containing image filenames, captions, and generated questions
def save_to_jsonl(output_jsonl_file, results):
    with open(output_jsonl_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

# Read an image and encode it as base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Convert a video into a set of base64-encoded frames
def process_video(video):
    video = cv2.VideoCapture(video)
    base64Frames = []

    # fps = video.get(cv2.CAP_PROP_FPS)

    frame_jump = 1
    frame_count = 0

    # Target frame dimensions.
    target_width = 336
    target_height = 336

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        # Save one frame per second.
        if frame_count % frame_jump == 0:
            resized_frame = cv2.resize(frame, (target_width, target_height))
            _, buffer = cv2.imencode(".jpg", cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        frame_count += 1
    video.release()
    return base64Frames
    # print(len(base64Frames), "frames read.")


# Call the gpt4o API to generate a single output
def get_output(idx, item, user_prompt = None):
    # base64_images = base64.b64encode(item['image'])
    video = item['videos'][0]
    video_path = os.path.join(file_path, video.split('/')[-3], video.split('/')[-2], video.split('/')[-1])
    if not user_prompt:
        question = item['conversation'][0]['human'].replace("<video>", "")
        user_prompt = (
            f"Answer the question below based on the frames provided, "
            f"which are sampled at about {sample_fps:.2f} FPS.\n"
            f"Question: {question}"
        )
    if 'system' in item.keys():
        system_prompt = item['system']
    else:
        system_prompt = None
    base64_images = process_video(video_path)
    # sys.exit(0)


    output = gpt4o_image_text_inference(idx, base64_images, user_prompt, system_prompt = system_prompt, model_name = model_name)
    # print(output)

    return {'question': user_prompt, 'labels': item['conversation'][0]['assistant'], 'response': output}

# Call the gpt4o API to generate an output and store it, guarding against data loss on crash
def get_output_and_restore(idx, item, f):
    base64_images = item['image']
    question = item['question']

    answer_CoT = gpt4o_image_text_inference(idx, base64_images, question)

    if question:
        data = {'filename': item['image_file'], 'question': question, 'answer_GT': item['answer'], answer_CoT: answer_CoT}
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


# ZeroBench features: ['question_id', 'question_text', 'question_images_decoded', 'question_answer', 'question_images', 'image_attribution']
def main():
    test_name = 'qa'
    json_dir = os.path.join(file_path, test_name)
    output_dir = os.path.join(file_path, f'{test_name}_gpt4o')
    os.makedirs(output_dir, exist_ok=True)
    json_list = os.listdir(json_dir)
    for json_file in tqdm(json_list):
        # 1. Read the image annotation file.
        print(f"processing {json_file}")
        annotations = read_json(os.path.join(json_dir, json_file))
        # annotations = Dataset.from_parquet(os.path.join(file_path, filename))

        # 2. Call the gpt4o API in parallel and store responses directly.
        output = []
        thread_num = 5 # number of worker threads in the pool
        batch_size = 10 # save results every 10
        output_jsonl_file = os.path.join(output_dir, f"{os.path.splitext(json_file)[0]}_answer_4o.jsonl")#'object_occlusion_answer_4o.jsonl'

        # get_output(1, annotations[0])
        # sys.exit(0)

        with open(output_jsonl_file, 'w', encoding='utf-8') as f:
            with concurrent.futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
                futures = [executor.submit(get_output, idx, item) for idx, item in enumerate(annotations)]
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    try:
                        res = future.result()
                        output.append(copy.deepcopy(res))

                        # Once completed tasks reach batch_size, save them to file.
                        if len(output) >= batch_size:
                            # Write the collected JSON objects to file.
                            lines = [json.dumps(sample, ensure_ascii=False) + '\n' for sample in output]
                            f.writelines(lines)
                            # Clear the output list.
                            f.flush()
                            output = []
                    except Exception as e:
                        print(f"Error processing a task: {e}")
            # Save any remaining results.
            if output:
                lines = [json.dumps(sample, ensure_ascii=False) + '\n' for sample in output]
                f.writelines(lines)
                f.flush()

        f.close()
        print(f"Results have been saved to {output_jsonl_file}")

class APIDataProcesser:
    def __init__(self, config):
        """
                Initialize config.
                config should contain: base_path, model_name, thread_num, batch_size, sample_fps, etc.
                """
        self.config = config
        self.thread_num = config.get('thread_num', 5)
        self.batch_size = config.get('batch_size', 10)
        self.input_file_path = Path(config.get('input_file_path', './'))
        self.output_file_dir = Path(config.get('output_file_dir', './'))
        self.model_name = config.get('model_name', 'gpt-4o-2024-11-20')
        self.output_file_name = f"{self.input_file_path.stem}_{self.model_name}_CoT.jsonl"
        self.image_base_path = config.get('image_base_path', './')
        self.output_file_path = self.output_file_dir / self.output_file_name
        self.checkpoints_path = self.output_file_dir / f'checkpoints_{self.output_file_name.split(".")[0]}_json'
        self.stats_path = self.output_file_dir / 'cost_statistics'
        self.temperature = 0.0

        # prompt template
        self.SYSTEM_PROMPT = 'You are a helpful assistant. Please analyze the images and answer the question by outputting only the letter of the correct option (e.g., A, B, C, or D). Do not include any reasoning or extra text.'

        os.makedirs(self.output_file_dir, exist_ok=True)
        self.input_data = self.read_input_data()
        # random.seed(42)
        # random.shuffle(self.input_data)
        # self.input_data = self.input_data[:25]
        
        self.stats = {
            "total_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "failed_calls": 0,
            "total_cost": 0
        }

        # Thread locks keep concurrent file writes ordered.
        self.write_lock = threading.Lock()
        self.stats_lock = threading.Lock()

        # Pricing: Input $1.25/M, Output $10/M.
        self.PRICE_INPUT = 1.25 / 1000000
        self.PRICE_OUTPUT = 10.0 / 1000000

        self._setup_logger()
        self.checkpoints = self._read_checkpoints()
        self.all_data_num = len(self.input_data)
        self.processed_data_num = len(self.checkpoints)

    def _read_checkpoints(self):
        # Set of indices already processed successfully.
        successful_ids = set()
        if os.path.exists(self.checkpoints_path):
            data = read_json(self.checkpoints_path)
            # Checkpoint format: [{'idx': 0, 'success': True}, ...]
            successful_ids = {item['idx'] for item in data if item.get('success')}
        return successful_ids

    def _setup_logger(self):
        log_dir = self.output_file_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"process_{time.strftime('%Y%m%d_%H%M%S')}.log"

        self.logger = logging.getLogger(self.model_name)
        self.logger.setLevel(logging.INFO)

        # Format: time - thread name - level - message.
        formatter = logging.Formatter('%(asctime)s - [%(threadName)s] - %(levelname)s - %(message)s')

        # File output.
        fh = logging.FileHandler(log_path, encoding='utf-8')
        fh.setFormatter(formatter)

        # Console output.
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
            # Save a report whether the run finishes or exits early.
            # Note: save_stats_report must handle division by zero when total_calls is 0.
            self.save_stats_report()
            self.logger.info("程序已退出，统计数据已持久化。")

    def worker_task(self, idx, item):
        """Run the per-thread processing task."""
        try:
            system_prompt = self.SYSTEM_PROMPT
            user_prompt = self._format_user_prompt(item)
            base64_images = self._process_image_input(item)

            results = gpt4o_image_text_inference(
                idx,
                base64_images,
                user_prompt,
                system_prompt=system_prompt,
                model_name=self.model_name,
                temperature=self.temperature
            )
            if results is not None:
                output, raw_response = results
                # Track token usage from the raw response list.
                self._update_token_stats(raw_response)

                # Attach the model answer to the original record.
                output_piece = item
                output_piece['model_answer'] = output
                output_piece['response_lst'] = raw_response
                if output == '':
                    return False, output_piece
                return True, output_piece
            else:
                self.logger.warning(f"第 {idx} 条数据请求失败，跳过...")
                return False, ''

        except Exception as e:
            self.logger.warning(f"Error in worker_task at index {idx}: {e}")
            return False, ''

    def _format_user_prompt(self, item):
        user_prompt = item['messages'][0]['content']
        user_prompt = re.sub(r"<image>", "", user_prompt).strip()
        return user_prompt

    def _process_image_input(self, item):
        base64_images = []
        image_paths = item['images']
        for image_path in image_paths:
            if 'Infinigen' in image_path:
                local_image_path = os.path.join(self.image_base_path['infinigen'], *image_path.split('/')[-3:])
            else:
                local_image_path = os.path.join(self.image_base_path['scannetpp'], *image_path.split('/')[-3:])
            base64_image = encode_image(local_image_path)
            base64_images.append(base64_image)
        return base64_images

    def _update_token_stats(self, raw_response):
        """Parse and accumulate token usage across API providers."""
        if not raw_response:
            with self.stats_lock:
                self.stats["failed_calls"] += 1
            return

        with self.stats_lock:
            # Count successful calls.
            self.stats["total_calls"] += len(raw_response)

            for elem in raw_response:
                usage = elem.get("usage", {})

                i_tokens, o_tokens = 0, 0
                # Case 1: OpenAI format.
                if "prompt_tokens" in usage:
                    i_tokens = usage['prompt_tokens']
                    o_tokens = usage['completion_tokens']
                # Case 2: Claude/Anthropic format.
                elif "input_tokens" in usage:
                    i_tokens = usage['input_tokens']
                    o_tokens = usage['output_tokens']
                else:
                    continue

                # Accumulate token counts.
                self.stats["prompt_tokens"] += i_tokens
                self.stats["completion_tokens"] += o_tokens
                self.stats["total_tokens"] += (i_tokens + o_tokens)

                # Accumulate cost.
                self.stats["total_cost"] += (i_tokens * self.PRICE_INPUT + o_tokens * self.PRICE_OUTPUT)

    def _run_parallel_tasks(self, annotations, output_path):
        """Run tasks in parallel and save results as they complete."""
        results_buffer = []
        already_done_count = len(self.checkpoints)
        self.logger.info(f"Total: {len(annotations)}, Already processed: {already_done_count}")

        with open(output_path, 'a', encoding='utf-8') as f:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_num) as executor:
                # Submit all tasks.
                future_to_idx = {}
                for i, item in enumerate(annotations):
                    # Skip indices that already succeeded.
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
                            self.checkpoints.add(idx)  # Update in-memory state.

                    # Flush once the buffer reaches batch_size.
                    if len(results_buffer) >= self.batch_size:
                        self._flush_to_file(f, results_buffer)
                        results_buffer = []  # Clear the buffer.

                # Flush remaining results.
                if results_buffer:
                    self._flush_to_file(f, results_buffer)

    def _flush_to_file(self, file_handle, data_list):
        """Thread-safe file write."""
        with self.write_lock:
            # Write result JSONL.
            lines = [json.dumps(item, ensure_ascii=False) + '\n' for item in data_list]
            file_handle.writelines(lines)
            file_handle.flush()

            # Persist progress (checkpoint) by converting the set back to a list.
            
            checkpoint_data = [{"idx": i, "success": True} for i in self.checkpoints]
            with open(self.checkpoints_path, 'w', encoding='utf-8') as f_cp:
                json.dump(checkpoint_data, f_cp, ensure_ascii=False, indent=2)

    def save_stats_report(self):
        """Generate and save the final statistics report."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_name = f"stats_{self.output_file_path.stem}_{timestamp}.txt"
        report_path = self.stats_path / report_name

        # Read the metric values from the dictionary.
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
        self.logger.info(f"离线统计报告已保存至: {report_path}")

    def analyze_token_stats_from_file(self, price_input=1.25 / 1000000, price_output=10.0 / 1000000):
        """
        Standalone helper: recompute token usage and cost from an existing jsonl file.
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
                    # Read the raw responses from the stored 'response_lst' field.
                    raw_responses = item.get('response_lst', [])

                    total_stats["total_items"] += 1

                    for resp in raw_responses:
                        usage = resp.get("usage", {})

                        # Compatible with OpenAI and Anthropic/Claude token field names.
                        i_tokens = usage.get('prompt_tokens') or usage.get('input_tokens') or 0
                        o_tokens = usage.get('completion_tokens') or usage.get('output_tokens') or 0

                        total_stats["prompt_tokens"] += i_tokens
                        total_stats["completion_tokens"] += o_tokens
                        total_stats["total_tokens"] += (i_tokens + o_tokens)
                        total_stats["total_cost"] += (i_tokens * price_input + o_tokens * price_output)

                except Exception as e:
                    print(f"解析行失败: {e}")

        # Compute the average output length.
        avg_output = total_stats["completion_tokens"] / total_stats["total_items"] if total_stats["total_items"] > 0 else 0

        # Build the report string.
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

        # Save the report next to the analyzed file.
        report_path = path.parent / f"offline_stats_{path.stem}.txt"
        with open(report_path, 'w', encoding='utf-8') as f_out:
            f_out.write(report_str)

        print(f"离线统计报告已保存至: {report_path}")
        return total_stats

    def aggregate_all_stats(self):
        """
        Enhanced: iterate over all .txt reports in the directory and aggregate
        token usage, cost, total calls, and a global weighted-average output length.
        """
        total_metrics = {
            "files_processed": 0,
            "total_calls": 0,         
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
                file_calls = 0        

                # Parse call count (compatible with "Total API Calls", "Total Valid Items", "Total API Responses Processed").
                calls_match = re.search(r'(?:Total API Calls|Total Valid Items|Total API Responses Processed)\s*[:\s]\s*([\d,]+)', content)
                if calls_match:
                    file_calls = int(calls_match.group(1).replace(',', ''))
                    # A non-zero call count means the file has data even if tokens are 0.
                    file_has_data = True

                # Parse input tokens.
                input_match = re.search(r'Total Input Tokens\s*[:\s]\s*([\d,]+)', content)
                if input_match:
                    file_input = int(input_match.group(1).replace(',', ''))
                    file_has_data = True

                # Parse output tokens.
                output_match = re.search(r'Total Output Tokens\s*[:\s]\s*([\d,]+)', content)
                if output_match:
                    file_output = int(output_match.group(1).replace(',', ''))
                    file_has_data = True

                # Parse cost.
                cost_match = re.search(r'(?:Cost|Cost \(This Run\))\s*[:\s]\s*\$?\s*([\d,]+\.\d+)', content)
                if cost_match:
                    file_cost = float(cost_match.group(1).replace(',', ''))

                # Accumulate only if valid data was parsed.
                if file_has_data:
                    total_metrics["files_processed"] += 1
                    total_metrics["total_calls"] += file_calls
                    total_metrics["input_tokens"] += file_input
                    total_metrics["output_tokens"] += file_output
                    total_metrics["total_tokens"] += (file_input + file_output)
                    total_metrics["cost"] += file_cost
                    print(f"  [解析成功] {file_path.name}: Calls={file_calls}, In={file_input}, Out={file_output}, Cost=${file_cost:.4f}")
                else:
                    print(f"  [跳过空白] {file_path.name}:: 未提取到有效数据")

        # Compute the global weighted-average output length.
        avg_output_len = 0.0
        if total_metrics["total_calls"] > 0:
            avg_output_len = total_metrics["output_tokens"] / total_metrics["total_calls"]

        # Print the final aggregated report.
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
    import argparse

    parser = argparse.ArgumentParser(description="跨视角依赖测试集纯多选推理（evaluation 用）")
    parser.add_argument("--input-file", default='/path/to/data/cross_view_dependency_sampled_singleview_1000.json',
                        help="cross_view_dependency_test_sampling.py 的输出（默认占位符，需替换为本机实际文件）")
    parser.add_argument("--output-dir", default='./output', help="输出目录")
    args = parser.parse_args()

    # Credentials and proxy are read from the environment; do not hard-code secrets.
    username = os.environ.get("API_USERNAME")
    password = os.environ.get("API_PASSWORD")
    proxy_url = os.environ.get("PROXY_URL")
    if proxy_url:
        os.environ["http_proxy"] = f"http://{username}:{password}@{proxy_url}:8080"
        os.environ["https_proxy"] = f"http://{username}:{password}@{proxy_url}:8080"

    my_config = {
        'input_file_path': args.input_file,
        'output_file_dir': args.output_dir,
        'image_base_path': {'scannetpp': '/path/to/data/scannetpp_sampled_modified', 'infinigen': '/path/to/data/infinigen_metadata_ver2'},
        'model_name': 'gemini-3-flash-preview',
        'thread_num': 40,
        'batch_size': 40
    }

    processer = APIDataProcesser(my_config)
    processer.process_data()
    processer.aggregate_all_stats()
