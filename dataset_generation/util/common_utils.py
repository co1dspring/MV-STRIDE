# -*- coding: gbk -*-
import random
import numpy as np
import math
from typing import Dict, List, Tuple, Set, Union, Any
from scipy.spatial.transform import Rotation as R
from icecream import ic
from pathlib import Path
import json
import logging
import sys
import os

def load_config(config_path: Path) -> Dict:
    """加载外部配置文件"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            # print(f"成功加载配置: {config_path}")
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"FATAL ERROR: Configuration file not found at {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"FATAL ERROR: Configuration file {config_path} has invalid JSON format.")

def setup_logging(output_dir: Path, log_file_name="process_log.txt"):
    log_path = output_dir / log_file_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取根记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # 设置日志级别

    # 设置格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 移除任何已存在的处理器，以防重复
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 1. 文件处理器 (FileHandler)
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 2. 控制台处理器 (StreamHandler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # 确保 loguru 或其他可能使用的 logger 也被配置
    # 如果您使用了 loguru，可能需要额外的配置来桥接

    return log_path

def save_json_data(data: list, output_path: Path, description: str):
    """
    通用 JSON 保存函数
    :param data: 要保存的列表数据
    :param output_path: 保存的完整路径 (Path 对象)
    :param description: 用于日志记录的描述性文字
    """
    if not data:
        logging.warning(f"跳过保存 {description}，数据为空。")
        return

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logging.info(f"已保存 {description}（{len(data)}条）到：{output_path}")
    except Exception as e:
        logging.error(f"保存 {description} 失败：{e}", exc_info=True)
