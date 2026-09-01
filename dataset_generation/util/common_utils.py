# -*- coding: utf-8 -*-
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
    """Load an external configuration file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            # print(f"Configuration loaded successfully: {config_path}")
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"FATAL ERROR: Configuration file not found at {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"FATAL ERROR: Configuration file {config_path} has invalid JSON format.")

def setup_logging(output_dir: Path, log_file_name="process_log.txt"):
    log_path = output_dir / log_file_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # Set the log level

    # Set the formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Remove any existing handlers to avoid duplication
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 1. File handler (FileHandler)
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 2. Console handler (StreamHandler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # Ensure that loguru or any other logger in use is also configured
    # If you use loguru, extra configuration may be needed to bridge it

    return log_path

def save_json_data(data: list, output_path: Path, description: str):
    """
    Generic JSON saving function.
    :param data: the list data to save
    :param output_path: full path to save to (Path object)
    :param description: descriptive text used for logging
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
