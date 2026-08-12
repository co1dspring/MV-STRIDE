import sys
import os
import json
import random
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QScrollArea, QSizePolicy, QTextEdit, QMessageBox
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from typing import List, Dict, Any, Union
from pathlib import Path

# ----------------------------------------------------------------------
# 新增：分类数据加载器
# ----------------------------------------------------------------------
class CategoryLoader:
    """负责发现和按需加载不同分类的JSON文件"""

    def __init__(self, json_dir_path: str):
        self.json_dir = Path(json_dir_path)
        self.categories: List[str] = []
        self._data_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.find_categories()

    def find_categories(self):
        """扫描目录，找出所有 .json 文件，并将文件名（不含扩展名）作为分类"""
        if not self.json_dir.is_dir():
            print(f"错误：JSON 目录不存在或不是目录 -> {self.json_dir}")
            return

        json_files = sorted(self.json_dir.glob("*.json"))
        # 使用文件名（不带扩展名）作为分类名
        self.categories = [f.stem for f in json_files]
        if not self.categories:
            print(f"警告：未在 {self.json_dir} 中找到任何 JSON 文件。")

    def load_category_data(self, category_name: str) -> List[Dict[str, Any]]:
        MAX_ITEMS_TO_LOAD = 200  # <--- 设置一个最大加载条目数
        """从缓存或文件中加载指定分类的数据"""
        if category_name in self._data_cache:
            return self._data_cache[category_name]

        if category_name not in self.categories:
            print(f"错误：分类 {category_name} 不存在。")
            return []

        file_path = self.json_dir / f"{category_name}.json"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    print(f"警告：文件 {file_path} 内容不是列表。")
                    data = []
                random.shuffle(data)
                # --- 限制加载条目数 ---
                original_count = len(data)
                if original_count > MAX_ITEMS_TO_LOAD:
                    data = data[:MAX_ITEMS_TO_LOAD]
                    print(f"注意：分类 {category_name} 原始 {original_count} 条，已截断至 {MAX_ITEMS_TO_LOAD} 条。")
                self._data_cache[category_name] = data
                print(f"成功加载分类数据：{category_name} ({len(data)} 条)")
                return data
        except json.JSONDecodeError as e:
            print(f"错误：JSON 文件格式不正确 -> {file_path}. 错误: {e}")
            return []
        except Exception as e:
            print(f"读取文件失败: {file_path}. 错误: {e}")
            return []


class DataViewer(QWidget):
    # 将 data_list 替换为 loader 对象
    def __init__(self, category_loader: CategoryLoader, image_base_path: str):
        super().__init__()
        self.loader = category_loader
        self.image_base_path = image_base_path

        # --- 在这里设置窗口大小 ---
        INITIAL_WIDTH = 2400
        INITIAL_HEIGHT = 1200

        # 推荐使用 resize 设置初始大小，允许用户调整
        self.resize(INITIAL_WIDTH, INITIAL_HEIGHT)

        # 直接从 loader 获取分类列表
        self.categories = self.loader.categories
        self.current_category_index = 0

        # 初始加载第一个分类的数据
        if self.categories:
            first_category = self.categories[0]
            self.current_category_data = self.loader.load_category_data(first_category)
        else:
            self.current_category_data = []

        self.current_item_index_in_category = 0
        self.total_item_count = self._get_total_count()  # 新增总数计算

        self.initUI()
        self.update_display()

    def _get_total_count(self):
        """计算所有已加载分类的数据总条数"""
        total = 0
        for data_list in self.loader._data_cache.values():
            total += len(data_list)
        return total

    def initUI(self):
        """初始化用户界面"""
        self.setWindowTitle('JSON数据可视化工具')
        self.setGeometry(100, 100, 2400, 1900)

        # 主垂直布局
        main_layout = QVBoxLayout()

        # 顶部导航布局
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton('上一条')
        self.next_button = QPushButton('下一条')
        self.prev_category_button = QPushButton('上一个分类')
        self.next_category_button = QPushButton('下一个分类')
        self.category_label = QLabel(f'分类：{self.categories[0]}')
        self.category_label.setAlignment(Qt.AlignCenter)
        self.category_label.setFont(QFont('Arial', 14, QFont.Bold))

        nav_layout.addWidget(self.prev_category_button)
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.category_label)
        nav_layout.addWidget(self.next_button)
        nav_layout.addWidget(self.next_category_button)

        main_layout.addLayout(nav_layout)

        # 显示内容区域（可滚动）
        self.content_scroll_area = QScrollArea()
        self.content_scroll_area.setWidgetResizable(True)
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_scroll_area.setWidget(self.content_widget)

        main_layout.addWidget(self.content_scroll_area)

        # 底部状态布局
        status_layout = QHBoxLayout()
        self.status_label = QLabel()
        status_layout.addWidget(self.status_label)
        main_layout.addLayout(status_layout)

        self.setLayout(main_layout)

        # 连接按钮事件
        self.prev_button.clicked.connect(self.prev_item)
        self.next_button.clicked.connect(self.next_item)
        self.prev_category_button.clicked.connect(self.prev_category)
        self.next_category_button.clicked.connect(self.next_category)

    def _clear_layout(self, layout):
        """清空布局中的所有控件（安全的非递归版本）"""
        if layout is not None:
            # 使用 while 循环而不是递归，以避免栈溢出
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    # 如果是子布局，也清空
                    self._clear_layout(item.layout())

    def update_display(self):
        """更新显示当前数据"""
        # 清空旧内容
        self._clear_layout(self.content_layout)

        # 确保分类列表不为空
        if not self.categories:
            self.status_label.setText("未找到任何分类。")
            return

        current_category_name = self.categories[self.current_category_index]

        # --- 关键修改：确保当前分类数据列表不为空 ---
        if not self.current_category_data:
            self.status_label.setText(f"分类 '{current_category_name}' 没有可用的数据。")
            self.category_label.setText(f'分类：{current_category_name}')
            return

        item_data = self.current_category_data[self.current_item_index_in_category]

        # 更新状态栏和分类标签
        total_in_category = len(self.current_category_data)
        self.category_label.setText(f'分类：{current_category_name}')
        self.status_label.setText(
            f'当前项目：{self.current_item_index_in_category + 1} / {total_in_category} (总计 {self.total_item_count} 条)')  # 使用总数

        # 显示 messages
        messages_list = item_data.get('messages', [])

        # 检查 messages_list 是否为列表且不为空
        if messages_list and isinstance(messages_list, list):
            formatted_parts = []

            # 按照 0, 2, 4... 的步长遍历索引
            for i in range(0, len(messages_list), 2):
                # 计算当前的轮次 (第一轮, 第二轮...)
                turn_idx = i // 2 + 1

                # 提取 Question (偶数索引)
                q_content = messages_list[i].get('content', 'N/A')
                formatted_parts.append(f"Question {turn_idx}: {q_content}")

                # 提取 Answer (奇数索引 i+1)
                # 做个安全检查，防止 messages 长度为奇数导致越界
                if i + 1 < len(messages_list):
                    a_content = messages_list[i + 1].get('content', 'N/A')
                    formatted_parts.append(f"Answer {turn_idx}: {a_content}")

                # 在每一轮对话之间加一个分割线，方便阅读
                formatted_parts.append("-" * 30)
            messages_text = "\n".join(formatted_parts)
            # messages_text = f"Question: {messages_list[0]['content']}\n\nAnswer: {messages_list[1]['content']}"
            # for msg_item in messages_list:
            #     # 使用 json.dumps 格式化为可读的字符串
            #     messages_text += json.dumps(msg_item, indent=2,
            #                                 ensure_ascii=False) + "\n\n"

            # 创建标题标签
            messages_label = QLabel("Messages:")
            messages_label.setFont(QFont('Arial', 12, QFont.Bold))
            self.content_layout.addWidget(messages_label)

            # --- 使用 QTextEdit 替代 QLabel ---
            messages_content_text_edit = QTextEdit()

            # 设置固定的高度，例如：300像素
            messages_content_text_edit.setFixedHeight(1000)

            # 设置为只读，这样用户就不能修改文本
            messages_content_text_edit.setReadOnly(True)

            # 设置文本内容
            messages_content_text_edit.setText(messages_text.strip())

            # 设置字体和字号
            # 你可以在这里自定义字体，例如：
            messages_content_text_edit.setFont(QFont('Times New Roman', 16))

            # 添加到布局中
            self.content_layout.addWidget(messages_content_text_edit)
        else:
            self.content_layout.addWidget(
                QLabel("Messages: (No messages found or format is incorrect)"))

        # 显示 images
        # 1. 创建一个新的 QWidget 作为图像容器
        image_container_widget = QWidget()

        # 2. 为图像容器设置 QHBoxLayout（横向布局）
        image_layout = QHBoxLayout(image_container_widget)
        # 可选：设置布局的间距和边距
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(10)  # 图像之间设置 10 像素间距

        # 可选：如果图像很多，限制缩放宽度以确保它们能并排放下
        MAX_IMAGE_WIDTH = 700
        images = item_data.get('images', [])
        for image_path in images:
            if self.image_base_path == "D:\Data\scannetpp\scannetpp_sampled_modified":
                full_path = os.path.join(self.image_base_path, *image_path.split('/')[-3:])
            else:
                full_path = os.path.join(self.image_base_path, *image_path.split('/')[-2:])
            # 创建一个垂直布局容器，用来包裹“路径标签”和“图片”
            single_image_container = QWidget()
            single_image_vbox = QVBoxLayout(single_image_container)
            single_image_vbox.setContentsMargins(0, 0, 0, 0)
            single_image_vbox.setSpacing(5)  # 路径和图片之间的间距

            # 1. 创建路径标签
            path_label = QLabel(f"Path: {image_path}")  # 或者用 full_path 显示绝对路径
            path_label.setStyleSheet("color: gray; font-size: 10pt;")  # 设置样式，让路径不那么突兀
            path_label.setWordWrap(True)  # 如果路径太长，允许换行
            path_label.setMaximumWidth(MAX_IMAGE_WIDTH)  # 限制宽度与图片一致
            # --- 新增代码开始 ---
            # 允许用户通过鼠标选中标签内的文本
            path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            # 将鼠标悬停样式设为“I”型光标，提示用户这是可编辑/选中的文本
            path_label.setCursor(Qt.IBeamCursor)
            # --- 新增代码结束 ---
            single_image_vbox.addWidget(path_label)
            try:
                if os.path.exists(full_path):
                    pixmap = QPixmap(full_path)
                    if not pixmap.isNull():
                        # 2. 创建图片标签
                        image_label = QLabel()
                        image_label.setPixmap(
                            pixmap.scaledToWidth(MAX_IMAGE_WIDTH, Qt.SmoothTransformation))
                        image_label.setAlignment(Qt.AlignCenter)
                        single_image_vbox.addWidget(image_label)
                    else:
                        single_image_vbox.addWidget(QLabel(f"无法加载图像：{image_path}"))
                else:
                    single_image_vbox.addWidget(QLabel(f"图像文件不存在：{full_path}"))
            except Exception as e:
                single_image_vbox.addWidget(QLabel(f"图像处理异常: {e}"))

                # 将这组（路径+图片）添加到横向的 image_layout 中
            image_layout.addWidget(single_image_container)
        # 3. 将图像容器（及其横向布局）添加到主垂直布局中
        self.content_layout.addWidget(image_container_widget)
        self.content_layout.addStretch(1)  # 添加伸展空间，使内容靠上

    def next_item(self):
        """显示当前分类中的下一条数据"""
        if self.current_item_index_in_category < len(
                self.current_category_data) - 1:
            self.current_item_index_in_category += 1
            self.update_display()
        else:
            self.status_label.setText("已到达当前分类的末尾。")

    def prev_item(self):
        """显示当前分类中的上一条数据"""
        if self.current_item_index_in_category > 0:
            self.current_item_index_in_category -= 1
            self.update_display()
        else:
            self.status_label.setText("已到达当前分类的开头。")

    def _load_new_category(self, new_index):
        """加载新的分类数据"""
        new_category_name = self.categories[new_index]
        new_data = self.loader.load_category_data(new_category_name)

        # --- 关键修改：检查新加载的数据是否有效 ---
        if not new_data:
            # 提示用户加载失败，并停留在当前分类
            QMessageBox.warning(self, "加载失败", f"无法加载分类 '{new_category_name}' 的数据，请检查文件内容或格式。")
        else:
            # 成功加载，更新所有状态
            self.current_category_index = new_index
            self.current_category_data = new_data
            self.current_item_index_in_category = 0
            # 必须重新计算总数，因为可能切换前后的总数变了
            self.total_item_count = self._get_total_count()
            self.update_display()

    def next_category(self):
        """切换到下一个分类"""
        if self.current_category_index < len(self.categories) - 1:
            # CORRECT: 调用 _load_new_category
            self._load_new_category(self.current_category_index + 1)
        else:
            self.status_label.setText("已到达最后一个分类。")

    def prev_category(self):
        """切换到上一个分类"""
        if self.current_category_index > 0:
            # CORRECT: 调用 _load_new_category
            self._load_new_category(self.current_category_index - 1)
        else:
            self.status_label.setText("已到达第一个分类。")


def main():
    app = QApplication(sys.argv)

    random.seed(42)

    # 请根据你的实际情况修改以下路径
    # JSON_DIR_PATH = "./QA_jsons_ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage/conversation/level_3"
    # JSON_DIR_PATH = "./QA_jsons_ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage"
    JSON_DIR_PATH = "./api/output"
    # JSON_DIR_PATH = "./ScannetppIphone_MultilevelCategories_20260124_sampled_MCA_Multistage_stage2_gemini-3-flash-preview_CoT_Cleaned_rel_ratio_1.00_no_system"
    # JSON_DIR_PATH = "./QA_jsons_Infinigen_MultilevelCategories_20260121_sampled_MCA_Multistage/conversation/level_3"
    # JSON_DIR_PATH = "./QA_jsons_Infinigen_MultilevelCategories_20260128_sampled_MCA_Multistage/atomic/level_1"
    # JSON_DIR_PATH = "./QA_jsons_Infinigen_MultilevelCategories_20260128_sampled_MCA_Multistage/atomic/level_2"
    # JSON_DIR_PATH = "./QA_jsons_Infinigen_MultilevelCategories_20260128_sampled_MCA_Multistage/atomic/level_3"
    # JSON_DIR_PATH = "./QA_jsons_ScannetppIphone_MultilevelCategories_20260125_sampled_MCA_Multistage/atomic/level_1"
    # JSON_DIR_PATH = "./QA_jsons_ScannetppIphone_MultilevelCategories_20260125_sampled_MCA_Multistage/atomic/level_2"
    # JSON_DIR_PATH = "./QA_jsons_ScannetppIphone_MultilevelCategories_20260125_sampled_MCA_Multistage/atomic/level_3"
    # JSON_DIR_PATH = "./QA_jsons_ScannetppIphone_MultilevelCategories_20260313_sampled_MCA_Multistage/atomic/level_2"
    # JSON_DIR_PATH = "./QA_jsons_Infinigen_MultilevelCategories_20260313_sampled_MCA_Multistage/atomic/level_2"
    # IMAGE_BASE_PATH = "./infinigen_metadata_ver2/saved_scenes"
    IMAGE_BASE_PATH = "D:\Data\scannetpp\scannetpp_sampled_modified"

    # 初始化加载器
    loader = CategoryLoader(JSON_DIR_PATH)

    if not loader.categories:
        print(f"错误：未在目录 {JSON_DIR_PATH} 中找到任何分类 JSON 文件。")
        sys.exit(1)

    # 检查图片目录是否存在（可选）
    if not os.path.exists(IMAGE_BASE_PATH):
        print(f"警告：图像基础目录不存在 -> {IMAGE_BASE_PATH}")
        # sys.exit(1) # 不退出，但图片会显示加载失败

    viewer = DataViewer(loader, IMAGE_BASE_PATH)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
