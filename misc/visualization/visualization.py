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
# Added: category data loader
# ----------------------------------------------------------------------
class CategoryLoader:
    """Discover and lazily load JSON files of different categories."""

    def __init__(self, json_dir_path: str):
        self.json_dir = Path(json_dir_path)
        self.categories: List[str] = []
        self._data_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.find_categories()

    def find_categories(self):
        """Scan the directory, find all .json files, and use the file name (without extension) as the category."""
        if not self.json_dir.is_dir():
            print(f"错误：JSON 目录不存在或不是目录 -> {self.json_dir}")
            return

        json_files = sorted(self.json_dir.glob("*.json"))
        # Use the file name (without extension) as the category name
        self.categories = [f.stem for f in json_files]
        if not self.categories:
            print(f"警告：未在 {self.json_dir} 中找到任何 JSON 文件。")

    def load_category_data(self, category_name: str) -> List[Dict[str, Any]]:
        MAX_ITEMS_TO_LOAD = 200  # <--- set a maximum number of items to load
        """Load the data for the specified category from cache or file."""
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
                # --- limit the number of items to load ---
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
    # Replace data_list with the loader object
    def __init__(self, category_loader: CategoryLoader, image_base_path: str):
        super().__init__()
        self.loader = category_loader
        self.image_base_path = image_base_path

        # --- set the window size here ---
        INITIAL_WIDTH = 2400
        INITIAL_HEIGHT = 1200

        # Use resize to set the initial size, allowing the user to adjust it
        self.resize(INITIAL_WIDTH, INITIAL_HEIGHT)

        # Get the category list directly from the loader
        self.categories = self.loader.categories
        self.current_category_index = 0

        # Initially load the data of the first category
        if self.categories:
            first_category = self.categories[0]
            self.current_category_data = self.loader.load_category_data(first_category)
        else:
            self.current_category_data = []

        self.current_item_index_in_category = 0
        self.total_item_count = self._get_total_count()  # newly added total count calculation

        self.initUI()
        self.update_display()

    def _get_total_count(self):
        """Compute the total number of data items across all loaded categories."""
        total = 0
        for data_list in self.loader._data_cache.values():
            total += len(data_list)
        return total

    def initUI(self):
        """Initialize the user interface."""
        self.setWindowTitle('JSON数据可视化工具')
        self.setGeometry(100, 100, 2400, 1900)

        # Main vertical layout
        main_layout = QVBoxLayout()

        # Top navigation layout
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

        # Display content area (scrollable)
        self.content_scroll_area = QScrollArea()
        self.content_scroll_area.setWidgetResizable(True)
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_scroll_area.setWidget(self.content_widget)

        main_layout.addWidget(self.content_scroll_area)

        # Bottom status layout
        status_layout = QHBoxLayout()
        self.status_label = QLabel()
        status_layout.addWidget(self.status_label)
        main_layout.addLayout(status_layout)

        self.setLayout(main_layout)

        # Connect button events
        self.prev_button.clicked.connect(self.prev_item)
        self.next_button.clicked.connect(self.next_item)
        self.prev_category_button.clicked.connect(self.prev_category)
        self.next_category_button.clicked.connect(self.next_category)

    def _clear_layout(self, layout):
        """Clear all widgets in the layout (safe non-recursive version)."""
        if layout is not None:
            # Use a while loop instead of recursion to avoid stack overflow
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    # Also clear it if it is a sub-layout
                    self._clear_layout(item.layout())

    def update_display(self):
        """Update and display the current data."""
        # Clear old content
        self._clear_layout(self.content_layout)

        # Ensure the category list is not empty
        if not self.categories:
            self.status_label.setText("未找到任何分类。")
            return

        current_category_name = self.categories[self.current_category_index]

        # --- Key change: ensure the current category data list is not empty ---
        if not self.current_category_data:
            self.status_label.setText(f"分类 '{current_category_name}' 没有可用的数据。")
            self.category_label.setText(f'分类：{current_category_name}')
            return

        item_data = self.current_category_data[self.current_item_index_in_category]

        # Update the status bar and category label
        total_in_category = len(self.current_category_data)
        self.category_label.setText(f'分类：{current_category_name}')
        self.status_label.setText(
            f'当前项目：{self.current_item_index_in_category + 1} / {total_in_category} (总计 {self.total_item_count} 条)')  # using total count

        # Display messages
        messages_list = item_data.get('messages', [])

        # Check whether messages_list is a list and not empty
        if messages_list and isinstance(messages_list, list):
            formatted_parts = []

            # Iterate over indices in steps of 0, 2, 4...
            for i in range(0, len(messages_list), 2):
                # Compute the current turn (first turn, second turn...)
                turn_idx = i // 2 + 1

                # Extract Question (even index)
                q_content = messages_list[i].get('content', 'N/A')
                formatted_parts.append(f"Question {turn_idx}: {q_content}")

                # Extract Answer (odd index i+1)
                # Perform a safety check to prevent out-of-bounds when the messages length is odd
                if i + 1 < len(messages_list):
                    a_content = messages_list[i + 1].get('content', 'N/A')
                    formatted_parts.append(f"Answer {turn_idx}: {a_content}")

                # Add a separator line between turns for readability
                formatted_parts.append("-" * 30)
            messages_text = "\n".join(formatted_parts)
            # messages_text = f"Question: {messages_list[0]['content']}\n\nAnswer: {messages_list[1]['content']}"
            # for msg_item in messages_list:
            #     # use json.dumps to format into a readable string
            #     messages_text += json.dumps(msg_item, indent=2,
            #                                 ensure_ascii=False) + "\n\n"

            # Create a title label
            messages_label = QLabel("Messages:")
            messages_label.setFont(QFont('Arial', 12, QFont.Bold))
            self.content_layout.addWidget(messages_label)

            # --- Use QTextEdit instead of QLabel ---
            messages_content_text_edit = QTextEdit()

            # Set a fixed height, e.g., 300 pixels
            messages_content_text_edit.setFixedHeight(1000)

            # Set it to read-only so the user cannot modify the text
            messages_content_text_edit.setReadOnly(True)

            # Set text content
            messages_content_text_edit.setText(messages_text.strip())

            # Set the font and font size
            # You can customize the font here, e.g.:
            messages_content_text_edit.setFont(QFont('Times New Roman', 16))

            # Add to the layout
            self.content_layout.addWidget(messages_content_text_edit)
        else:
            self.content_layout.addWidget(
                QLabel("Messages: (No messages found or format is incorrect)"))

        # Display images
        # 1. Create a new QWidget as the image container
        image_container_widget = QWidget()

        # 2. Set a QHBoxLayout (horizontal layout) for the image container
        image_layout = QHBoxLayout(image_container_widget)
        # Optional: set the layout spacing and margins
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(10)  # set 10 pixel spacing between images

        # Optional: if there are many images, limit the scaled width so they fit side by side
        MAX_IMAGE_WIDTH = 700
        images = item_data.get('images', [])
        for image_path in images:
            if self.image_base_path == "/path/to/data/scannetpp/scannetpp_sampled_modified":
                full_path = os.path.join(self.image_base_path, *image_path.split('/')[-3:])
            else:
                full_path = os.path.join(self.image_base_path, *image_path.split('/')[-2:])
            # Create a vertical layout container wrapping the path label and image
            single_image_container = QWidget()
            single_image_vbox = QVBoxLayout(single_image_container)
            single_image_vbox.setContentsMargins(0, 0, 0, 0)
            single_image_vbox.setSpacing(5)  # spacing between path and image

            # 1. Create the path label
            path_label = QLabel(f"Path: {image_path}")  # or use full_path to display the absolute path
            path_label.setStyleSheet("color: gray; font-size: 10pt;")  # set a style so the path is less obtrusive
            path_label.setWordWrap(True)  # allow wrapping if the path is too long
            path_label.setMaximumWidth(MAX_IMAGE_WIDTH)  # limit the width to match the image
            # --- New code start ---
            # Allow the user to select the text inside the label with the mouse
            path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            # Set the mouse hover style to an I-beam cursor, indicating the text is selectable
            path_label.setCursor(Qt.IBeamCursor)
            # --- New code end ---
            single_image_vbox.addWidget(path_label)
            try:
                if os.path.exists(full_path):
                    pixmap = QPixmap(full_path)
                    if not pixmap.isNull():
                        # 2. Create the image label
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

                # Add this group (path + image) to the horizontal image_layout
            image_layout.addWidget(single_image_container)
        # 3. Add the image container (and its horizontal layout) to the main vertical layout
        self.content_layout.addWidget(image_container_widget)
        self.content_layout.addStretch(1)  # add stretch so the content is pushed to the top

    def next_item(self):
        """Show the next data item in the current category."""
        if self.current_item_index_in_category < len(
                self.current_category_data) - 1:
            self.current_item_index_in_category += 1
            self.update_display()
        else:
            self.status_label.setText("已到达当前分类的末尾。")

    def prev_item(self):
        """Show the previous data item in the current category."""
        if self.current_item_index_in_category > 0:
            self.current_item_index_in_category -= 1
            self.update_display()
        else:
            self.status_label.setText("已到达当前分类的开头。")

    def _load_new_category(self, new_index):
        """Load the data of a new category."""
        new_category_name = self.categories[new_index]
        new_data = self.loader.load_category_data(new_category_name)

        # --- Key change: check whether the newly loaded data is valid ---
        if not new_data:
            # Notify the user of the load failure and stay on the current category
            QMessageBox.warning(self, "加载失败", f"无法加载分类 '{new_category_name}' 的数据，请检查文件内容或格式。")
        else:
            # Load succeeded, update all state
            self.current_category_index = new_index
            self.current_category_data = new_data
            self.current_item_index_in_category = 0
            # Must recompute the total count because it may have changed
            self.total_item_count = self._get_total_count()
            self.update_display()

    def next_category(self):
        """Switch to the next category."""
        if self.current_category_index < len(self.categories) - 1:
            # CORRECT: call _load_new_category
            self._load_new_category(self.current_category_index + 1)
        else:
            self.status_label.setText("已到达最后一个分类。")

    def prev_category(self):
        """Switch to the previous category."""
        if self.current_category_index > 0:
            # CORRECT: call _load_new_category
            self._load_new_category(self.current_category_index - 1)
        else:
            self.status_label.setText("已到达第一个分类。")


def main():
    app = QApplication(sys.argv)

    random.seed(42)

    # Please adjust the following paths to match your environment
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
    IMAGE_BASE_PATH = "/path/to/data/scannetpp/scannetpp_sampled_modified"

    # Initialize the loader
    loader = CategoryLoader(JSON_DIR_PATH)

    if not loader.categories:
        print(f"错误：未在目录 {JSON_DIR_PATH} 中找到任何分类 JSON 文件。")
        sys.exit(1)

    # Check whether the image directory exists (optional)
    if not os.path.exists(IMAGE_BASE_PATH):
        print(f"警告：图像基础目录不存在 -> {IMAGE_BASE_PATH}")
        # sys.exit(1) # do not exit, but images will fail to load

    viewer = DataViewer(loader, IMAGE_BASE_PATH)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
