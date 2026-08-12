import os
import json
import streamlit as st
import pandas as pd
from pathlib import Path

# ==================== 配置区域 ====================
INPUT_JSON_PATH = "./human_eval/human_eval_sampled_multiview_200.json"
OUTPUT_JSON_PATH = "./human_eval/human_eval_verification_results.json"
# ==================================================

# 1. 页面基本配置
st.set_page_config(page_title="3D VLM Human Verification Tool", layout="wide")

# 自定义 CSS：微动动效、按钮颜色与排版优化
st.markdown("""
    <style>
    /* 调整卡片和标签字号 */
    .stAlert p {
        font-size: 1.1rem !important;
        line-height: 1.5 !important;
    }
    .meta-badge {
        background-color: #f0f2f6;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.9rem;
        margin-right: 10px;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)


# 2. 图像路径转换函数
def process_image_path(raw_path):
    if 'Infinigen_MMSIBench_ver2' in raw_path:
        prefix = "/cache/xj/data/Infinigen_MMSIBench_ver2"
        if raw_path.startswith(prefix):
            cleaned_path = raw_path[len(prefix):].lstrip("/")
        else:
            cleaned_path = raw_path.lstrip("/")

        # 拼装新路径 (注意：Streamlit 运行路径可能不同，建议配合 Path 保证鲁棒性)
        new_path = Path("../infinigen_metadata_ver2") / cleaned_path
    else:
        prefix = "/cache/xj/data"
        if raw_path.startswith(prefix):
            cleaned_path = raw_path[len(prefix):].lstrip("/")
        else:
            cleaned_path = raw_path.lstrip("/")

        # 拼装新路径 (注意：Streamlit 运行路径可能不同，建议配合 Path 保证鲁棒性)
        new_path = Path("D:/Data/scannetpp") / cleaned_path
    return new_path


# 3. 初始化 Session State (状态保持)
if 'dataset' not in st.session_state:
    # 优先读取已有的标注进度文件
    if os.path.exists(OUTPUT_JSON_PATH):
        with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            st.session_state.dataset = json.load(f)
        st.toast("📂 已加载历史标注进度！", icon="ℹ️")
    elif os.path.exists(INPUT_JSON_PATH):
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            st.session_state.dataset = json.load(f)
            # 初始化标注字段
            for item in st.session_state.dataset:
                if "verification_status" not in item:
                    item["verification_status"] = "未标注"
    else:
        st.error(f"未找到输入数据，请检查路径：{INPUT_JSON_PATH}")
        st.stop()

if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0

# 简化变量名
dataset = st.session_state.dataset
total_count = len(dataset)
current_idx = st.session_state.current_idx

# 4. 侧边栏：实时统计面板 (无需手动点击刷新)
st.sidebar.header("📊 实时验证统计看板")

# 数据提取与转换
status_list = [item.get("verification_status", "未标注") for item in dataset]
df = pd.DataFrame(status_list, columns=["Status"])
counts = df["Status"].value_counts().to_dict()

# 补齐所有类别
all_categories = ["正确", "Visual validity 错误", "Answer correctness 错误", "Language clarity 错误", "Cross-view dependency 错误", "未标注"]
for cat in all_categories:
    counts.setdefault(cat, 0)

annotated = total_count - counts["未标注"]
accuracy = (counts["正确"] / annotated * 100) if annotated > 0 else 0.0

# 渲染指标仪表盘
st.sidebar.metric("已标注进度", f"{annotated} / {total_count}", f"{annotated / total_count * 100:.1f}%")
st.sidebar.metric("整体准确率 (Accuracy)", f"{accuracy:.1f}%")
st.sidebar.progress(annotated / total_count)

st.sidebar.subheader("❌ 错误类型明细")
st.sidebar.markdown(f"""
* **Visual validity 错误**: {counts["Visual validity 错误"]} 件 ({(counts["Visual validity 错误"] / max(1, annotated) * 100):.1f}%)
* **Answer correctness 错误**: {counts["Answer correctness 错误"]} 件 ({(counts["Answer correctness 错误"] / max(1, annotated) * 100):.1f}%)
* **Language clarity 错误**: {counts["Language clarity 错误"]} 件 ({(counts["Language clarity 错误"] / max(1, annotated) * 100):.1f}%)
* **Cross-view dependency 错误**: {counts["Cross-view dependency 错误"]} 件 ({(counts["Cross-view dependency 错误"] / max(1, annotated) * 100):.1f}%)
""")

# 5. 主页面布局
st.title("🧪 3D VLM 数据集多视角一致性人工验证工具")
st.caption("按照审稿人意见进行 200 条采样多维校验。当所有指标全部通过时判定为【正确】，否则记录对应的错误类型。")

# 顶部导航控制条
col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
with col_nav1:
    if st.button("◀ 上一条 (Previous)", use_container_width=True) and current_idx > 0:
        st.session_state.current_idx -= 1
        st.rerun()
with col_nav2:
    st.markdown(f"<h3 style='text-align: center;'>进度: {current_idx + 1} / {total_count}</h3>", unsafe_allow_html=True)
with col_nav3:
    if st.button("下一条 (Next) ▶", use_container_width=True) and current_idx < total_count - 1:
        st.session_state.current_idx += 1
        st.rerun()

st.divider()

# 获取当前展示条目
current_item = dataset[current_idx]

# 左右分栏：左侧展示数据，右侧进行标注
col_left, col_right = st.columns([3, 1.2])

with col_left:
    # 5.1 显示元数据
    scene_name = current_item.get('scene_name')
    category = current_item.get('category')
    source = current_item.get('data_source')

    st.markdown(
        f'<span class="meta-badge"><b>Scene ID:</b> {scene_name}</span>'
        f'<span class="meta-badge"><b>Category:</b> {category}</span>'
        f'<span class="meta-badge"><b>Source:</b> {source}</span>',
        unsafe_allow_html=True
    )
    st.write("")  # 留空

    # 5.2 对话框数据解析与展示
    for msg in current_item.get("messages", []):
        if msg["role"] == "user":
            st.markdown("##### 👤 Question (User)")
            cleaned_question = msg["content"].replace("<image>", "").strip()
            st.info(cleaned_question)
        elif msg["role"] == "assistant":
            st.markdown("##### 🤖 Answer (Assistant)")
            st.success(msg["content"].strip())

    st.divider()

    # 5.3 图像展示区
    st.markdown("##### 🖼️ Scene View Image")
    raw_images = current_item.get("images", [])
    if raw_images:
        processed_images = [process_image_path(img) for img in raw_images]
        # 如果有多张图，自适应分栏展示
        img_cols = st.columns(len(processed_images))
        for idx, img_path in enumerate(processed_images):
            with img_cols[idx]:
                if img_path.exists():
                    st.image(str(img_path), caption=f"View {idx + 1}", use_container_width=True)
                else:
                    st.error(f"图片不存在: {img_path}")
    else:
        st.warning("该样本没有关联图像。")

with col_right:
    # 5.4 人工判定面板
    st.markdown("### 📝 人工多维检验面板")

    # 获取当前已保存的状态
    current_status = current_item.get("verification_status", "未标注")
    st.markdown(f"当前标记状态：`{current_status}`")

    st.write("---")

    # 重点：利用 Streamlit 的按钮矩阵实现“点击即保存并自动跳转下一条”
    # 这样能极大减轻标注员的鼠标点击负担
    options = [
        ("🟢 校验完全正确 (Pass)", "正确"),
        ("❌ Visual validity 错误", "Visual validity 错误"),
        ("❌ Answer correctness 错误", "Answer correctness 错误"),
        ("❌ Language clarity 错误", "Language clarity 错误"),
        ("❌ Cross-view dependency 错误", "Cross-view dependency 错误")
    ]

    for label, value in options:
        # 使用 unique key 避免 Streamlit 组件 key 冲突
        if st.button(label, key=f"btn_{value}_{current_idx}", use_container_width=True):
            # 1. 存入内存
            st.session_state.dataset[current_idx]["verification_status"] = value
            # 2. 自动保存到本地文件，防止意外掉电/程序关闭
            with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                json.dump(st.session_state.dataset, f, indent=4, ensure_ascii=False)

            # 3. 自动跳转下一页 (如果未到最后)
            if current_idx < total_count - 1:
                st.session_state.current_idx += 1
            st.rerun()

    st.write("---")

    # 5.5 侧栏手动导出按钮
    # 修改为（去掉 variant 参数，只保留 use_container_width）：
    if st.button("💾 导出最终质检 JSON (手动)", use_container_width=True):
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)
        st.success(f"导出成功！已保存至 {OUTPUT_JSON_PATH}")
        st.balloons()
