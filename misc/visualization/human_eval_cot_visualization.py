import os
import json
import streamlit as st
import pandas as pd
from pathlib import Path

# ==================== Config ====================
# 与 mvstride/evaluation/human_eval_cot_sampling.py 的输出文件名保持一致
INPUT_JSON_PATH = "./human_eval/cot_with_original_aligned_sampled_200.json"
OUTPUT_JSON_PATH = "./human_eval/human_eval_cot_verification_results.json"
# ==================================================

# 1. Basic page config
st.set_page_config(page_title="3D VLM CoT Human Verification Tool", layout="wide")

# Custom CSS: subtle motion, button colors, and layout polish.
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


# 2. Image path conversion (keeps the original mapping logic).
def process_image_path(raw_path):
    """把 json 里的图像路径映射回本机真实文件：先按原样尝试，再按仓库 data/ 布局回退。"""
    candidates = [Path(raw_path)]
    for marker, local_root in (
        ("saved_scenes", "data/infinigen/saved_scenes"),
        ("scannetpp_sampled_modified", "data/scannetpp/scannetpp_sampled_modified"),
    ):
        if marker in raw_path:
            tail = raw_path.split(marker, 1)[1].lstrip("/\\")
            candidates.append(Path(local_root) / tail)
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


# 3. Initialize session state.
if 'dataset' not in st.session_state:
    # Try to load existing annotation progress first.
    if os.path.exists(OUTPUT_JSON_PATH):
        with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            st.session_state.dataset = json.load(f)
        st.toast("📂 已加载历史标注进度！", icon="ℹ️")
    elif os.path.exists(INPUT_JSON_PATH):
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            st.session_state.dataset = json.load(f)
            # Initialize the annotation and notes fields.
            for item in st.session_state.dataset:
                if "verification_status" not in item:
                    item["verification_status"] = "未标注"
                if "verification_notes" not in item:
                    item["verification_notes"] = ""
    else:
        st.error(f"未找到输入数据，请检查路径：{INPUT_JSON_PATH}")
        st.stop()

if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0

# Convenient short aliases.
dataset = st.session_state.dataset
total_count = len(dataset)
current_idx = st.session_state.current_idx

# 4. Sidebar: live statistics panel.
st.sidebar.header("📊 CoT 验证实时看板")

# Extract and aggregate the status values.
status_list = [item.get("verification_status", "未标注") for item in dataset]
df = pd.DataFrame(status_list, columns=["Status"])
counts = df["Status"].value_counts().to_dict()

# Ensure all verdict categories are present.
all_categories = [
    "正确 (Pass)",
    "Factual consistency 错误",
    "Reasoning faithfulness 错误",
    "Final-answer consistency 错误",
    "Hallucination 错误",
    "未标注"
]
for cat in all_categories:
    counts.setdefault(cat, 0)

annotated = total_count - counts["未标注"]
accuracy = (counts["正确 (Pass)"] / annotated * 100) if annotated > 0 else 0.0

# Render the metrics dashboard.
st.sidebar.metric("已标注进度", f"{annotated} / {total_count}", f"{annotated / total_count * 100:.1f}%")
st.sidebar.metric("CoT 完美正确率 (Accuracy)", f"{accuracy:.1f}%")
st.sidebar.progress(annotated / total_count)

st.sidebar.subheader("❌ 细分错误类型统计")
st.sidebar.markdown(f"""
* **Factual consistency 错误**: {counts["Factual consistency 错误"]} 件 ({(counts["Factual consistency 错误"] / max(1, annotated) * 100):.1f}%)
  <small style='color: gray;'>（物体/方向/Bbox/角度与 GT 不符）</small>
* **Reasoning faithfulness 错误**: {counts["Reasoning faithfulness 错误"]} 件 ({(counts["Reasoning faithfulness 错误"] / max(1, annotated) * 100):.1f}%)
  <small style='color: gray;'>（未基于给定子问题，凭空捏造事实）</small>
* **Final-answer consistency 错误**: {counts["Final-answer consistency 错误"]} 件 ({(counts["Final-answer consistency 错误"] / max(1, annotated) * 100):.1f}%)
  <small style='color: gray;'>（最终推理结果与 Label 答案不一致）</small>
* **Hallucination 错误**: {counts["Hallucination 错误"]} 件 ({(counts["Hallucination 错误"] / max(1, annotated) * 100):.1f}%)
  <small style='color: gray;'>（出现不存在的物体、错误视角、错误空间关系）</small>
""", unsafe_allow_html=True)

# 5. Main page layout.
st.title("🧪 多视角空间推理 CoT 人工验证与对齐工具")
st.caption("针对审稿人对于 CoT 数据质量的质疑，进行 200 条多视角空间推理数据的精细化评估。")

# Top navigation bar.
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

# Get the item currently being shown.
current_item = dataset[current_idx]

# Two-column layout: data on the left, annotation on the right.
col_left, col_right = st.columns([3, 1.2])

with col_left:
    # 5.1 Show metadata.
    scene_name = current_item.get('scene_name')
    category = current_item.get('category')
    source = current_item.get('data_source')

    st.markdown(
        f'<span class="meta-badge"><b>Scene ID:</b> {scene_name}</span>'
        f'<span class="meta-badge"><b>Category:</b> {category}</span>'
        f'<span class="meta-badge"><b>Source:</b> {source}</span>',
        unsafe_allow_html=True
    )
    st.write("")  # spacer

    # 5.2 Parse and show the dialogue (mainly the CoT reasoning).
    for msg in current_item.get("messages", []):
        if msg["role"] == "user":
            st.markdown("##### 👤 Question & Input Sub-questions (User)")
            cleaned_question = msg["content"].replace("<image>", "").strip()
            st.info(cleaned_question)
        elif msg["role"] == "assistant":
            st.markdown("##### 🤖 Generated CoT & Answer (Assistant)")
            st.success(msg["content"].strip())

    st.divider()

    # 5.3 Image display area.
    st.markdown("##### 🖼️ Scene View Image (多视角参考图)")
    raw_images = current_item.get("images", [])
    if raw_images:
        processed_images = [process_image_path(img) for img in raw_images]
        # Show multiple views in auto-sized columns.
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
    # 5.4 Human verdict panel.
    st.markdown("### 📝 CoT 多维质检面板")

    # Show the current status.
    current_status = current_item.get("verification_status", "未标注")
    current_notes = current_item.get("verification_notes", "")

    st.markdown(f"当前标记：`{current_status}`")
    if current_notes:
        st.info(f"历史备注：{current_notes}")

    st.write("---")

    # Notes input box for later organizing error cases.
    note_input = st.text_input(
        "📝 错误备注 (选填，如：'把 west 错写成 east')",
        value=current_notes,
        key=f"note_{current_idx}"
    )

    st.write("---")

    # Verdict buttons matching the paper design.
    options = [
        ("🟢 校验完全正确 (Pass)", "正确 (Pass)"),
        ("❌ Factual consistency 错误", "Factual consistency 错误"),
        ("❌ Reasoning faithfulness 错误", "Reasoning faithfulness 错误"),
        ("❌ Final-answer consistency 错误", "Final-answer consistency 错误"),
        ("❌ Hallucination 错误", "Hallucination 错误")
    ]

    for label, value in options:
        # Unique key avoids Streamlit component key conflicts.
        if st.button(label, key=f"btn_{value}_{current_idx}", use_container_width=True):
            # 1. Store in memory (status + notes).
            st.session_state.dataset[current_idx]["verification_status"] = value
            st.session_state.dataset[current_idx]["verification_notes"] = note_input

            # 2. Autosave to a local file to prevent losing progress.
            with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                json.dump(st.session_state.dataset, f, indent=4, ensure_ascii=False)

            # 3. Auto-advance to the next item (unless at the end).
            if current_idx < total_count - 1:
                st.session_state.current_idx += 1
            st.rerun()

    st.write("---")

    # 5.5 Sidebar manual export button.
    if st.button("💾 导出最终质检 JSON (手动)", use_container_width=True):
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)
        st.success(f"导出成功！已保存至 {OUTPUT_JSON_PATH}")
        st.balloons()
