import streamlit as st
import json
import os
from pathlib import Path

# 配置页面
st.set_page_config(page_title="MLLM 多图序列质检工具", layout="wide")

# 自定义 CSS：处理滚动容器、大字号文本和布局
st.markdown("""
    <style>
    /* 强制图片水平排列的滚动容器 */
    .scroll-container {
        display: flex;
        overflow-x: auto;
        gap: 15px;
        padding: 10px;
    }
    /* 增大对话框字号 */
    .stAlert p {
        font-size: 1.2rem !important;
        line-height: 1.6 !important;
    }
    /* 角色标签字号 */
    .role-label {
        font-size: 1.3rem;
        font-weight: bold;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    st.title("🖼️ MLLM 空间序列数据质检")
    data_path = Path("./qc_task_v1")

    # 图像固定宽度设置
    FIXED_WIDTH = 450

    # 1. 加载数据
    json_path = data_path / 'qc_samples_100.json'
    if not os.path.exists(json_path):
        st.error(f"未找到数据文件: {json_path}")
        return

    if 'data' not in st.session_state:
        with open(json_path, 'r', encoding='utf-8') as f:
            st.session_state.data = json.load(f)
        st.session_state.results = {}

    data = st.session_state.data
    total_count = len(data)

    if 'idx' not in st.session_state:
        st.session_state.idx = 0

    # 2. 侧边栏：详尽统计
    st.sidebar.header("📊 质检进度统计")
    reviewed_ids = list(st.session_state.results.keys())
    reviewed_count = len(reviewed_ids)

    # 计算正确率
    correct_count = sum(1 for v in st.session_state.results.values() if v is True)
    wrong_count = sum(1 for v in st.session_state.results.values() if v is False)

    st.sidebar.metric("总样本数", total_count)
    st.sidebar.metric("已审核", f"{reviewed_count} ({int(reviewed_count / total_count * 100)}%)")

    col_s1, col_s2 = st.sidebar.columns(2)
    col_s1.success(f"正确: {correct_count}")
    col_s2.error(f"错误: {wrong_count}")

    if reviewed_count > 0:
        acc = (correct_count / reviewed_count) * 100
        st.sidebar.info(f"当前准确率: {acc:.1f}%")
        st.sidebar.progress(reviewed_count / total_count)

    # 3. 顶部导航栏
    col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
    with col_nav1:
        if st.button("⬅️ 上一条 (Previous)") and st.session_state.idx > 0:
            st.session_state.idx -= 1
            st.rerun()
    with col_nav2:
        st.markdown(f"<h3 style='text-align: center;'>第 {st.session_state.idx + 1} / {total_count} 条</h3>", unsafe_allow_html=True)
    with col_nav3:
        if st.button("下一条 (Next) ➡️") and st.session_state.idx < total_count - 1:
            st.session_state.idx += 1
            st.rerun()

    st.divider()

    # 4. 核心展示区
    current_item = data[st.session_state.idx]
    img_list = current_item.get("images", [])

    # --- 图像显示区 ---
    st.markdown("#### 🖼️ 图像序列")
    if img_list:
        # 使用 columns 模拟水平平铺
        cols = st.columns(len(img_list))
        for i, img_rel_path in enumerate(img_list):
            with cols[i]:
                full_img_path = data_path / img_rel_path
                if full_img_path.exists():
                    st.image(str(full_img_path), caption=f"图像 {i+1}", width=FIXED_WIDTH)
                else:
                    st.error(f"图片不存在: {img_rel_path}")
    else:
        st.warning("该条目不包含任何图像。")

    st.divider()

    # --- 对话与判定区 ---
    col_text, col_qc = st.columns([2, 1])

    with col_text:
        st.markdown(f"**类别:** `{current_item['category']}` | **层级:** `{current_item['level']}` | **场景:** `{current_item['scene_name']}`")

        for msg in current_item["messages"]:
            if msg["role"] == "user":
                st.markdown('<p class="role-label">🧑‍💻 问题 (Question):</p>', unsafe_allow_html=True)
                st.info(msg["content"])
            else:
                st.markdown('<p class="role-label">🤖 答案 (Answer):</p>', unsafe_allow_html=True)
                st.success(msg["content"])

    with col_qc:
        st.markdown("#### ⚖️ 人工质检判定")
        current_status = st.session_state.results.get(st.session_state.idx)

        c1, c2 = st.columns(2)
        # 点击后自动跳转下一条
        if c1.button("✅ 正确 (Correct)", use_container_width=True):
            st.session_state.results[st.session_state.idx] = True
            if st.session_state.idx < total_count - 1:
                st.session_state.idx += 1
                st.rerun()

        if c2.button("❌ 错误 (Wrong)", use_container_width=True):
            st.session_state.results[st.session_state.idx] = False
            if st.session_state.idx < total_count - 1:
                st.session_state.idx += 1
                st.rerun()

        if current_status is not None:
            if current_status:
                st.write("✨ 当前标记状态：**正确**")
            else:
                st.write("⚠️ 当前标记状态：**错误**")

    # 5. 导出逻辑
    st.sidebar.divider()
    if st.sidebar.button("💾 导出最终质检 JSON", use_container_width=True):
        output_results = []
        for i, item in enumerate(data):
            res_item = item.copy()
            # 获取质检结果，如果没检则标为未审核
            res_item["qc_status"] = st.session_state.results.get(i, "Unreviewed")
            output_results.append(res_item)

        save_path = data_path / "qc_final_results.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output_results, f, indent=4, ensure_ascii=False)

        st.sidebar.success(f"文件已保存至: {save_path}")
        st.balloons()


if __name__ == "__main__":
    main()
