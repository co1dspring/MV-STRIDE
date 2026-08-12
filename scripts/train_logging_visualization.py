import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os


def plot_training_logs(jsonl_path):
    # 1. 加载数据
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(data)

    print("\n" + "=" * 50)
    print(f"{'Step Range':<20} | {'Avg Mean Length':<15}")
    print("-" * 50)

    if 'completions/mean_length' in df.columns:
        # 创建分组标识，每 50 行一组
        df['step_group'] = (df.index // 50) * 50
        group_stats = df.groupby('step_group')['completions/mean_length'].mean()

        for step, avg_len in group_stats.items():
            range_str = f"{step}-{step + 49}"
            print(f"{range_str:<20} | {avg_len:>15.2f}")
    else:
        print("警告: 未在日志中发现 'completions/mean_length' 列")
    print("=" * 50 + "\n")

    # 转换进度指标
    if 'percentage' in df.columns:
        df['percent_num'] = df['percentage'].astype(str).str.replace('%', '').astype(float)
    else:
        df['percent_num'] = df.index

    # 2. 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle(f'GRPO Training Monitor (Standard): {os.path.basename(jsonl_path)}', fontsize=20, fontweight='bold')

    # --- 图 1: Rewards (Mean Values) ---
    ax1 = axes[0, 0]
    ax1.plot(df['percent_num'], df['reward'], label='Total Reward', color='blue', linewidth=2)
    reward_keys = {
        'rewards/MultiModalAccuracyORM/mean': ('Acc Mean', 'green'),
        'rewards/Format/mean': ('Format Mean', 'gray'),
        'rewards/RepetitionPenalty/mean': ('Repet Mean', 'red')
    }
    for key, (label, col) in reward_keys.items():
        if key in df.columns:
            ax1.plot(df['percent_num'], df[key], label=label, alpha=0.6, linestyle='--', color=col)
    ax1.set_title('Reward Mean Trends', fontsize=14)
    ax1.set_xlabel('Progress (%)')
    ax1.legend(loc='upper left', fontsize='small')

    # --- 图 2: Loss & KL Divergence ---
    ax2 = axes[0, 1]
    ax2.plot(df['percent_num'], df['loss'], label='Total Loss', color='firebrick', linewidth=1.5)
    ax2.set_ylabel('Loss', color='firebrick')
    if 'kl' in df.columns:
        ax2_twin = ax2.twinx()
        ax2_twin.plot(df['percent_num'], df['kl'], label='KL', color='forestgreen', alpha=0.5)
        ax2_twin.set_ylabel('KL', color='forestgreen')
    ax2.set_title('Loss & KL Divergence', fontsize=14)
    ax2.set_xlabel('Progress (%)')

    # --- 图 3: Completion Length ---
    ax3 = axes[0, 2]
    if all(k in df.columns for k in ['completions/min_length', 'completions/max_length']):
        ax3.fill_between(df['percent_num'], df['completions/min_length'],
                         df['completions/max_length'], alpha=0.15, color='orange')
    if 'completions/mean_length' in df.columns:
        ax3.plot(df['percent_num'], df['completions/mean_length'], label='Mean Len', color='darkorange', linewidth=2)
    ax3.set_title('Response Length Dynamics', fontsize=14)
    ax3.set_xlabel('Progress (%)')
    ax3.legend()

    # --- 图 4: LR & Grad Norm (关键更新) ---
    ax4 = axes[1, 0]
    if 'learning_rate' in df.columns:
        ax4.plot(df['percent_num'], df['learning_rate'], color='darkviolet', label='LR', linewidth=2)
        ax4.set_ylabel('Learning Rate', color='darkviolet')
        ax4.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    if 'grad_norm' in df.columns:
        ax4_twin = ax4.twinx()
        ax4_twin.plot(df['percent_num'], df['grad_norm'], color='black', label='Grad Norm', alpha=0.4, linestyle=':')
        ax4_twin.set_ylabel('Grad Norm', color='black')
    ax4.set_title('LR & Grad Norm', fontsize=14)
    ax4.set_xlabel('Progress (%)')

    # --- 图 5: Entropy & Reward Std (关键更新) ---
    ax5 = axes[1, 1]
    if 'entropy/mean' in df.columns:
        ln1 = ax5.plot(df['percent_num'], df['entropy/mean'], color='saddlebrown', label='Policy Entropy')
        ax5.set_ylabel('Entropy', color='saddlebrown')

    # 增加各奖励项的 Std 监控
    ax5_twin = ax5.twinx()
    std_keys = {
        'reward_std': ('Total Reward Std', 'blue'),
        'rewards/MultiModalAccuracyORM/std': ('Acc Std', 'green')
    }
    for key, (label, col) in std_keys.items():
        if key in df.columns:
            ax5_twin.plot(df['percent_num'], df[key], label=label, alpha=0.5, linestyle='-.', color=col)
    ax5_twin.set_ylabel('Standard Deviation', color='blue')
    ax5.set_title('Entropy & Reward Variance', fontsize=14)
    ax5.set_xlabel('Progress (%)')
    # 合并图例显示
    ax5.legend(loc='upper left', fontsize='x-small')
    ax5_twin.legend(loc='upper right', fontsize='x-small')

    # --- 图 6: Memory & Speed ---
    ax6 = axes[1, 2]
    if 'memory(GiB)' in df.columns:
        ax6.plot(df['percent_num'], df['memory(GiB)'], label='Memory (GiB)', color='dimgray', linewidth=2)
        ax6.set_ylabel('Memory (GiB)', color='dimgray')
    if 'train_speed(iter/s)' in df.columns:
        ax6_twin = ax6.twinx()
        ax6_twin.plot(df['percent_num'], df['train_speed(iter/s)'], label='Speed', color='deepskyblue', alpha=0.5)
        ax6_twin.set_ylabel('Speed (iter/s)', color='deepskyblue')

    ax6.set_title('System Resources', fontsize=14)
    ax6.set_xlabel('Progress (%)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_img = jsonl_path.replace('.jsonl', '_v3_metrics.png')
    plt.savefig(output_img, dpi=150)
    print(f"可视化完成！图表已保存至: {output_img}")
    plt.show()


# 使用方法
if __name__ == "__main__":
    # 替换为你的 jsonl 文件路径
    # LOG_FILE = r"C:\Users\xWX1396084\Downloads\v4-20260128-095422\logging.jsonl"
    LOG_FILE = r"C:\Users\xWX1396084\Downloads\v4-20260207-115840\logging.jsonl"
    if os.path.exists(LOG_FILE):
        plot_training_logs(LOG_FILE)
    else:
        print(f"请确保存在文件: {LOG_FILE}")
