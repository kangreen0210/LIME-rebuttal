import matplotlib.pyplot as plt
import numpy as np

# 数据
labels = ['AI2D', 'ScienceQA']
data1=[22.02,39.44]
data2=[54.72,60.94]

# data1 = [64.60, 39.44]
# data2 = [30.19, 22.02]


# 设置图形样式
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(6, 4))

# 设置柱状图位置和宽度
bar_width = 0.35
index = np.arange(len(labels))
# 绘制柱状图
plt.bar(index, data1, bar_width, label='LIME-M', color='#66c2ff', edgecolor='black')
plt.bar(index + bar_width, data2, bar_width, label='Original', color='#ff9933', edgecolor='black')

# 添加标签和标题
# 添加标签和标题
# plt.xlabel('Groups', fontsize=12)
plt.ylabel('Avg Score', fontsize=16)
# plt.title('Comparison of Two Data Sets', fontsize=14)
plt.xticks(index + bar_width / 2, labels, fontsize=14, fontstyle='italic')
plt.yticks(fontsize=14, fontstyle='italic')

# 去除顶部和右侧边框
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 添加图例
<<<<<<< HEAD
plt.legend(fontsize=16, frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))

=======
plt.legend(fontsize=16, frameon=False, loc='center left', bbox_to_anchor=(1, 0.5)) # 位置调整为右上角
>>>>>>> 865c7069caf994108f2fb1c2648cb346c8741a4e

# 调整图形边距
plt.tight_layout()

# # 显示图形
# plt.show()


# 显示图形
<<<<<<< HEAD
plt.savefig('./static_figs/text_only.jpg')
plt.savefig('./static_figs/text_only.pdf')
=======
plt.savefig('/ML-A100/team/mm/zk/lmms-eval/static_figs/text_only.jpg')
plt.savefig('/ML-A100/team/mm/zk/lmms-eval/static_figs/text_only.pdf')
>>>>>>> 865c7069caf994108f2fb1c2648cb346c8741a4e
