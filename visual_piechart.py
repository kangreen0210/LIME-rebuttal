import json
import statistics
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from tqdm import tqdm




task_name = 'textvqa_val'
file_dir = f'./logs_new/{task_name}'

# read all .json files in the directory
files = os.listdir(file_dir)
files_list = [f for f in files if f.endswith('.json')]
# print(files_list)
# read all .json files in the files_list
data_dict = {}
for file in files_list:
    with open(f'{file_dir}/{file}', 'r') as f:
        # delete the .json extension
        model_name = file[:-5]
        data_dict[model_name] = json.load(f)['logs']

# print(data_dict.keys())
# stastic the 'chosen' of data_dict['idefics2']
# 'chosen' type: middle, esay, difficult, delete
list_middle = []
list_easy = []
list_difficult = []
list_delete = []

for i, element in tqdm(enumerate(data_dict['idefics2'])):
    if element['chosen']== 'delete':
        list_delete.append(element['doc_id'])
    elif element['chosen']== 'easy':
        list_easy.append(element['doc_id'])
    elif element['chosen']== 'middle':
        list_middle.append(element['doc_id'])
    elif element['chosen']== 'difficult':
        list_difficult.append(element['doc_id'])


print(len(list_delete), len(list_easy), len(list_middle), len(list_difficult))

labels = ['delete', 'easy', 'middle', 'difficult']
sizes = [len(list_delete), len(list_easy), len(list_middle), len(list_difficult)]
colors = ['#FF6347', '#FFD700', '#1E90FF', '#32CD32']  # Tomato, Gold, DodgerBlue, LimeGreen
explode = (0.1, 0, 0, 0.1)  # "explode" the 1st and 4th slice slightly

# Create a pie chart
fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                  autopct='%1.1f%%', shadow=True, startangle=140, 
                                  wedgeprops={'edgecolor': 'black', 'linewidth': 1.5, 'linestyle': 'solid'},
                                  textprops={'fontsize': 14})

# Enhance the look with some additional styling
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_weight('bold')

# Add a legend with title
ax.legend(wedges, labels, title="Task Difficulty", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

# Add a title with custom font size and color
plt.title(f'The distribution of the chosen labels in {task_name}', fontsize=16, color='darkblue', weight='bold')

# Save the pie chart
plt.savefig(f'./static_figs/pie_{task_name}.png')

# Show the plot
plt.show()
