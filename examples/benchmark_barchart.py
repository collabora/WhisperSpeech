import matplotlib.pyplot as plt
import numpy as np


t2s_models = ['tiny-en+pl', 'hq-fast-en+pl', 'base-en+pl', 'fast-medium-en+pl+yt', 'small-en+pl']
s2a_models = ['tiny-en+pl', 'hq-fast-en+pl', 'base-en+pl', 'small-en+pl']
processing_times = [
    [15.06, 15.17, 20.76, 37.34],
    [15.29, 15.29, 21.26, 37.65],
    [16.09, 16.35, 21.94, 38.78],
    [17.38, 17.50, 23.32, 39.59],
    [19.66, 20.05, 25.31, 42.05]
]


fig, ax = plt.subplots(figsize=(10, 6))


bar_width = 0.6
x = np.arange(len(t2s_models))
bottom = np.zeros(len(t2s_models))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i in range(len(s2a_models)):
    heights = [row[i] for row in processing_times]
    bars = ax.bar(x, heights, bar_width, bottom=bottom, label=s2a_models[i], color=colors[i])
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2, f'{height:.2f}', ha='center', va='center', color='white')
    
    bottom += heights


ax.set_xlabel('T2S Model')
ax.set_ylabel('Total Processing Time (s)')
ax.set_title('Processing Time for T2S and S2A Model Combinations')
ax.set_xticks(x)
ax.set_xticklabels(t2s_models)
ax.legend(title='S2A Model')


plt.tight_layout()
plt.show()