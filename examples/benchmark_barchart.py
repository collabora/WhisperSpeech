import matplotlib.pyplot as plt
import numpy as np

# Extract the necessary data from the table
t2s_models = ['tiny-en+pl', 'hq-fast-en+pl', 'base-en+pl', 'fast-medium-en+pl+yt', 'small-en+pl']
s2a_models = ['tiny-en+pl', 'hq-fast-en+pl', 'base-en+pl', 'small-en+pl']
processing_times = [
    [15.06, 15.17, 20.76, 37.34],
    [15.29, 15.29, 21.26, 37.65],
    [16.09, 16.35, 21.94, 38.78],
    [17.38, 17.50, 23.32, 39.59],
    [19.66, 20.05, 25.31, 42.05]
]

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Create the stacked horizontal bar chart
bar_height = 0.6
y = np.arange(len(t2s_models))
left = np.zeros(len(t2s_models))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i in range(len(s2a_models)):
    widths = [row[i] for row in processing_times]
    bars = ax.barh(y, widths, bar_height, left=left, label=s2a_models[i], color=colors[i])
    
    # Add labels to each layer of the stacked bar
    for bar in bars:
        width = bar.get_width()
        ax.text(bar.get_x() + width / 2, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', ha='center', va='center', color='white')
    
    left += widths

# Customize the chart
ax.set_ylabel('T2S Model')
ax.set_xlabel('Total Processing Time (s)')
ax.set_title('Processing Time for T2S and S2A Model Combinations')
ax.set_yticks(y)
ax.set_yticklabels(t2s_models)
ax.legend(title='S2A Model')

# Display the chart
plt.tight_layout()
plt.show()