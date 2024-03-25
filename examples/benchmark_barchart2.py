import matplotlib.pyplot as plt
import numpy as np

# Extract the necessary data from the table
t2s_models = ['tiny-en+pl', 'hq-fast-en+pl', 'base-en+pl', 'fast-medium-en+pl+yt', 'small-en+pl']
s2a_models = ['tiny-en+pl', 'hq-fast-en+pl', 'base-en+pl', 'small-en+pl']

first_segment_times = [
    [2.90, 2.96, 4.06, 7.31],
    [2.99, 2.98, 4.21, 7.40],
    [3.13, 3.16, 4.31, 7.49],
    [3.36, 3.40, 4.43, 7.68],
    [3.83, 3.87, 4.91, 8.11]
]

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Create the stacked horizontal bar chart
bar_height = 0.6
y = np.arange(len(t2s_models))
left = np.zeros(len(t2s_models))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i in range(len(s2a_models)):
    widths = [row[i] for row in first_segment_times]
    bars = ax.barh(y, widths, bar_height, left=left, label=s2a_models[i], color=colors[i])
    
    # Add labels to each layer of the stacked bar
    for bar in bars:
        width = bar.get_width()
        ax.text(bar.get_x() + width / 2, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', ha='center', va='center', color='white')
    
    left += widths

# Customize the chart
ax.set_ylabel('T2S Model')
ax.set_xlabel('First Segment Processing Time (s)')
ax.set_title('First Segment Processing Time for T2S and S2A Model Combinations')
ax.set_yticks(y)
ax.set_yticklabels(t2s_models)
ax.legend(title='S2A Model')

# Display the chart
plt.tight_layout()
plt.show()