import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data preparation

excel_file = r'filepath\fig3&SIfigS78_results.xlsx'
dfs = pd.read_excel(excel_file, sheet_name=None)

# Iterate through each sheet and output sheet name and the first few rows of the DataFrame
for sheet_name, df in dfs.items():
    print(f"Sheet name: {sheet_name}")
    print(df.head())  # Output the first few rows of the DataFrame, adjust as needed

    def get_label_rotation(angle, offset):
        rotation = np.rad2deg(angle + offset)
        if angle <= np.pi:
            alignment = "right"
            rotation = rotation + 180
        else: 
            alignment = "left"
        return rotation, alignment

    def add_labels(angles, values, labels, pvalues, offset, ax):
        padding = 0.1  # Adjust the distance between labels and bars
        for angle, value, label, pvalue in zip(angles, values, labels, pvalues):
            rotation, alignment = get_label_rotation(angle, offset)
            value_formatted = "{:.3f}".format(value)  # Format values to three decimal places
            star_label = ""

            # Add asterisks based on pvalue
            if pvalue < 0.01:
                star_label = "***"
            elif 0.01 <= pvalue < 0.05:
                star_label = "**"
            elif 0.05 <= pvalue < 0.1:
                star_label = "*"

            # Annotate value and asterisks
            combined_label = "{}{}".format(value_formatted, star_label)
            ax.text(
                x=angle,
                y=value + padding,
                s=combined_label,
                ha=alignment,
                va="bottom",  # Adjust vertical alignment
                rotation=rotation,
                rotation_mode="anchor"
            )

    ANGLES = np.linspace(0, 2 * np.pi, len(df), endpoint=False)
    VALUES = df["value"].values
    LABELS = df["name"].values
    PVALUES = df["Pvalue"].values  # Get P-value column
    GROUP = df["group"].values
    PAD = 2
    ANGLES_N = len(VALUES) + PAD * len(np.unique(GROUP))
    ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
    WIDTH = (2 * np.pi) / len(ANGLES)
    OFFSET = np.pi / 2
    GROUPS_SIZE = [len(i[1]) for i in df.groupby("group")]
    offset = 0
    IDXS = []
    for size in GROUPS_SIZE:
        IDXS += list(range(offset + PAD, offset + size + PAD))
        offset += size + PAD
    
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['figure.dpi'] = 300
    # plt.rcParams['font.size'] = 18
    if 'AP' in sheet_name:
        plt.rcParams['font.size'] = 14
    else:
        plt.rcParams['font.size'] = 18
    
    # Create chart
    fig, ax = plt.subplots(figsize=(20, 10), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(OFFSET)

    value_max = np.max(VALUES)
    value_min = np.min(VALUES)
    y_upper = value_max + 0.1 * value_max
    y_lower = value_min
    ax.set_ylim(y_lower, y_upper)
    ax.set_frame_on(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    COLORS = ['#366695','#518FAE','#73BBD7','#ADDBE3','#E85F54','#EF8A47','#FDD16E','#FEE7B8',
            '#366695','#518FAE','#73BBD7','#ADDBE3','#E85F54','#EF8A47','#FDD16E','#FEE7B8',
            '#366695','#518FAE','#73BBD7','#ADDBE3','#E85F54','#EF8A47','#FDD16E','#FEE7B8']

    ax.bar(
        ANGLES[IDXS], VALUES, width=WIDTH, color=COLORS, 
        edgecolor="white", linewidth=2
    )

    # Add labels with asterisks
    add_labels(ANGLES[IDXS], VALUES, LABELS, PVALUES, OFFSET, ax)
    plt.tight_layout()
    fig.set_size_inches(20, 10)

    plt.show()