import pandas as pd

df = pd.read_csv("data\outputs\overall_outputs\evaluation_full.csv")

metrics = [
    "specificity",
    "actionability",
    "relevance",
    "helpfulness",
    "groundedness"
]

rows = []

for system, group in df.groupby("system"):

    row = {"system": system}

    for metric in metrics:
        mean = group[metric].mean()
        std = group[metric].std()

        row[metric] = f"{mean:.2f} ± {std:.2f}"

    rows.append(row)

result_df = pd.DataFrame(rows)

print(result_df)

result_df.to_csv("summary_table.csv", index=False)