import pandas as pd
from tqdm import tqdm
from constants import ROOT, DATA_DIR


def main():
    monthly_features = pd.read_csv(
        ROOT / "Sustainability_Analysis/monthly_features.csv"
    ).set_index("project_id")

    col_bases = []
    month_indices = []

    for col in monthly_features.columns:
        if col not in ["project_id", "Unnamed: 2861", " "]:
            month_index = col.split("_")[0]
            col_base = "_".join(col.split("_")[1:])
            col_bases.append(col_base)
            month_indices.append(month_index)

    col_bases = list(set(col_bases))
    month_indices = sorted(list([int(x) for x in set(month_indices)]))

    project_history = {}

    for project in tqdm(monthly_features.index.unique()):
        project_history[project] = []
        for month_idx in month_indices:
            colnames = [f"{month_idx}_{col_base}" for col_base in col_bases]
            row = monthly_features.loc[project, colnames]
            row.index = col_bases
            row.name = month_idx
            project_history[project].append(row)

    single_proj_history = None
    for proj, rows in project_history.items():
        single_proj_history = pd.concat(rows, axis=1)
        single_proj_history.to_csv(
            DATA_DIR / "single_project_histories" / f"single_project_{proj}.csv"
        )


if __name__ == "__main__":
    main()
