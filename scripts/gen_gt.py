import os
import pandas as pd


gt_xlsx = "./data/NHMD_LINES_100/info.xlsx"
output_folder = "./data/NHMD_GT"


def write_gt():
    dataframe1 = pd.read_excel(gt_xlsx)
    files = dataframe1['image']
    for idx, filename in enumerate(files):
        output_path = os.path.join(output_folder, filename[:-3]+'txt')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            lines = dataframe1.iloc[idx]['text']
            f.write(lines)

if __name__ == '__main__':
    write_gt()