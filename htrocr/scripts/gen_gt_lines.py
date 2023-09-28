import os
import pandas as pd

folder_loc = "./NHMD_GT/image"
gt_xlsx = "./NHMD_GT/info.xlsx"
output_path = "./Data/GT.txt"

def write_filenames():
    files = os.listdir(folder_loc)
    files = [(os.path.join(folder_loc, f), f) for f in files] # add path to each file
    files.sort(key=lambda x: os.path.getmtime(x[0]))
    with open(output_path, "w") as f:
        for filename in files:
            path = os.path.join(folder_loc, filename[1])
            if os.path.isfile(path) and filename[1].startswith('sp'):
                f.write(filename[1]+"\t\n")

def write_transcriptions():
    dataframe1 = pd.read_excel(gt_xlsx)
    files = dataframe1['image']
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, filename in enumerate(files):
            lines = dataframe1.iloc[idx]['text'].split("\n")
            for i, line in enumerate(lines):
                newname = filename[:-3] + str(i+1) + ".jpg"
                f.write(newname+"\t"+line+"\n")

if __name__ == '__main__':
    write_transcriptions()