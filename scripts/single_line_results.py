import shutil
import os
import io

RESULTS_DIR = './data/results'
OUTPUT_DIR = './data/single_line_results'

def cp_dirs():
    #for dir in os.listdir(RESULTS_DIR):
        dir = 'TROCR_NHMD_LINES_IAM_BASE'
        if dir != '.DS_Store':
            # print(file)
            old_path = os.path.join(RESULTS_DIR, dir)
            new_path = os.path.join(OUTPUT_DIR, dir)
            # os.makedirs(new_path, exist_ok = True)
            shutil.copytree(old_path, new_path)

def transform_to_single_line(dir):
    data_path = os.path.join(OUTPUT_DIR, dir)
    for file in os.listdir(data_path):
        if file != '.DS_Store' and not os.path.isdir(os.path.join(data_path, file)):
            file_path = os.path.join(data_path, file)
            with open(file_path, 'r+') as r :
                filedata = r.read()

            # Replace the target string
            filedata = filedata.replace('\n', ' ')

            # # Write the file out again
            with open(os.path.join(OUTPUT_DIR, '../trocr_results/large/single_line/'+file), 'w') as w:
                w.write(filedata)

if __name__ == '__main__':
    # cp_dirs()
    transform_to_single_line('../trocr_results/large')
