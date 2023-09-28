import os
import pandas as pd

transcription_file = "./NHMD_GT/gt_test.txt"
gt_file = './NHMD_GT/gt_test.txt'
output_folder = './data/trocr_results/large'

def gen_numerated_mappings():
    mappings = []
    with open(gt_file, encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            # print(line.split('\t')[0].split('.')[0]+".att.txt")
            mappings.append(line.split('\t')[0].split('.')[0]+".att.txt")        
    return mappings

def extract_lines_to_files(mappings):
    with open(transcription_file, encoding="utf-8") as f:
        lines = f.readlines()
        start = True
        file = ''
        final_res = ''
        for line in lines:
            if line.startswith("D-"):
                line_segs = line.split('\t')
                id = int(line_segs[0][2:])
                result = line.split("\t")[2]
                if not start and file == mappings[id]:
                    final_res += result
                elif start:
                    file = mappings[id]
                    final_res += result
                    start = False
                else:
                    output_path = os.path.join(output_folder, file)
                    with open(output_path, "w") as o:
                        o.write(final_res)
                    final_res=''
                    final_res += result
                    file = mappings[id]
            if line.startswith("Generate test with beam=10:"):
                output_path = os.path.join(output_folder, file)
                with open(output_path, "w") as o:
                    o.write(final_res)

if __name__ == '__main__':
    mappings=gen_numerated_mappings()
    extract_lines_to_files(mappings)
