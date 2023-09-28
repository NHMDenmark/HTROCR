import pandas as pd
import math
# data = pd.read_csv('taxondata.csv', sep='\t')
# output = ""
# for l in data['scientificName']:
#     if (l.startswith('SH')) or (l.startswith('BOLD')):
#         continue
#     output += f'{l}\n'

# with open('taxon_info_.txt', 'w') as w:
#     w.write(output)


data = pd.read_csv('taxon.txt', sep='\t')
output = ""
for l in data['scientificName']:
    l = str(l)
    if (l.startswith('SH')) or (l.startswith('BOLD')) or (l == 'nan'):
        continue
    output += f'{l}\n'

with open('taxon_info_.txt', 'w') as w:
    w.write(output)


