import pandas as pd

data = pd.read_csv('taxondata.csv', sep='\t')
output = ""
for l in data['scientificName']:
    if (l.startswith('SH')) or (l.startswith('BOLD')):
        continue
    output += f'{l}\n'

with open('taxon_info_2.txt', 'w') as w:
    w.write(output)