from collectiontransformers.CollectionTransformer import CollectionTransformer
import json
import re
import os


class MMDTransformer(CollectionTransformer):
    def __init__(self, path):
        super().__init__(path)
        with open('config/mmd.json') as f:
            self.params = json.load(f)

    def format_textline(self, line):
        '''
        GT transcriptions do not include line breaks. Mapping should be done manually.
        '''
        return line, False
    
    def extract_line_gt(self, file, output, ql, root):
        output
        for i in range(1,25):
            output += '{}_line_{}.jpg\t{}\n'.format(file.replace('.txt', ''), i, '')
        return output, ''