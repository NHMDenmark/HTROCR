from htrocr.database.collectiontransformers.CollectionTransformer import CollectionTransformer
import json
import re


class EmunchTransformer(CollectionTransformer):
    def __init__(self, path):
        super().__init__(path)
        with open('config/emunch.json') as f:
            self.params = json.load(f)

    def format_textline(self, line):
        '''
        Remove supplementary annotations that do not match the gt.
        '''
        l = line.strip()
        l = re.sub(r"{\s\.\.\.\s}", "", l)
        l = re.sub(r"{\s\…\s}", "", l)
        l = re.sub(r"\\", "", l)
        l = re.sub(r"\/", "", l)
        l = re.sub(r"‹", "", l)
        l = re.sub(r"›", "", l)
        l = re.sub(r"½", " 1/2", l)
        if re.search('\s*\.\.\.\s*', l) or \
        re.search('\s*\…s*', l) or \
        re.search('{\w+?}', l):
            return l, True
        formatted_lines = re.split(r'\s{3,}', l)
        return formatted_lines, False