import torch
import json
from utils import parse_number
from torch.utils.data import Dataset

def load_dataset(filename):
    with open(filename) as f:
        data = json.load(f)

    for instance in data:

        has_number = False
        numbers = []
        for x in instance["nl"]:
            numbers.append(parse_number(x))
            if numbers[-1] is not None:
                has_number = True
        instance["numbers"] = numbers
        instance["has_number"] = has_number
        return data

class AlignmentDataset(Dataset):



    def __init__(self, split):
        assert split in ('train', 'dev', 'test')
        self.loaded_data = load_dataset('../data/squall.json')
        self.data = []
        for raw_item in self.loaded_data:
            sql_statement = ""
            for sql_item in raw_item['sql']:
                sql_word = sql_item[1]
                sql_statement += sql_word + " "
            sql_words = sql_statement.split(" ")
            raw_item['sql'] = sql_words
            for align in raw_item['align']:
                current_data_item = {'nl':raw_item['nl'], 'sql':raw_item['sql'],'nl_indices':align[0],'sql_indices':align[1]}
                self.data.append(current_data_item)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def collate(batch):

        return [item for item in batch]