from model import alignment_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer

dataset = alignment_dataset.AlignmentDataset('train')

data_loader_for_inspection = DataLoader(
    dataset, batch_size=2, shuffle=True, collate_fn=alignment_dataset.AlignmentDataset.collate)
for i in range(5):
    a = next(iter(data_loader_for_inspection))
    for i in range(len(a)):

        ins = a[i]
        print(ins['nl'])
        nl_query = ins['nl']
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        nl_indices = set(ins['nl_indices'])
        new_nl_indices = []
        current_new_nl_index = 0
        tokens = []
        for index, word in enumerate(nl_query):

            word_tokens = bert_tokenizer.tokenize(word)
            if len(word_tokens) == 0:
                word_tokens = ['.']
            for _ in range(len(word_tokens)):
                if index in nl_indices:
                    new_nl_indices.append(current_new_nl_index)
                current_new_nl_index += 1
            tokens.extend(word_tokens)
        print(tokens)
        print(nl_indices)
        print(new_nl_indices)
        ins['nl_indices'] = new_nl_indices
        print(ins)
        print(ins['nl_indices'])
