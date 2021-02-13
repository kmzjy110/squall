import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel
from alignment_dataset import AlignmentDataset

if torch.cuda.is_available():

    def from_numpy(ndarray):
        return torch.from_numpy(ndarray).pin_memory().cuda()
else:
    from torch import from_numpy



class AlignmentModel(nn.Module):
    def get_validation_metric(self, dataset, batch_size=8):
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, collate_fn=AlignmentDataset.collate)
        self.eval()
        technically_correct = 0
        total = 0

        total_zero=0
        total_one=0
        zero_correct =0
        one_correct=0
        with torch.no_grad():
            for batch in data_loader:
                logits, word_seq_len, sql_seq_len, all_sql_lengths = self.forward(batch)
                predicted_labels = logits.argmax(-1)
                actual_labels = self.get_labels(batch, logits, all_sql_lengths)

                mask = (actual_labels!=-1)
                predicted_labels_all_mask = predicted_labels[mask]
                actual_labels_all_mask = actual_labels[mask]

                technically_correct +=(predicted_labels_all_mask==actual_labels_all_mask).sum().item()
                total += actual_labels_all_mask.shape[0]

                mask = (actual_labels==0)
                predicted_labels_zero_mask = predicted_labels[mask]
                actual_labels_zero_mask = actual_labels[mask]

                zero_correct += (predicted_labels_zero_mask==actual_labels_zero_mask).sum().item()
                total_zero +=actual_labels_zero_mask.shape[0]

                mask = (actual_labels==1)
                predicted_labels_one_mask = predicted_labels[mask]
                actual_labels_one_mask = actual_labels[mask]

                one_correct +=(predicted_labels_one_mask==actual_labels_one_mask).sum().item()
                total_one +=actual_labels_one_mask.shape[0]


        print("technically correct:",technically_correct)
        print("total:",total)

        print()
        print("zero_correct:", zero_correct)
        print("zero_total:", total_zero )

        print()
        print("one_correct:", one_correct)
        print("one_total:", total_one)
        return 0.5 * one_correct/total_one + 0.5*zero_correct/total_zero

    def __init__(self, device):
        super().__init__()
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.bert_feature_dim = self.bert_model.pooler.dense.in_features
        self.indicator_vector = torch.nn.parameter.Parameter(torch.rand(self.bert_feature_dim, requires_grad=True,dtype=torch.double), requires_grad=True).to(device)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = self.bert_feature_dim, nhead=8)

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,num_layers=6)
        self.norm_layer = nn.LayerNorm(self.bert_feature_dim)
        self.projection_layer = nn.Linear(self.bert_feature_dim, 2)
        self.device = device
        #self.loss = nn.NLLLoss(weight=torch.tensor([0.17105,1]),reduction='mean', ignore_index=-1)
        self.loss = nn.NLLLoss(weight=torch.tensor([1.0, 1.0]), reduction='mean', ignore_index=-1)
        self.lsm = nn.LogSoftmax(dim=2)

    def get_labels(self, batch, logits, all_sql_lengths):
        sql_indices = [ins['sql_indices'] for ins in batch]
        labels = torch.zeros((logits.shape[0], logits.shape[1]))
        for i, row in enumerate(sql_indices):
            for j in row:
                labels[i, j] = 1
        for i in range(labels.shape[0]):
            labels[i, all_sql_lengths[i]:] = -1
        labels = labels.to(self.device)
        return labels

    def compute_loss(self, batch):
        logits, word_seq_len, sql_seq_len, all_sql_lengths = self.forward(batch)
        logits = logits[:, word_seq_len:, :]
        labels = self.get_labels(batch,logits,all_sql_lengths)
        logits = logits.reshape((-1,logits.shape[-1])).float()
        labels = labels.reshape((-1,)).long()
        #loss = self.cross_entropy_loss(logits, labels, reduction='mean') do LSM and the NLL
        loss = self.loss(logits,labels)
        return loss


    def forward(self, batch):
        nl_indices = [ins['nl_indices'] for ins in batch]
        features, bert_word_features, bert_sql_features, all_sql_lengths= self._bert_embed(batch)
        mask = np.zeros(bert_word_features.shape)
        one_vector = np.ones(bert_word_features.shape[-1])
        for i, row in enumerate(nl_indices):
            for j in row:
                mask[i,j] = one_vector
        mask_tensor = torch.tensor(mask,dtype=torch.float).to(self.device)
        indicator_broadcast = self.indicator_vector * mask_tensor
        indicator_broadcast = indicator_broadcast.to(self.device)
        bert_word_features = bert_word_features + indicator_broadcast

        input_vec = torch.cat((bert_word_features.float(),bert_sql_features.float()),1)
        transformer_out = self.transformer_encoder(input_vec)
        transformer_out = self.norm_layer(transformer_out)
        transformer_out = self.projection_layer(transformer_out)
        transformer_out = self.lsm(transformer_out)
        return transformer_out, bert_word_features.shape[1], bert_sql_features.shape[1], all_sql_lengths





    def _bert_embed(self, instances):
        word_seq_max_len = max([len(ins["nl"]) for ins in instances])
        sql_seq_max_len = max([len(ins['sql']) for ins in instances])
        all_sql_lengths = [len(ins['sql']) for ins in instances]
        all_input_ids = np.zeros((len(instances), 2048), dtype=int)
        all_input_type_ids = np.zeros((len(instances), 2048), dtype=int)
        all_input_mask = np.zeros((len(instances), 2048), dtype=int)
        all_word_end_mask = np.zeros((len(instances), 2048), dtype=int)
        all_sql_end_mask = np.zeros((len(instances), 2048), dtype=int)
        subword_max_len = 0


        for sentence_idx, ins in enumerate(instances):
            nl_query = ins['nl']

            tokens = []
            token_types = []
            word_end_mask = []
            sql_end_mask = []

            tokens.append("[CLS]")
            token_types.append(0)
            word_end_mask.append(0)
            sql_end_mask.append(0)
            for index, word in enumerate(nl_query):

                word_tokens = self.bert_tokenizer.tokenize(word)
                if len(word_tokens)==0:
                    word_tokens=['.']
                for _ in range(len(word_tokens)):
                    word_end_mask.append(0)
                    sql_end_mask.append(0)
                    token_types.append(0)
                word_end_mask[-1] = 1
                tokens.extend(word_tokens)


            tokens.append("[SEP]")
            word_end_mask.append(0)
            sql_end_mask.append(0)
            token_types.append(0)



            for index,sql_word in enumerate(ins['sql']):
                sql_tokens = self.bert_tokenizer.tokenize(sql_word)
                if len(sql_tokens)==0:
                    sql_tokens = ['.']
                for _ in range(len(sql_tokens)):
                    word_end_mask.append(0)
                    sql_end_mask.append(0)
                    token_types.append(1)
                sql_end_mask[-1] = 1
                tokens.extend(sql_tokens)
            tokens.append("[SEP]")
            word_end_mask.append(0)
            sql_end_mask.append(0)
            token_types.append(1)

            for i in range(word_seq_max_len - len(ins['nl'])):
                word_end_mask.append(1)
            for i in range(sql_seq_max_len - len(ins['sql'])):
                sql_end_mask.append(1)

            input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
            input_type_ids = token_types
            input_mask = [1] * len(input_ids)
            subword_max_len = max(subword_max_len, len(word_end_mask)+1, len(sql_end_mask)+1)
            all_input_ids[sentence_idx, :len(input_ids)] = input_ids
            all_input_type_ids[sentence_idx, :len(input_ids)] = input_type_ids
            all_input_mask[sentence_idx, :len(input_mask)] = input_mask
            all_word_end_mask[sentence_idx, :len(word_end_mask)] =word_end_mask
            all_sql_end_mask[sentence_idx, :len(sql_end_mask)] = sql_end_mask

        all_input_ids = from_numpy(np.ascontiguousarray(all_input_ids[:, :subword_max_len])).long()
        all_input_type_ids = from_numpy(np.ascontiguousarray(all_input_type_ids[:, :subword_max_len])).long()
        all_input_mask = from_numpy(np.ascontiguousarray(all_input_mask[:, :subword_max_len])).long()
        all_word_end_mask = from_numpy(np.ascontiguousarray(all_word_end_mask[:, :subword_max_len])).long()
        all_sql_end_mask = from_numpy(np.ascontiguousarray(all_sql_end_mask[:, :subword_max_len])).long()

        features, _ = self.bert_model(all_input_ids, token_type_ids=all_input_type_ids, attention_mask=all_input_mask)

        bert_word_features = features.masked_select(all_word_end_mask.to(torch.bool).unsqueeze(-1)).reshape(
            len(instances), word_seq_max_len, features.shape[-1]).float()
        bert_sql_features = features.masked_select(all_sql_end_mask.to(torch.bool).unsqueeze(-1)).reshape(
            len(instances), sql_seq_max_len, features.shape[-1]).float()

        return features, bert_word_features, bert_sql_features, all_sql_lengths












































