from model.alignment_dataset import AlignmentDataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model.alignment_model import AlignmentModel
import torch

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot

device = torch.device("cpu")
dataset = AlignmentDataset('train')

data_loader_for_inspection = DataLoader(
    dataset, batch_size=2, shuffle=True, collate_fn=AlignmentDataset.collate)
model = AlignmentModel(device).to(device)
model.eval()
for i in range(1):
    batch = next(iter(data_loader_for_inspection))

    logits, word_seq_len, sql_seq_len, all_sql_lengths = model.forward(batch)

    logits = logits[:, word_seq_len:, :]
    testy = model.get_labels(batch,logits,all_sql_lengths)
    mask = (testy!=-1)
    testy = testy[mask]
    testy=testy.reshape(-1)
    lr_probs = logits[:,:,1]
    lr_probs = lr_probs[mask]
    lr_probs = torch.exp(lr_probs.reshape(-1))
    yhat = logits.argmax(-1)
    yhat = yhat[mask]
    yhat = yhat.reshape(-1)



    lr_probs =lr_probs.detach().numpy()
    testy = testy.detach().numpy()
    yhat = yhat.detach().numpy()
    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)
    # summarize scores
    print(lr_precision, lr_recall)
    print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    # plot the precision-recall curves
    no_skill = len(testy[testy == 1]) / len(testy)
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()

