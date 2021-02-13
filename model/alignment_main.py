import torch
from alignment_dataset import AlignmentDataset
from torch.utils.data import DataLoader
import tqdm
from alignment_model import AlignmentModel

print(torch.cuda.is_available())
# assert torch.cuda.is_available()
device = torch.device("cuda")
#device = torch.device("cpu")
print("device:", device)

model = AlignmentModel(device).to(device)


def train(model, num_epochs, batch_size, model_file,
          learning_rate=8e-4, dataset_cls=AlignmentDataset):

    dataset = dataset_cls('train')
    train_size = int(len(dataset)*0.8)
    valid_size = int(len(dataset)*0.1)
    test_size = len(dataset) - valid_size - train_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                               [train_size, valid_size, test_size])
    data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(data_loader),
        pct_start=0.1,  # Warm up for 10% of the total training time
    )
    best_metric = 0.0
    for epoch in tqdm.trange(num_epochs, desc="training", unit="epoch"):
        with tqdm.tqdm(
                data_loader,
                desc="epoch {}".format(epoch + 1),
                unit="batch",
                total=len(data_loader)) as batch_iterator:
            model.train()
            total_loss = 0.0
            for i, batch in enumerate(batch_iterator, start=1):
                optimizer.zero_grad()
                loss = model.compute_loss(batch)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step()
                batch_iterator.set_postfix(mean_loss=total_loss / i)
            validation_metric = model.get_validation_metric(dataset=valid_dataset)
            batch_iterator.set_postfix(
                mean_loss=total_loss / i,
                validation_metric=validation_metric)
            if validation_metric > best_metric:
                print(
                    "Obtained a new best validation metric of {:.3f}, saving model "
                    "checkpoint to {}...".format(validation_metric, model_file))
                torch.save(model.state_dict(), model_file)
                best_metric = validation_metric

train(model,num_epochs=50,batch_size=64,model_file="alignment_model.pt", learning_rate=8e-5)