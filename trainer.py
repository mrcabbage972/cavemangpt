from datetime import datetime

import torch.utils.data

from cfg import model_cfg, train_cfg
from dataset import CavemanGPTDataset
from model import CavemanGPT


def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        optimizer.zero_grad()
        outputs = model(**data)

        loss = outputs[1]
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 1 == 0:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            running_loss = 0.

    return last_loss

input_texts = ["abc" * (i+1) for i in range(5)]
training_set = CavemanGPTDataset(input_texts, block_size=8)

input_texts_val = ["abcd" * (i+1) for i in range(5)]
validation_set = CavemanGPTDataset(input_texts_val, block_size=8)

training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False)

# Report split sizes
print('Training set has {} instances'.format(len(training_set)))
print('Validation set has {} instances'.format(len(validation_set)))

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

epoch_number = 0

EPOCHS = train_cfg['epochs']

best_vloss = 1_000_000.

model  = CavemanGPT(model_cfg)

optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'])

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number)

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    for i, vdata in enumerate(validation_loader):
        voutputs = model(**vdata)
        vloss = voutputs[1]
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1