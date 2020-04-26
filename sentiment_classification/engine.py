from tqdm import tqdm
import torch
import torch.nn as nn


def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1,1))


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()

    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d['input_ids']
        # mask = d['mask']
        # token_type_ids = d['token_type_ids']
        targets = d['targets']

        ids.to(device, dtype=torch.long)
        # mask.to(device, dtype=torch.long)
        # token_type_ids.to(device, dtype=torch.long)
        targets.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(
            ids=ids
            # mask=mask,
            # token_type_ids=token_type_ids
        )

        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

    def eval_fn(data_loader, model, device):
        model.eval()
        fin_targets = []
        fin_outputs = []

        with torch.no_grad():
            for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
                ids = d['ids']
                # mask = d['mask']
                # token_type_ids = d['token_type_ids']
                targets = d['targets']

                ids.to(device, dtype=torch.long)
                # mask.to(device, dtype=torch.long)
                # token_type_ids.to(device, dtype=torch.long)
                targets.to(device, dtype=torch.long)

                
                outputs = model(
                    ids=ids
                    # mask=mask,
                    # token_type_ids=token_type_ids
                )

                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

            return fin_outputs, fin_targets
        


def eval_fn():
    pass
