from tqdm import tqdm
import torch
import math


def perform_epoch(epoch_id, model, train_dataloader, eval_dataloader,
                  eval_dataset, batch_size, accelerator, optimizer,
                  lr_scheduler):
    # Training
    model.train()
    for batch in tqdm(train_dataloader):
        # temp = batch.pop('token_type_ids')
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    # Evaluation
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            # temp = batch.pop('masked_token_type_ids')
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(batch_size)))

    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Epoch {epoch_id}: Perplexity: {perplexity}")
