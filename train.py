#build train loop
from fusion_model import get_model
import torch
from dataset import get_dataloaders
from config import config
from torch.optim import AdamW

model=get_model()
train_dataloader=get_dataloaders()

optimizer = AdamW(model.parameters(), lr=config.train.optimizer.lr)

for epoch in range(config.train.num_epochs):
    
    model.train()
    total_train_loss=0

    batch:dict[str, torch.Tensor]
    for batch in train_dataloader: #train
        
        model(**batch)
        loss=model.loss
        
        optimizer.zero_grad() 
        loss.backward()        
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{config.train.num_epochs}], Average Training Loss: {avg_train_loss:.4f}")