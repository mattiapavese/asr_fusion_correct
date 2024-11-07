#build train loop
from fusion_model import get_model
import torch
from dataset import get_dataloaders
from config import config
from torch.optim import AdamW
import os
import shutil

train_logs_dir="train_logs"
if not os.path.exists(train_logs_dir):
    os.makedirs(train_logs_dir)

exp_logs_dir=os.path.join(train_logs_dir, config.train.exp_name)
if os.path.exists(exp_logs_dir):
    raise ValueError((f"An experiment directory named {exp_logs_dir} already exists. "
                      "Consider modifying train.exp_name in config.yml ."))
else:
    os.makedirs(exp_logs_dir)

train_logs_path=os.path.join(exp_logs_dir, "train.logs.tsv")
with open(train_logs_path,"w") as f:
    f.write("Epoch\tStep\tAvg Train Loss\tAvg Val Loss\n")

#copy the configuration file to eventually reproduce the job
shutil.copy("config.yml", os.path.join(exp_logs_dir, "config.yml"))


model=get_model()

train_dataloader, val_dataloader=get_dataloaders()

optimizer = AdamW(model.parameters(), lr=config.train.optimizer.lr)

step=0
total_train_loss=0
logging_steps=config.train.logging_steps

for epoch in range(config.train.num_epochs):
    
    model.train()
    print(f"Training Epoch: {epoch+1}")

    batch:dict[str, torch.Tensor]
    for i, batch in enumerate(train_dataloader):
        step+=1

        model(**batch)
        loss=model.loss
        
        optimizer.zero_grad() 
        loss.backward()        
        optimizer.step()

        total_train_loss += loss.item()
    
        if step%logging_steps==0:
            #perform validation and logging
            total_val_loss=0

            for _, batch in enumerate(val_dataloader):
                model(**batch)
                val_loss=model.loss

                total_val_loss += val_loss.item()
            
            avg_train_loss = total_train_loss / logging_steps
            avg_val_loss = total_val_loss / len(val_dataloader)

            
            print((f"[Epoch: {epoch+1}/{config.train.num_epochs}, "
                #    f"-- {int( (i+1)*100/len(train_dataloader) )} "
                   f"Step: {step}],"
                   f"Average Training Loss: {avg_train_loss:.4f}, "
                   f"Average Validation Loss: {avg_val_loss:.4f}"))
            
            with open(train_logs_path, "a") as f:
                f.write(f"{epoch+1}\t{step}\t{avg_train_loss}\t{avg_val_loss}\n")

            total_train_loss=0

