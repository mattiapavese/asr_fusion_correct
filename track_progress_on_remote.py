#NB make sure the train.exp_name in config matches the same running in remote

import subprocess, os
import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
import numpy as np
import polars
from config import config

def update_line(line:Line2D, data_x, data_y):

    if len(data_x) and len(data_y):
        line.set_ydata(data_y)
        line.set_xdata(data_x)
        
        plt.pause(0.1)  # pause briefly to update the plot


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
train_logs_path_remote=os.path.join(config.utils.remote_machine_workdir, train_logs_path)


fig, ax = plt.subplots()
line_train_loss, = ax.plot([],[], '-r')
line_val_loss, = ax.plot([],[], '-b')

ax.set_yscale("log")

curr_size=0
while True:
    command=(f"scp -p {config.utils.remote_machine_port} -i ~/.ssh/id_ed25519 "
             f"{config.utils.remote_machine_user}@{config.utils.remote_machine_ip}:"
             f"{train_logs_path_remote} {exp_logs_dir}").split(" ") 
    
    subprocess.run(command)

    time.sleep(5)

    if os.path.getsize(train_logs_path)!=curr_size:        

        df=polars.read_csv(train_logs_path, separator="\t", quote_char=None)
        steps=df["Step"]
        avg_train_loss=df["Avg Train Loss"]
        avg_val_loss=df["Avg Val Loss"]

        update_line(line_train_loss, steps, avg_train_loss)
        update_line(line_val_loss, steps, avg_val_loss)

        max_y=polars.concat([avg_train_loss,avg_val_loss]).max()
        min_y=polars.concat([avg_train_loss,avg_val_loss]).min()
        ax.set_ylim( min_y*0.9, max_y*1.1 )
        ax.set_xlim( steps.min()*0.9, steps.max()*1.1)

        fig.canvas.draw()
        fig.canvas.flush_events()

        plt.draw()


#plt.ioff()  # Turn off interactive mode if you want the plot to stay open
#plt.show()