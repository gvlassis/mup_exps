import argparse
import os
import models
import utils
import torch
import sys
import matplotlib
import matplotlib.pyplot
import signal
import numpy

MAX_NUM_EPOCHS = 300
BATCH_SIZE = 256
LEARNING_RATE_LIST = [0.0005, 0.001, 0.005, 0.01, 0.05]
DATASET = "cifar"
DATASET_DEVICE = "cuda"
θ_LIST = [1, 4, 8]
MODEL_DEVICE = "cuda"
NUM_MODELS = 10

def SIGINT_handler(sigint, frame):
    print("\n✋ SIGINT received")
    exit(1)
signal.signal(signal.SIGINT, SIGINT_handler)

script_path = os.path.abspath(__file__)
src_path = os.path.dirname(script_path)
root_path = os.path.dirname(src_path)
res_path = root_path+"/res"
cifar_path = root_path+"/cifar"
imagenet_path = root_path+"/imagenet"
out_path = root_path+"/out"

matplotlib.rc_file(res_path+"/matplotlibrc")
matplotlib.pyplot.style.use(res_path+"/blackberry_dark.mplstyle")
matplotlib.font_manager.fontManager.addfont(res_path+"/FiraGO-Regular.otf")
matplotlib.font_manager.fontManager.addfont(res_path+"/FiraGO-Bold.otf")
figure = matplotlib.figure.Figure()
figure_gridspec = figure.add_gridspec(nrows=1,ncols=1)
axes = figure.add_subplot(figure_gridspec[0,0])

print("💾 Loading data")
if DATASET=="cifar":
    cifar_train_X = torch.load(cifar_path+"/train_X.pt", map_location=DATASET_DEVICE)
    cifar_train_Y = torch.load(cifar_path+"/train_Y.pt", map_location=DATASET_DEVICE)
    cifar_train_dataset = torch.utils.data.TensorDataset(cifar_train_X, cifar_train_Y)

    train_dataset, val_dataset = torch.utils.data.random_split(cifar_train_dataset, [len(cifar_train_dataset)-10000, 10000])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

elif DATASET=="imagenet":
    imagenet_train_X = torch.load(imagenet_path+"/train_X.pt", map_location=DATASET_DEVICE)
    # ImageNet labels start from 1
    imagenet_train_Y = torch.load(imagenet_path+"/train_Y.pt", map_location=DATASET_DEVICE)-1
    imagenet_train_dataset = torch.utils.data.TensorDataset(imagenet_train_X, imagenet_train_Y)

    train_dataset, val_dataset = torch.utils.data.random_split(imagenet_train_dataset, [len(imagenet_train_dataset)-50000, 50000])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("%d train samples" % len(train_dataset))
print("%d validation samples" % len(val_dataset))

proxy = models.θNet_cifar(1).to(MODEL_DEVICE)

for θ in θ_LIST:
    print("🏛️  θ=%d" % θ)
    mean_list = []
    std_list = []
    for learning_rate in LEARNING_RATE_LIST:
        print("🦸 Learning rate=%.5f" % learning_rate)
        model_list = []
        for no_model in range(NUM_MODELS):
            print("🧠 Model %d" % no_model, end="")

            target = models.θNet_cifar(θ).to(MODEL_DEVICE)
            utils.init_μP(proxy, target)

            optimizer = utils.Adam_μP(proxy, target, learning_rate)
            loss_function = torch.nn.NLLLoss()

            # Beginning-of-epoch
            val_loss_list = []
            val_acc_list = []
            for epoch in range(MAX_NUM_EPOCHS):
                target.eval()
                with torch.no_grad():
                    val_loss_sum = 0
                    val_hits = 0
                    for batch, (batch_X, batch_Y) in enumerate(val_dataloader):
                        batch_Y_ = target(batch_X.to(device=MODEL_DEVICE, dtype=torch.float))

                        batch_loss = loss_function(batch_Y_, batch_Y.to(device=MODEL_DEVICE, dtype=torch.long))
                        val_loss_sum += batch_loss.item()*len(batch_X)

                        batch_pred = torch.argmax(batch_Y_, dim=-1)
                        batch_hits = torch.sum( batch_pred==batch_Y.to(MODEL_DEVICE) ).item()
                        val_hits += batch_hits

                    val_loss_list.append(val_loss_sum/len(val_dataset))
                    val_acc_list.append(val_hits/len(val_dataset))

                target.train()
                for batch, (batch_X, batch_Y) in enumerate(train_dataloader):
                    # Forward
                    batch_Y_ = target(batch_X.to(device=MODEL_DEVICE, dtype=torch.float))

                    batch_loss = loss_function(batch_Y_, batch_Y.to(device=MODEL_DEVICE, dtype=torch.long))

                    # Backward
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()

            model_list.append(min(val_loss_list))
            print(", validation loss=%.2f" % (model_list[-1]))

        mean_list.append(numpy.mean(model_list))
        std_list.append(numpy.std(model_list))

    axes.plot(LEARNING_RATE_LIST, mean_list, marker="o", label="θ=%d" % (θ))
    axes.fill_between(LEARNING_RATE_LIST, numpy.array(mean_list)-numpy.array(std_list), numpy.array(mean_list)+numpy.array(std_list), alpha=3/8)
    axes.grid(True)
    axes.legend()
    axes.set_xlabel("Learning rate")
    axes.set_ylabel("Validation loss")
    axes.set_xscale("log",base=2)

    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    figure.savefig(out_path+"/figure.pdf")
