import argparse
import os
import models
import utils
import torch
import sys
import matplotlib
import matplotlib.pyplot
import signal
import time
import numpy
import collect

MAX_NUM_EPOCHS = 100
BATCH_SIZE = 256
LR_LIST = [0.0005, 0.001, 0.005, 0.01, 0.05]
DATASET = "cifar"
DATASET_DEVICE = "cuda"
θ_LIST = [1, 4, 8]
MODEL_DEVICE = "cuda"
NUM_MODELS = 7

def SIGINT_handler(sigint, frame):
    print("\n✋ SIGINT received")
    exit(1)
signal.signal(signal.SIGINT, SIGINT_handler)

script_path = os.path.abspath(__file__)
src_path = os.path.dirname(script_path)
root_path = os.path.dirname(src_path)
res_path = "%s/res" % (root_path)
cifar_path = "%s/cifar" % (root_path)
imagenet_path = "%s/imagenet" % (root_path)
out_path = "%s/out" % (root_path)
run_path = "%s/%d" % (out_path, int(time.time()))
print("📁 run_path=%s" % (run_path))

print("💾 Loading data")
if DATASET=="cifar":
    cifar_train_X = torch.load("%s/train_X.pt" % (cifar_path), map_location=DATASET_DEVICE)
    cifar_train_Y = torch.load("%s/train_Y.pt" % (cifar_path), map_location=DATASET_DEVICE)
    cifar_train_dataset = torch.utils.data.TensorDataset(cifar_train_X, cifar_train_Y)

    train_dataset, val_dataset = torch.utils.data.random_split(cifar_train_dataset, [len(cifar_train_dataset)-10000, 10000])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

elif DATASET=="imagenet":
    imagenet_train_X = torch.load("%s/train_X.pt" % (imagenet_path), map_location=DATASET_DEVICE)
    # ImageNet labels start from 1
    imagenet_train_Y = torch.load("%s/train_Y.pt" % (imagenet_path), map_location=DATASET_DEVICE)-1
    imagenet_train_dataset = torch.utils.data.TensorDataset(imagenet_train_X, imagenet_train_Y)

    train_dataset, val_dataset = torch.utils.data.random_split(imagenet_train_dataset, [len(imagenet_train_dataset)-50000, 50000])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("%d train samples" % len(train_dataset))
print("%d validation samples" % len(val_dataset))

proxy = models.θNet_cifar(1).to(MODEL_DEVICE)

for θ in θ_LIST:
    print("🏛️  θ=%d" % θ)
    θ_path = "%s/θ=%d" % (run_path, θ)

    for lr in LR_LIST:
        print("🦸 Learning rate=%s" % lr)
        lr_path = "%s/lr=%s" % (θ_path, lr)
        if not os.path.isdir(lr_path):
            os.makedirs(lr_path)

        for model in range(NUM_MODELS):
            print("🧠 Model %d" % model)
            model_path = "%s/model=%d.dat" % (lr_path, model)
            print("\x1b[1mepoch val_loss val_acc train_loss train_acc\x1b[0m")
            with open(model_path,"w") as file:
                file.write("epoch val_loss val_acc train_loss train_acc\n")

            # target = models.θViT_cifar(P=4, L=2, heads=4, θ=θ).to(MODEL_DEVICE)
            target = models.θNet_cifar(θ).to(MODEL_DEVICE)
            # utils.init_SP(target, κ=1/100)
            utils.init_μP(proxy, target, κ=1/10)

            # optimizer = torch.optim.Adam(target.parameters(), lr=lr)
            optimizer = utils.Adam_μP(proxy, target, lr)

            loss_function = torch.nn.NLLLoss()

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

                    val_loss = val_loss_sum/len(val_dataset)
                    val_acc = (val_hits/len(val_dataset))*100

                target.train()
                train_loss_sum = 0
                train_hits = 0
                for batch, (batch_X, batch_Y) in enumerate(train_dataloader):
                    # Forward
                    batch_Y_ = target(batch_X.to(device=MODEL_DEVICE, dtype=torch.float))

                    batch_loss = loss_function(batch_Y_, batch_Y.to(device=MODEL_DEVICE, dtype=torch.long))
                    train_loss_sum += batch_loss.item()*len(batch_X)

                    batch_pred = torch.argmax(batch_Y_, dim=-1)
                    batch_hits = torch.sum( batch_pred==batch_Y.to(MODEL_DEVICE) ).item()
                    train_hits += batch_hits

                    # Backward
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()

                train_loss = train_loss_sum/len(train_dataset)
                train_acc = (train_hits/len(train_dataset))*100

                print("%d %.2f %.2f %.2f %.2f" % (epoch, val_loss, val_acc, train_loss, train_acc))
                with open(model_path,"a") as file:
                    file.write("%d %.2f %.2f %.2f %.2f\n" % (epoch, val_loss, val_acc, train_loss, train_acc))

args = [run_path]
collect.main(args)

os.system("TEXINPUTS=%s: OPENTYPEFONTS=%s: lualatex --shell-escape --output-directory=%s --interaction=batchmode '\def\\runpath{%s}\input{%s/plots.tex}'" % (res_path, res_path, run_path, run_path, res_path))