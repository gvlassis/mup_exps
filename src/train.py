import argparse
import os
import models
import torch
import matplotlib
import matplotlib.pyplot
import signal

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--max_num_epochs", metavar="INT", help="The maximum number of epochs (you can safely stop prematurely by sending SIGINT)", type=int, default=100)
parser.add_argument("--batch_size", metavar="INT", help="The number of samples in each batch", type=int, default=256)
parser.add_argument("--update_frequency", metavar="INT", help="Every how many batches the training loss will be printed", type=int, default=500)
# Use PyTorch defaults for ADAM
parser.add_argument("--learning_rate", metavar="FLOAT", help="Learning rate of ADAM", type=float, default=0.001)
parser.add_argument("--beta1", metavar="FLOAT", help="ADAM's β1", type=float, default=0.9)
parser.add_argument("--beta2", metavar="FLOAT", help="ADAM's β2", type=float, default=0.999)
parser.add_argument("--dataset", choices=["cifar", "imagenet"], help="Dataset to use", default="cifar")
parser.add_argument("--dataset_device", choices=["cuda","mps","cpu"], help="Device that stores the dataset", default="cpu")
parser.add_argument("--architecture", choices=["θNet", "VGG16"], help="Architecture to use. Exact model depends on the dataset", type=str, default="θNet")
parser.add_argument("--model_arguments", nargs="*", help="Arguments passed to model constructor", type=str, default=[])
parser.add_argument("--model_device", choices=["cuda","mps","cpu"], help="Device that stores the model", default="cpu")
args=parser.parse_args()

def SIGINT_handler(sigint, frame):
    print("\n👻 SIGINT received")
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
training_curves = matplotlib.figure.Figure()
training_curves.suptitle("Training curves")
training_curves_gridspec = training_curves.add_gridspec(nrows=1,ncols=2)
loss_curves = training_curves.add_subplot(training_curves_gridspec[0,0], xlabel="epochs", ylabel="loss")
acc_curves = training_curves.add_subplot(training_curves_gridspec[0,1], xlabel="epochs", ylabel="accuracy")

print("💾 Loading data")
if args.dataset=="cifar":
    cifar_train_X = torch.load(cifar_path+"/train_X.pt", map_location=args.dataset_device)
    cifar_train_Y = torch.load(cifar_path+"/train_Y.pt", map_location=args.dataset_device)
    cifar_train_dataset = torch.utils.data.TensorDataset(cifar_train_X, cifar_train_Y)

    train_dataset, val_dataset = torch.utils.data.random_split(cifar_train_dataset, [len(cifar_train_dataset)-10000, 10000])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

elif args.dataset=="imagenet":
    imagenet_train_X = torch.load(imagenet_path+"/train_X.pt", map_location=args.dataset_device)
    # ImageNet labels start from 1
    imagenet_train_Y = torch.load(imagenet_path+"/train_Y.pt", map_location=args.dataset_device)-1
    imagenet_train_dataset = torch.utils.data.TensorDataset(imagenet_train_X, imagenet_train_Y)

    train_dataset, val_dataset = torch.utils.data.random_split(imagenet_train_dataset, [len(imagenet_train_dataset)-50000, 50000])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
print("%d training samples" % len(train_dataset))
print("%d validation samples" % len(val_dataset))

print("🧠 Initializing model")
model = getattr(models, args.architecture+"_"+args.dataset)(*args.model_arguments).to(args.model_device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
loss_function = torch.nn.NLLLoss()

epoch_list = []
# Beginning-of-epoch
val_loss_max_list = []
val_loss_min_list = []
val_acc_max_list = []
val_acc_min_list = []
# End-of-epoch
train_loss_max_list = []
train_loss_min_list = []
train_acc_max_list = []
train_acc_min_list = []
for epoch in range(args.max_num_epochs):
    epoch_list.append(epoch)

    model.eval()
    batch_loss_list = []
    batch_acc_list = []
    with torch.no_grad():
        for batch, (batch_X, batch_Y) in enumerate(train_dataloader):
            batch_Y_ = model(batch_X.to(device=args.model_device, dtype=torch.float))

            batch_loss = loss_function(batch_Y_, batch_Y.to(device=args.model_device, dtype=torch.long))
            batch_loss_list.append(batch_loss.item())

            batch_pred = torch.argmax(batch_Y_, dim=-1)
            batch_hits = torch.sum( batch_pred==batch_Y.to(args.model_device) )
            batch_acc = (batch_hits/len(batch_X)).item()
            batch_acc_list.append(batch_acc)
        val_loss_max_list.append(max(batch_loss_list))
        val_loss_min_list.append(min(batch_loss_list))
        val_acc_max_list.append(max(batch_acc_list))
        val_acc_min_list.append(min(batch_acc_list))

        print("🕑 Epoch %d/%d, validation loss∈[%.2f,%.2f], validation accuracy∈[\x1b[31m%.2f%%\x1b[0m,\x1b[31m%.2f%%\x1b[0m]" % (epoch, args.max_num_epochs, val_loss_min_list[-1], val_loss_max_list[-1], val_acc_min_list[-1]*100, val_acc_max_list[-1]*100))

        loss_curves.cla()
        acc_curves.cla()

        loss_curves.fill_between(epoch_list, val_loss_min_list, val_loss_max_list, facecolor="C0", alpha=3/8, label="Validation")
        acc_curves.fill_between(epoch_list, val_acc_min_list, val_acc_max_list, facecolor="C0", alpha=3/8, label="Validation")

    model.train()
    batch_loss_list = []
    batch_acc_list = []
    for batch, (batch_X, batch_Y) in enumerate(train_dataloader):
        # Forward
        batch_Y_ = model(batch_X.to(device=args.model_device, dtype=torch.float))

        batch_loss = loss_function(batch_Y_, batch_Y.to(device=args.model_device, dtype=torch.long))
        batch_loss_list.append(batch_loss.item())

        batch_pred = torch.argmax(batch_Y_, dim=-1)
        batch_hits = torch.sum( batch_pred==batch_Y.to(args.model_device) )
        batch_acc = (batch_hits/len(batch_X)).item()
        batch_acc_list.append(batch_acc)

        if (batch%args.update_frequency)==0:
            print("Batch %d/%d, batch loss=%.2f, batch accuracy=\x1b[35m%.2f%%\x1b[0m" % (batch, len(train_dataloader), batch_loss_list[-1], batch_acc_list[-1]*100))

        # Backward
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
    train_loss_max_list.append(max(batch_loss_list))
    train_loss_min_list.append(min(batch_loss_list))
    train_acc_max_list.append(max(batch_acc_list))
    train_acc_min_list.append(min(batch_acc_list))

    loss_curves.fill_between(epoch_list, train_loss_min_list, train_loss_max_list, facecolor="C1", alpha=3/8, label="Training")
    acc_curves.fill_between(epoch_list, train_acc_min_list, train_acc_max_list, facecolor="C1", alpha=3/8, label="Training")

    loss_curves.grid()
    loss_curves.legend()
    acc_curves.grid()
    acc_curves.legend()

    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    training_curves.savefig(out_path+"/training_curves.pdf")
