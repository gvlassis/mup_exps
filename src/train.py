import argparse
import os
import models
import torch
import matplotlib
import matplotlib.pyplot
import signal

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--theta", metavar="INT", help="The width scale parameter θ", type=int, default=1)
parser.add_argument("--max_num_epochs", metavar="INT", help="The maximum number of epochs (you can safely stop prematurely by sending SIGINT)", type=int, default=100)
parser.add_argument("--batch_size", metavar="INT", help="The number of samples in each batch", type=int, default=256)
# Use PyTorch defaults for ADAM
parser.add_argument("--learning_rate", metavar="FLOAT", help="Learning rate of ADAM", type=float, default=0.001)
parser.add_argument("--beta1", metavar="FLOAT", help="ADAM's β1", type=float, default=0.9)
parser.add_argument("--beta2", metavar="FLOAT", help="ADAM's β2", type=float, default=0.999)
parser.add_argument("--datasets_directory", metavar="PATH", help="The path of the directory of train_X.pt, train_Y.pt, val_X.pt, val_Y.pt, test_Y.pt", type=os.path.abspath)
parser.add_argument("--model_directory", metavar="PATH", help="The path of the directory where model. will by saved", type=os.path.abspath)
parser.add_argument("--update_frequency", metavar="INT", help="Every how many batches the batch_loss will be printed", type=int, default=500)
parser.add_argument("--device", choices=["cuda","mps","cpu"], help="Device used for training", default="cpu")
args=parser.parse_args()

def SIGINT_handler(sigint, frame):
    print("\n👻 SIGINT received")
    exit(1)
signal.signal(signal.SIGINT, SIGINT_handler)

script_path = os.path.abspath(__file__)
src_path = os.path.dirname(script_path)
root_path = os.path.dirname(src_path)
res_path = root_path+"/res"
out_path = root_path+"/out"

if args.device=="cuda":
    if torch.cuda.is_available():
        print("\x1b[32m🖥️  cuda specified and found\x1b[0m")
        device="cuda"
    else:
        print("\x1b[31m🖥️  cuda specified but not found. Falling back to cpu\x1b[0m")
        device="cpu"
elif args.device=="mps":
    if torch.backends.mps.is_available():
        print("\x1b[32m🖥️  mps specified and found\x1b[0m")
        device="mps"
    else:
        print("\x1b[31m🖥️  mps specified but not found. Falling back to cpu\x1b[0m")
        device="cpu"
else:
    print("\x1b[32m🖥️  Using cpu\x1b[0m")
    device="cpu"

matplotlib.rc_file(res_path+"/matplotlibrc")
matplotlib.pyplot.style.use(res_path+"/blackberry_dark.mplstyle")
training_curves = matplotlib.figure.Figure()
training_curves.suptitle("Training curves")
training_curves_gridspec = training_curves.add_gridspec(nrows=1,ncols=2)
loss_curves = training_curves.add_subplot(training_curves_gridspec[0,0], xlabel="epochs", ylabel="loss")
acc_curves = training_curves.add_subplot(training_curves_gridspec[0,1], xlabel="epochs", ylabel="accuracy")

print("🧠 Initializing model")
# model = models.θNet(args.theta).to(device)
model = models.VGG16().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
loss_function = torch.nn.NLLLoss()

print("💾 Loading data")
train_X=torch.load(args.datasets_directory+"/train_X.pt", map_location=device)
train_Y=torch.load(args.datasets_directory+"/train_Y.pt", map_location=device)
# Use 50000 images of ImageNet's training dataset as validation split (ImageNet's validation dataset will be subsequently used as testing split)
train_dataset=torch.utils.data.TensorDataset(train_X[:-50000],train_Y[:-50000])
train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
train_val_X=train_X[-50000:]
train_val_Y=train_Y[-50000:]

train_val_loss_list = [] # Will contain the beginning-of-epoch loss on the validation split of the training dataset
train_val_acc_list = [] # Will contain the beginning-of-epoch accuracy on the validation split of the training dataset
train_train_max_loss_list = [] # Will contain the end-of-epoch max loss on the training split of the training dataset
train_train_min_loss_list = [] # Will contain the end-of-epoch min loss on the training split of the training dataset
train_train_max_acc_list = [] # Will contain the end-of-epoch max accuracy on the training split of the training dataset
train_train_min_acc_list = [] # Will contain the end-of-epoch min accuracy on the training split of the training dataset
epoch_list = []
for epoch in range(args.max_num_epochs):
    epoch_list.append(epoch)

    model.eval()
    with torch.no_grad():
        # train_val_X is torch.uint8, so you have to convert it to torch.float first
        train_val_Y_ = model(train_val_X.float())
        train_val_loss = loss_function(train_val_Y_, (train_val_Y-1).long())
        train_val_loss_list.append(train_val_loss.item())
        train_val_acc = (torch.sum(torch.argmax(train_val_Y_, dim=-1)+1==train_val_Y)/train_val_Y.shape[0]).item()
        train_val_acc_list.append(train_val_acc)
        print("🕑 Epoch %d/%d, validation loss=%.2f, validation accuracy=\x1b[31m%.2f%%\x1b[0m" % (epoch, args.max_num_epochs, train_val_loss.item(), train_val_acc*100))
        loss_curves.cla()
        loss_curves.plot(epoch_list, train_val_loss_list, color="C0", marker="o", label="beginning-of-epoch validation loss")
        acc_curves.cla()
        acc_curves.plot(epoch_list, train_val_acc_list, color="C0", marker="o", label="beginning-of-epoch validation accuracy")

    model.train()
    batch_loss_list = []
    batch_acc_list = []
    for batch, (batch_X, batch_Y) in enumerate(train_dataloader):
        # Forward
        # batch_X is torch.uint8, so you have to convert it to torch.float first
        batch_Y_ = model(batch_X.float())
        batch_loss = loss_function(batch_Y_, (batch_Y-1).long())
        batch_loss_list.append(batch_loss.item())
        batch_acc = (torch.sum(torch.argmax(batch_Y_, dim=-1)+1==batch_Y)/batch_Y.shape[0]).item()
        batch_acc_list.append(batch_acc)
        if (batch%args.update_frequency)==0:
            print("Batch %d/%d, batch loss=%.2f, batch accuracy=\x1b[35m%.2f%%\x1b[0m" % (batch, len(train_dataloader), batch_loss.item(), batch_acc*100))

        # Backward
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
    train_train_max_loss_list.append(max(batch_loss_list))
    train_train_min_loss_list.append(min(batch_loss_list))
    train_train_max_acc_list.append(max(batch_acc_list))
    train_train_min_acc_list.append(min(batch_acc_list))
    loss_curves.fill_between(epoch_list, train_train_min_loss_list, train_train_max_loss_list, facecolor="C1", alpha=3/8, label="Max/Min training loss")
    acc_curves.fill_between(epoch_list, train_train_min_acc_list, train_train_max_acc_list, facecolor="C1", alpha=3/8, label="Max/Min training accuracy")
    loss_curves.grid()
    loss_curves.legend()
    acc_curves.grid()
    acc_curves.legend()
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    training_curves.savefig(out_path+"/training_curves.pdf")
