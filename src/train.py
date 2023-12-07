import argparse
import os
import models
import torch

LOSS_FUNCTION = torch.nn.NLLLoss()
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

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
args=parser.parse_args()

script_path = os.path.abspath(__file__)
src_path = os.path.dirname(script_path)
root_path = os.path.dirname(src_path)
out_path = os.path.join(root_path, "out")

print("🖥️  DEVICE=%s (Default: cuda>mps>cpu)" % DEVICE)

print("🧠 Initializing model")

model = models.θNet(args.theta).to(DEVICE)


init_mult = 1

input_layer = model.layer1
hidden_layers = [model.layer2,model.layer3,model.layer4,model.layer5,model.layer6,model.layer7]
output_layer = model.layer8
for layer in [input_layer] + hidden_layers:
    muP_init_input_hidden( layer, init_mult )
muP_init_output( output_layer, init_mult )

# Create your optimizer with these parameter group
# model = models.VGG16().to(DEVICE)

# Standard ADAM

# OPTIMIZER = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))

lr_mult = 1

# muP SGD
# Create parameter groups
param_groups = [
    {"params": input_layer.parameters() , "lr": lr_mult*input_layer.out_features },
    {"params": model.layer2.parameters(), "lr": lr_mult},
    {"params": model.layer3.parameters(), "lr": lr_mult},
    {"params": model.layer4.parameters(), "lr": lr_mult},
    {"params": model.layer5.parameters(), "lr": lr_mult},
    {"params": model.layer6.parameters(), "lr": lr_mult},
    {"params": model.layer7.parameters(), "lr": lr_mult},
    {"params": output_layer.parameters(), "lr": lr_mult*output_layer.out_features}
]

OPTIMIZER = torch.optim.SGD(param_groups, momentum=0.9)

print("💾 Loading data")
train_X=torch.load(os.path.join(args.datasets_directory,"train_X.pt"), map_location=DEVICE)
train_Y=torch.load(os.path.join(args.datasets_directory,"train_Y.pt"), map_location=DEVICE)
train_dataset=torch.utils.data.TensorDataset(train_X,train_Y)
train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
# val_X=torch.load(os.path.join(args.datasets_directory,"val_X.pt"), map_location=DEVICE)
# val_Y=torch.load(os.path.join(args.datasets_directory,"val_Y.pt"), map_location=DEVICE)

for epoch in range(args.max_num_epochs):
    # val_Y_ = model(val_X.float())
    # val_loss = LOSS_FUNCTION(val_Y_, (val_Y-1).long())
    # val_acc = (torch.sum(torch.argmax(val_Y_, dim=-1)==(val_Y-1))/val_Y.shape[0]).item()
    # print("🕑 Epoch %d/%d, validation loss=%.2f, validation accuracy=\x1b[31m%.2f%%\x1b[0m" % (epoch, args.max_num_epochs, val_loss, val_acc*100))

    print("🕑 Epoch %d/%d" % (epoch, args.max_num_epochs))

    for batch, (batch_X, batch_Y) in enumerate(train_dataloader):
        # Forward
        # batch_X is torch.uint8, so you have to convert it to torch.float first
        batch_Y_ = model(batch_X.float())
        batch_loss = LOSS_FUNCTION(batch_Y_, (batch_Y-1).long())
        batch_acc = (torch.sum(torch.argmax(batch_Y_, dim=-1)+1==batch_Y)/batch_Y.shape[0]).item()
        if (batch%args.update_frequency)==0:
            print("Batch %d/%d, batch loss=%.2f, batch accuracy=\x1b[35m%.2f%%\x1b[0m" % (batch, len(train_dataloader), batch_loss.item(), batch_acc*100))

        # Backward
        OPTIMIZER.zero_grad()
        batch_loss.backward()
        OPTIMIZER.step()
