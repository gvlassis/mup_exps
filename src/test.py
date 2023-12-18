import argparse
import os
import models
import torch

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--batch_size", metavar="INT", help="The number of samples in each batch", type=int, default=256)
parser.add_argument("--dataset", choices=["cifar", "imagenet"], help="Dataset to use", default="cifar")
parser.add_argument("--dataset_device", choices=["cuda","mps","cpu"], help="Device that stores the dataset", default="cpu")
parser.add_argument("--architecture", choices=["θNet", "VGG16"], help="Architecture to use. Exact model depends on the dataset", type=str, default="θNet")
parser.add_argument("--model_arguments", nargs="*", help="Arguments passed to model constructor", type=str, default=[])
parser.add_argument("--model_device", choices=["cuda","mps","cpu"], help="Device that stores the model", default="cpu")
args=parser.parse_args()

script_path = os.path.abspath(__file__)
src_path = os.path.dirname(script_path)
root_path = os.path.dirname(src_path)
res_path = root_path+"/res"
cifar_path = root_path+"/cifar"
imagenet_path = root_path+"/imagenet"
out_path = root_path+"/out"

print("💾 Loading data")
if args.dataset=="cifar":
    test_X = torch.load(cifar_path+"/test_X.pt", map_location=args.dataset_device)
    test_Y = torch.load(cifar_path+"/test_Y.pt", map_location=args.dataset_device)
    test_dataset = torch.utils.data.TensorDataset(test_X, test_Y)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

elif args.dataset=="imagenet":
    test_X = torch.load(imagenet_path+"/val_X.pt", map_location=args.dataset_device)
    # ImageNet labels start from 1
    test_Y = torch.load(imagenet_path+"/val_Y.pt", map_location=args.dataset_device)-1
    test_dataset = torch.utils.data.TensorDataset(imagenet_train_X, imagenet_train_Y)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
print("%d testing samples" % len(test_dataset))

print("🧠 Loading model")
model = getattr(models, args.architecture+"_"+args.dataset)(*args.model_arguments).to(args.model_device)
model.load_state_dict(torch.load(out_path+"/model.pt"))
loss_function = torch.nn.NLLLoss()

model.eval()
with torch.no_grad():
    test_loss_sum = 0
    test_hits = 0
    for batch, (batch_X, batch_Y) in enumerate(test_dataloader):
        batch_Y_ = model(batch_X.to(device=args.model_device, dtype=torch.float))

        batch_loss = loss_function(batch_Y_, batch_Y.to(device=args.model_device, dtype=torch.long))
        test_loss_sum += batch_loss.item()*len(batch_X)

        batch_pred = torch.argmax(batch_Y_, dim=-1)
        batch_hits = torch.sum( batch_pred==batch_Y.to(args.model_device) ).item()
        test_hits += batch_hits

    test_loss = test_loss_sum/len(test_dataset)
    test_acc = test_hits/len(test_dataset)
    print("🏆 test loss=%.2f, test accuracy=\x1b[31m%.2f%%\x1b[0m" % (test_loss, test_acc*100))