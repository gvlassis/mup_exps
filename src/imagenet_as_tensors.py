import argparse
import os
import glob
import numpy
import torch
import torchvision

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("CLS_LOC_path", help="The path of the CLS-LOC directory, which contains the images under train/, val/ and test/", type=os.path.abspath)
parser.add_argument("output_path", help="The path of the directory which will contain train_X, train_Y, val_X, val_Y, test_X")
args=parser.parse_args()

script_path = os.path.abspath(__file__)
src_path = os.path.dirname(script_path)
root_path = os.path.dirname(src_path)
res_path = os.path.join(root_path, "res")

# Use NumPy's Structured Datatypes
WNID_to_ILSVRC2015_CLSLOC_ID = dict(numpy.loadtxt(os.path.join(res_path, "map_clsloc.txt"), dtype="U9,i", usecols=(0,1)))
val_Y = torch.tensor(numpy.loadtxt(os.path.join(res_path, "ILSVRC2015_clsloc_validation_ground_truth.txt")))
torch.save(val_Y,os.path.join(args.output_path,"val_Y.pt"))

train_images_paths = glob.glob("%s/train/**/*.JPEG" % args.CLS_LOC_path, recursive=True)
print("📷 There are \x1b[36m%d\x1b[0m train images" % len(train_images_paths))
train_X_list = []
train_Y_list = []
for image_path in train_images_paths:
    train_X_list.append(torchvision.io.read_image(image_path))
    WNID = os.path.basename(image_path).split("_")[0]
    train_Y_list.append(WNID_to_ILSVRC2015_CLSLOC_ID[WNID])
train_X = torch.stack(train_X_list)
train_Y = torch.tensor(train_Y_list)
torch.save(train_X,os.path.join(args.output_path,"train_X.pt"))
torch.save(train_Y,os.path.join(args.output_path,"train_Y.pt"))

val_images_paths = glob.glob("%s/val/*.JPEG" % args.CLS_LOC_path, recursive=True)
print("📷 There are \x1b[36m%d\x1b[0m validation images" % len(val_images_paths))
val_X_list = []
for image_path in val_images_paths:
    val_X_list.append(torchvision.io.read_image(image_path))
val_X = torch.stack(val_X_list)
torch.save(val_X,os.path.join(args.output_path,"val_X.pt"))

test_images_paths = glob.glob("%s/test/*.JPEG" % args.CLS_LOC_path, recursive=True)
print("📷 There are \x1b[36m%d\x1b[0m test images" % len(test_images_paths))
test_X_list = []
for image_path in test_images_paths:
    test_X_list.append(torchvision.io.read_image(image_path))
test_X = torch.stack(test_X_list)
torch.save(test_X,os.path.join(args.output_path,"test_X.pt"))