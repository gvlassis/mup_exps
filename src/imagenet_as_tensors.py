import argparse
import os
import glob
import numpy
import torch
import torchvision

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("CLS_LOC_path", help="The path of the CLS-LOC directory, which contains the images under train/, val/ and test/", type=os.path.abspath)
parser.add_argument("output_path", help="The path of the directory which will contain train_X, train_Y, val_X, val_Y, test_X", type=os.path.abspath)
parser.add_argument("update_frequency", help="\x1b[1mTrain\x1b[0m images interval between updates", type=int)
parser.add_argument("height", help="The height of images", type=int)
args=parser.parse_args()

script_path = os.path.abspath(__file__)
src_path = os.path.dirname(script_path)
root_path = os.path.dirname(src_path)
res_path = os.path.join(root_path, "res")

# Use NumPy's Structured Datatypes
WNID_to_ILSVRC2015_CLSLOC_ID = dict(numpy.loadtxt(os.path.join(res_path, "map_clsloc.txt"), dtype="U9,i", usecols=(0,1)))
val_Y = torch.tensor(numpy.loadtxt(os.path.join(res_path, "ILSVRC2015_clsloc_validation_ground_truth.txt"), dtype="i"))
torch.save(val_Y,os.path.join(args.output_path,"val_Y.pt"))

train_images_paths = glob.glob("%s/train/**/*.JPEG" % args.CLS_LOC_path, recursive=True)
train_images = len(train_images_paths)
print("📷 There are \x1b[36m%d\x1b[0m training images" % train_images)
train_X = torch.empty((train_images, 3, args.height, args.height), dtype=torch.uint8)
train_Y = torch.empty((train_images), dtype=torch.int)
for image, image_path in enumerate(train_images_paths):
    if image%args.update_frequency==0:
        print("🍻 \x1b[96m%d/%d\x1b[0m" % (image, train_images-1))

    train_X[image] = torchvision.io.read_image(image_path)
    WNID = os.path.basename(image_path).split("_")[0]
    train_Y[image] = WNID_to_ILSVRC2015_CLSLOC_ID[WNID]
torch.save(train_X,os.path.join(args.output_path,"train_X.pt"))
torch.save(train_Y,os.path.join(args.output_path,"train_Y.pt"))

val_images_paths = glob.glob("%s/val/*.JPEG" % args.CLS_LOC_path, recursive=True)
val_images = len(val_images_paths)
print("📷 There are \x1b[36m%d\x1b[0m validation images" % val_images)
val_X = torch.empty((val_images, 3, args.height, args.height), dtype=torch.uint8)
for image, image_path in enumerate(val_images_paths):
    val_X[image] = torchvision.io.read_image(image_path)
torch.save(val_X,os.path.join(args.output_path,"val_X.pt"))

test_images_paths = glob.glob("%s/test/*.JPEG" % args.CLS_LOC_path, recursive=True)
test_images = len(test_images_paths)
print("📷 There are \x1b[36m%d\x1b[0m test images" % test_images)
test_X = torch.empty((test_images, 3, args.height, args.height), dtype=torch.uint8)
for image_path in test_images_paths:
    test_X[image] = torchvision.io.read_image(image_path)
torch.save(test_X,os.path.join(args.output_path,"test_X.pt"))

print("🍾 Done!")