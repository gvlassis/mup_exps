import argparse
import os
import pickle
import torch

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("cifar_10_batches_py", help="The cifar-10-batches-py directory, which contains the images in data_batch_[1-5] and test_batch", type=os.path.abspath)
parser.add_argument("out_path", help="The directory which will contain train_X, train_Y, test_X, test_Y", type=os.path.abspath)
args=parser.parse_args()

print("💾 Loading data")
with open(args.cifar_10_batches_py+"/data_batch_1", "rb") as data_batch_1:
	train_dict1 = pickle.load(data_batch_1, encoding="bytes")
train1_X = torch.unflatten(torch.as_tensor(train_dict1[b'data']), dim=-1, sizes=(3,32,32))
train1_Y = torch.as_tensor(train_dict1[b'labels'])

with open(args.cifar_10_batches_py+"/data_batch_2", "rb") as data_batch_2:
	train_dict2 = pickle.load(data_batch_2, encoding="bytes")
train2_X = torch.unflatten(torch.as_tensor(train_dict2[b'data']), dim=-1, sizes=(3,32,32))
train2_Y = torch.as_tensor(train_dict2[b'labels'])

with open(args.cifar_10_batches_py+"/data_batch_3", "rb") as data_batch_3:
	train_dict3 = pickle.load(data_batch_3, encoding="bytes")
train3_X = torch.unflatten(torch.as_tensor(train_dict3[b'data']), dim=-1, sizes=(3,32,32))
train3_Y = torch.as_tensor(train_dict3[b'labels'])

with open(args.cifar_10_batches_py+"/data_batch_4", "rb") as data_batch_4:
	train_dict4 = pickle.load(data_batch_4, encoding="bytes")
train4_X = torch.unflatten(torch.as_tensor(train_dict4[b'data']), dim=-1, sizes=(3,32,32))
train4_Y = torch.as_tensor(train_dict4[b'labels'])

with open(args.cifar_10_batches_py+"/data_batch_5", "rb") as data_batch_5:
	train_dict5 = pickle.load(data_batch_5, encoding="bytes")
train5_X = torch.unflatten(torch.as_tensor(train_dict5[b'data']), dim=-1, sizes=(3,32,32))
train5_Y = torch.as_tensor(train_dict5[b'labels'])

with open(args.cifar_10_batches_py+"/test_batch", "rb") as test_batch:
	test_dict = pickle.load(test_batch, encoding="bytes")
test_X = torch.unflatten(torch.as_tensor(test_dict[b'data']), dim=-1, sizes=(3,32,32))
test_Y = torch.as_tensor(test_dict[b'labels'])

print("💾 Saving data")
if not os.path.isdir(args.out_path):
        os.makedirs(args.out_path)
torch.save(torch.cat((train1_X, train2_X, train3_X, train4_X, train5_X)),args.out_path+"/train_X.pt")
torch.save(torch.cat((train1_Y, train2_Y, train3_Y, train4_Y, train5_Y)),args.out_path+"/train_Y.pt")
torch.save(test_X,args.out_path+"/test_X.pt")
torch.save(test_Y,args.out_path+"/test_Y.pt")