import argparse
import os
import numpy
import sys

def main(args):
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("RUN_PATH", help="e.g. ./out/1708557689", type=os.path.abspath)
	args=parser.parse_args(args)

	θ_DIRS_LIST = [args.RUN_PATH+"/"+child for child in os.listdir(args.RUN_PATH) if os.path.isdir(args.RUN_PATH+"/"+child)]
	LR_DIRS_LISTS = [[θ_dir+"/"+child for child in os.listdir(θ_dir) if os.path.isdir(θ_dir+"/"+child)] for θ_dir in θ_DIRS_LIST]
	MODEL_PATHS_LISTS = [[[lr_dir+"/"+child for child in os.listdir(lr_dir) if os.path.isfile(lr_dir+"/"+child)] for lr_dir in LR_DIRS_LISTS[i]] for i,_ in enumerate(θ_DIRS_LIST)]

	for i,θ_dir in enumerate(θ_DIRS_LIST):
		θ = int(θ_dir.split("θ=")[-1])
		print("🏛️  θ=%d" % θ)

		θ_path = args.RUN_PATH+("/θ=%d.dat" % θ)

		print("\x1b[1mlr mean_best_val_loss top_best_val_loss bot_best_val_loss mean_best_val_acc top_best_val_acc bot_best_val_acc\x1b[0m")
		with open(θ_path,"w") as file:
			file.write("lr mean_best_val_loss top_best_val_loss bot_best_val_loss mean_best_val_acc top_best_val_acc bot_best_val_acc\n")

		for j,lr_dir in enumerate(LR_DIRS_LISTS[i]):
			lr = float(lr_dir.split("lr=")[-1])

			best_val_loss_list = []
			best_val_acc_list = []
			for _,model_path in enumerate(MODEL_PATHS_LISTS[i][j]):
				with open(model_path,"r") as file:
					# Skip header
					header = file.readline()
					best_val_loss = float("inf")
					for line in file:
						cols = line.rstrip().split(' ')
						epoch, val_loss, val_acc, train_loss, train_acc = int(cols[0]), float(cols[1]), float(cols[2]), float(cols[3]), float(cols[4])
						if best_val_loss>val_loss:
							best_val_loss = val_loss
							best_val_acc = val_acc

				best_val_loss_list.append(best_val_loss)
				best_val_acc_list.append(best_val_acc)

			mean_best_val_loss = numpy.mean(best_val_loss_list)
			std_best_val_loss = numpy.std(best_val_loss_list)
			top_best_val_loss = mean_best_val_loss+std_best_val_loss
			bot_best_val_loss = mean_best_val_loss-std_best_val_loss

			mean_best_val_acc = numpy.mean(best_val_acc_list)
			std_best_val_acc = numpy.std(best_val_acc_list)
			top_best_val_acc = mean_best_val_acc+std_best_val_acc
			bot_best_val_acc = mean_best_val_acc-std_best_val_acc

			print("%.5f %.2f %.2f %.2f %.2f %.2f %.2f" % (lr, mean_best_val_loss, top_best_val_loss, bot_best_val_loss, mean_best_val_acc, top_best_val_acc, bot_best_val_acc))
			with open(θ_path,"a") as file:
				file.write("%.5f %.2f %.2f %.2f %.2f %.2f %.2f\n" % (lr, mean_best_val_loss, top_best_val_loss, bot_best_val_loss, mean_best_val_acc, top_best_val_acc, bot_best_val_acc))

if __name__ == "__main__":
	main(sys.argv[1:])
