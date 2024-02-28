import argparse
import os
import numpy
import sys

def main(args):
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("RUN_PATH", help="e.g. ./out/1708557689", type=os.path.abspath)
	args=parser.parse_args(args)

	θ_LIST = sorted([int(child.split("=")[-1]) for child in os.listdir(args.RUN_PATH) if os.path.isdir("%s/%s"  % (args.RUN_PATH,child))])
	LR_LISTS = [sorted([float(child.split("=")[-1]) for child in os.listdir("%s/θ=%d" % (args.RUN_PATH,θ)) if os.path.isdir("%s/θ=%d/%s" % (args.RUN_PATH,θ,child))]) for θ in θ_LIST]
	MODEL_LISTS = [[sorted([int(child.split("=")[-1].split(".dat")[0]) for child in os.listdir("%s/θ=%d/lr=%s" % (args.RUN_PATH,θ,lr)) if os.path.isfile("%s/θ=%d/lr=%s/%s" % (args.RUN_PATH,θ,lr,child))]) for lr in LR_LISTS[i]] for i,θ in enumerate(θ_LIST)]

	for i,θ in enumerate(θ_LIST):
		print("🏛️  θ=%d" % θ)
		θ_path = "%s/θ=%d.dat" % (args.RUN_PATH, θ)

		print("\x1b[1mlr mean_best_val_loss top_best_val_loss bot_best_val_loss mean_best_val_acc top_best_val_acc bot_best_val_acc\x1b[0m")
		with open(θ_path,"w") as file:
			file.write("lr mean_best_val_loss top_best_val_loss bot_best_val_loss mean_best_val_acc top_best_val_acc bot_best_val_acc\n")

		for j,lr in enumerate(LR_LISTS[i]):
			lr_path = "%s/θ=%d/lr=%s" % (args.RUN_PATH, θ, lr)

			best_val_loss_list = []
			best_val_acc_list = []
			for _,model in enumerate(MODEL_LISTS[i][j]):
				model_path = "%s/model=%d.dat" % (lr_path, model)
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

			print("%s %.2f %.2f %.2f %.2f %.2f %.2f" % (lr, mean_best_val_loss, top_best_val_loss, bot_best_val_loss, mean_best_val_acc, top_best_val_acc, bot_best_val_acc))
			with open(θ_path,"a") as file:
				file.write("%s %.2f %.2f %.2f %.2f %.2f %.2f\n" % (lr, mean_best_val_loss, top_best_val_loss, bot_best_val_loss, mean_best_val_acc, top_best_val_acc, bot_best_val_acc))

if __name__ == "__main__":
	main(sys.argv[1:])
