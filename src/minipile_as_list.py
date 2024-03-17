import numpy
import os
import argparse
import tokenizers
import datasets
datasets.logging.set_verbosity_error()
import pickle

def encode(batch):
	doc = batch["text"][0]
	ids = numpy.array(tokenizer.encode(doc).ids+[eot_id], dtype=dtype)
	blocks = (len(ids)//(args.block_length))+1
	ids_list = [ids[block*args.block_length:(block+1)*args.block_length] for block in range(blocks)]
	len_list = [len(i) for i in ids_list]
	return {"ids": ids_list, "len": len_list}

script_path = os.path.abspath(__file__)
src_path = os.path.dirname(script_path)
root_path = os.path.dirname(src_path)
minipile_path = root_path+"/minipile"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--out_path", metavar="PATH", help="The directory which will contain train_X.pickle, val_X.pickle, test_X.pickle", type=os.path.abspath, default=minipile_path)
parser.add_argument("--tokenizer", metavar="ID", help="Tokenizer from Hugging Face", type=str, default="gpt2")
parser.add_argument("--block_length", metavar="INT", help="", type=int, default="1024")
args=parser.parse_args()

if not os.path.isdir(args.out_path):
    os.makedirs(args.out_path)

tokenizer = tokenizers.Tokenizer.from_pretrained(args.tokenizer)
tokenizer.encode_special_tokens = False
eot_id = tokenizer.get_vocab()["<|endoftext|>"]
vocab_size = tokenizer.get_vocab_size()
print("📖 vocab_size=%d" % vocab_size)
if vocab_size<=256:
	dtype = numpy.uint8
elif vocab_size<=65536:
	dtype = numpy.uint16
else:
	dtype = numpy.uint32

cores = os.cpu_count()
print("🖥️  %d cores found" % cores)

train_dataset = datasets.load_dataset("JeanKaddour/minipile", split="train", trust_remote_code=True)
print("📄 \x1b[36m%d\x1b[0m training documents" % train_dataset.num_rows)
# batch_size=1 so I can return a batch
train_dataset = train_dataset.map(encode, remove_columns=["text"], batched=True, batch_size=1, num_proc=cores)
print("🧱 \x1b[36m%d\x1b[0m training blocks" % train_dataset.num_rows)
print("🪙  \x1b[36m%d\x1b[0m training tokens" % sum(train_dataset["len"]))
with open(args.out_path+"/train_X.pickle", "wb") as file:
    pickle.dump(train_dataset["ids"], file)

val_dataset = datasets.load_dataset("JeanKaddour/minipile", split="validation", trust_remote_code=True)
print("📄 \x1b[36m%d\x1b[0m validation documents" % val_dataset.num_rows)
# batch_size=1 so I can return a batch
val_dataset = val_dataset.map(encode, remove_columns=["text"], batched=True, batch_size=1, num_proc=cores)
print("🧱 \x1b[36m%d\x1b[0m validation blocks" % val_dataset.num_rows)
print("🪙  \x1b[36m%d\x1b[0m validation tokens" % sum(val_dataset["len"]))
with open(args.out_path+"/val_X.pickle", "wb") as file:
    pickle.dump(val_dataset["ids"], file)

test_dataset = datasets.load_dataset("JeanKaddour/minipile", split="test", trust_remote_code=True)
print("📄 \x1b[36m%d\x1b[0m testing documents" % test_dataset.num_rows)
# batch_size=1 so I can return a batch
test_dataset = test_dataset.map(encode, remove_columns=["text"], batched=True, batch_size=1, num_proc=cores)
print("🧱 \x1b[36m%d\x1b[0m testing blocks" % test_dataset.num_rows)
print("🪙  \x1b[36m%d\x1b[0m testing tokens" % sum(test_dataset["len"]))
with open(args.out_path+"/test_X.pickle", "wb") as file:
    pickle.dump(test_dataset["ids"], file)

print("🍾 Done!")