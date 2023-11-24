import numpy
import os

# CS stuff
TYPE = "float32"
B = {"double64": 8, "float32": 4, "float16": 2, "int8": 1}[TYPE]
BATCH_SIZE = 256 # Higher BATCH_SIZE leads to higher UTILIZATION
GPU = "A100"
PEAK_double64_Ops = {"A100": 10*10**12}[GPU]
PEAK_Ops = PEAK_double64_Ops * (8/B)
UTILIZATION = 1/3
Ops = PEAK_Ops * UTILIZATION
DATASET_SIZE = 1.2 * 10**6
t__t_ratio = 3

# ML stuff
θ =  numpy.array([1,2,4,6,8,10,12,14,16,18,20,40,60,80,100,120])

# no = (884682*θ**2)+(1536846*θ)+4005000
no = (1179594*θ**2)+(769614*θ)+100 # V2

μ = B*no

# MAD = (4644864*θ**2)+(2863104*θ)+4000000
MAD = (4939776*θ**2)+(2095104*θ) # V2

t_1_batch = MAD*BATCH_SIZE/Ops

t_1_epoch = MAD*DATASET_SIZE/Ops

t__1_epoch = t__t_ratio * t_1_epoch

t__100_epochs = 100 * t__1_epoch

def user_friendly_number(number):
    number_K = number/10**3
    number_M = number/10**6
    number_B = number/10**9

    if number_B >= 1:
        return "%.2f B" % (number_B)
    elif number_M >= 1:
        return "%.2f M" % (number_M)
    elif number_K >= 1:
        return "%.2f K" % (number_K)

def user_friendly_μ(μ):
    μ_KiB = μ/2**10
    μ_MiB = μ/2**20
    μ_GiB = μ/2**30

    if μ_GiB >= 1:
        return "%.2f GiB" % (μ_GiB)
    elif μ_MiB >= 1:
        return "%.2f MiB" % (μ_MiB)
    elif μ_KiB >= 1:
        return "%.2f KiB" % (μ_KiB)

def user_friendly_O(O):
    O_K = O/10**3
    O_M = O/10**6
    O_G = O/10**9
    O_T = O/10**12
    O_P = O/10**15

    if O_P >= 1:
        return "%.2f PO" % (O_P)
    elif O_T >= 1:
        return "%.2f TO" % (O_T)
    elif O_G >= 1:
        return "%.2f GO" % (O_G)
    elif O_M >= 1:
        return "%.2f MO" % (O_M)
    elif O_K >= 1:
        return "%.2f KO" % (O_K)

def user_friendly_s(s):
    μs = s*10**6
    ms = s*10**3
    m = s/60
    h = m/60
    d = h/24

    if d >= 1:
        return "%.2f d" % (d)
    elif h >= 1:
        return "%.2f h" % (h)
    elif m >= 1:
        return "%.2f m" % (m)
    elif s >= 1:
        return "%.2f s" % (s)
    elif ms >= 1:
        return "%.2f ms" % (ms)
    elif μs >= 1:
        return "%.2f us" % (μs)

script_path = os.path.abspath(__file__)
src_path = os.path.dirname(script_path)
root_path = os.path.dirname(src_path)

with open(os.path.join(root_path, "res/card_template.tex"), "r") as card_template:
    card_template_lines = card_template.readlines()

with open(os.path.join(root_path, "out/model_card.tex"), "w") as model_card:
    model_card.writelines(card_template_lines[:10])

    model_card.write("\caption{type=%s, batch size=%d, GPU=%s, utilization=%.2f, dataset size=%s, $\\frac{t^{'}}{t}=%d$}\n" % (TYPE,BATCH_SIZE,GPU,UTILIZATION,user_friendly_number(DATASET_SIZE),t__t_ratio))

    model_card.writelines(card_template_lines[10:14])

    for i in range(len(θ)):
        model_card.write("%d & %s & %s & %s & %s & %s \\\\\n" % (θ[i], user_friendly_number(no[i]), user_friendly_μ(μ[i]), user_friendly_O(MAD[i]), user_friendly_s(t_1_batch[i]), user_friendly_s(t__100_epochs[i])))

    model_card.writelines(card_template_lines[14:])