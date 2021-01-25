mykeys = ["headRotX", "rWristRotX", "torsoRotX"]
mydata = ["1_1", "1_2", "2_1", "4_1", "5_1", "6_1", "6_2", "7_1", "7_2", "7_3"]
myfeatures = ["rWristRotX euc", "headRotX ori"]
my_mean = [1.515465204886579, 1.0749099015977657, 1.0]  # this needs to be calculated
calculate_diff = True
input_size = int(calculate_diff) + len(myfeatures)
diff_mean = 0.05
testset = ["6_1", "6_2", "7_1", "7_2", "7_3"]
trainset = ["1_1", "1_2", "2_1", "4_1", "5_1"]
goals = [
    "cup_red",
    "plate_blue",
    "jug",
    "plate_green",
    "plate_red",
    "cup_green",
    "cup_blue",
    "cup_pink",
    "plate_pink",
    "bowl",
]
EVAL_TIME = 360
EVAL_STEP = 10
STOP_TRAIN = 0.9
colors = ["k", "g"]
CULL_COEFF = 0.2
