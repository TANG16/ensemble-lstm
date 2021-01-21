import utils as ut
from config import mydata, trainset, testset, goals, EVAL_TIME

ut.evaluate_baseline("gaze", testset)
ut.evaluate_baseline("rWristRotX euc", testset)
ut.evaluate_baseline("headRotX ori", testset)
