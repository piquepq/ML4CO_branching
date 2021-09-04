# test config
# sample generation config
N_JOBS = 4  # number of parallel jobs
NODE_RECORD_PROB = 0.05  # probability of running the expert strategy and collecting samples.
TIME_LIMIT = 3600  # time limit for solving each instance
TRAIN_SIZE = 100  # number of samples for training. I recommend set it as 100000
VALID_SIZE = 20  # number of samples for validation. I recommend set it as 20000


# behavior cloning config
MAX_EPOCHS = 10  # I recommend set it as 1000
EPOCH_SIZE = 4
BATCH_SIZE = 2  # large batch size will run out of GPU memory
PRETRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 2
LEARN_RATE = 1e-4
TOP_K = [1, 3, 5, 10]

#
# # suggest config
# # sample generation config
# N_JOBS = 24  # number of parallel jobs
# NODE_RECORD_PROB = 0.05  # probability of running the expert strategy and collecting samples.
# TIME_LIMIT = 3600  # time limit for solving each instance
# TRAIN_SIZE = 1000  # number of samples for training. I recommend set it as 100000
# VALID_SIZE = 200  # number of samples for validation. I recommend set it as 20000
#
#
# # behavior cloning config
# MAX_EPOCHS = 10  # I recommend set it as 1000
# EPOCH_SIZE = 10000
# BATCH_SIZE = 24  # large batch size will run out of GPU memory
# PRETRAIN_BATCH_SIZE = 128
# VALID_BATCH_SIZE = 128
# LEARN_RATE = 1e-4
# TOP_K = [1, 3, 5, 10]