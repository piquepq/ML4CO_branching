# sample generation config
NODE_RECORD_PROB = 0.05  # probability of running the expert strategy and collecting samples.
TIME_LIMIT = 3600  # time limit for solving each instance
TRAIN_SIZE = 10  # number of samples for training
VALID_SIZE = 2  # number of samples for validation


# behavior cloning config
MAX_EPOCHS = 1000
BATCH_SIZE = 12  # large batch size will run out of GPU memory
PRETRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 128
LEARN_RATE = 2e-4
TOP_K = [1, 3, 5, 10]