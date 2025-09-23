import torch

class Config:
    SEQ_LEN_SHORT = 60
    SEQ_LEN_MID = 20
    SEQ_LEN_LONG = 12
    INPUT_DIM_SHORT = -1
    INPUT_DIM_MID = 1
    INPUT_DIM_LONG = 1

    DROPOUT = 0.10

    # Short-Scale Encoder
    EMBED_DIM_SHORT = 128
    NUM_LAYERS_SHORT = 4
    NUM_HEADS_SHORT = 8
    FF_DIM_SHORT = 512

    # Mid-Scale Encoder
    EMBED_DIM_MID = 64
    NUM_LAYERS_MID = 2
    NUM_HEADS_MID = 4
    FF_DIM_MID = 256

    # Long-Scale Encoder
    EMBED_DIM_LONG = 32
    NUM_LAYERS_LONG = 1
    NUM_HEADS_LONG = 2
    FF_DIM_LONG = 128

    # Hierarchical Fuser
    FUSER_COMMON_DIM = 128
    FUSER_NUM_LAYERS = 2
    FUSER_NUM_HEADS = 4

    # Regression Head
    REG_HEAD_HIDDEN_DIM = FUSER_COMMON_DIM // 2

    FLAT_EMBED_DIM = 128
    FLAT_NUM_LAYERS = 6
    FLAT_NUM_HEADS = 8
    FLAT_FF_DIM = 512

    N_SPLITS = 5

    BATCH_SIZE = 4096
    LEARNING_RATE = 2e-4

    EPOCHS = 50
    WEIGHT_DECAY = 1e-5
    WARMUP_EPOCHS = 1

config = Config()