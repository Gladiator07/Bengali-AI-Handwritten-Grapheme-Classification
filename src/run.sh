export CUDA_VISIBLE_DEVICES=0
export IMG_HEIGHT=137
export IMG_WIDTH=236
export EPOCHS=50
export TRAIN_BATCH_SIZE=64
export TEST_BATCH_SIZE=8
export MODEL_MEAN="(0.485, 0.456, 0.406"
export MODEL_STD="(0.229, 0.224, 0.225)"
export BASE_MODEl="resnet34"
export TRAINING_FOLDS="(0, 1, 2, 3)"
export TRAINING_FOLDS="(4,)"
export TRAINING_FOLDS_CSV="/root/input/train_folds.csv"
export TRAINING_FOLDS="(0, 1, 2, 4)"
export TRAINING_FOLDS="(3,)"

export TRAINING_FOLDS="(0, 1, 4, 3)"
export TRAINING_FOLDS="(2,)"

export TRAINING_FOLDS="(0, 4, 2, 3)"
export TRAINING_FOLDS="(1,)"

export TRAINING_FOLDS="(0, 4, 2, 3)"
export TRAINING_FOLDS="(0,)"

python3 train.py
