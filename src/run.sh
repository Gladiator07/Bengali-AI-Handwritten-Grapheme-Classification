export CUDA_VISIBLE_DEVICES=0
export IMG_HEIGHT=137
export IMG_WIDTH=236
export EPOCHS=50
export TRAIN_BATCH_SIZE=256
export TEST_BATCH_SIZE=8
export MODEL_MEAN="(0.485, 0.456, 0.406)"
export MODEL_STD="(0.229, 0.224, 0.225)"
export BASE_MODEL="resnet34"

export TRAINING_FOLDS_CSV="/root/input/train_folds.csv"
export MODEL_DIR="../models"

declare -a val #array declaration
for i in 0 1 2 3 4
do
for TESTING_FOLDS in 0 1 2 3 4
do
if [[ $TESTING_FOLDS != $i ]]
then
val+=($TESTING_FOLDS)
else
export VALIDATION_FOLDS="$TESTING_FOLDS"
fi
done
export TRAINING_FOLDS="${val[*]}"
python3 train.py
val=()
done
