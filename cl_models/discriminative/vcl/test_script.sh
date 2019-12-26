VTYPE='drs'
TASK_TYPE='split'
RPATH='./results/mnist/'
NTASKS=5
DATASET='mnist'
CORESET_SIZE=50
FBDGT='False'
TRAIN_SIZE=1000
BATCH_SIZE=1
B=20
LR=0.0002
GRAD='adam'
DISC='True'
ER='False'
BITER=1
SEED=0

RUNS=1

while [ $SEED -lt $RUNS ]
do

	echo $SEED

    python VCL_test.py --result_path $RPATH --task_type $TASK_TYPE --num_tasks $NTASKS --dataset $DATASET --epoch 1 --coreset_size $CORESET_SIZE --fixed_budget $FBDGT --coreset_mode ring_buffer --batch_size $BATCH_SIZE --train_size $TRAIN_SIZE --test_size -1 --vcl_type $VTYPE --B $B --learning_rate $LR --grad_type $GRAD --batch_iter $BITER --discriminant $DISC --ER $ER --lambda_disc 0.0001 --seed $SEED
    ((SEED++))
done