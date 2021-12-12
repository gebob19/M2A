datadir=/ # point to somethingsomething dataset 

python main.py something RGB \
     --arch resnet18 --num_segments 8 --wd 5e-4 \
     --gd 20 --lr 0.02 --lr_steps 20 --epochs 30 \
     --batch-size 32 -j 32 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --temporal_module=tdn+tam --shift_div=8 --shift_place=blockres --npb \
     --exp_name=tdn+tam --n_batch_multiplier 4 \
     --datapath=$datadir 
