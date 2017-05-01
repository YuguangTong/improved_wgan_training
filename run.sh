#
RUNNUM=1
RUNTIME=500
ITER=50000
L1_WEIGHT=0.9
MODEL=wgan-gp

python gan_SR.py \
       --mode=$MODEL \
       --summary_dir=summary/$RUNNUM \
       --train_dir=train/$RUNNUM \
       --max_runtime=$RUNTIME \
       --max_iter=$ITER \
       --gen_l1_weight=$L1_WEIGHT \
       > run.log \
       2> err.log

# finish
sudo shutdown -h now
