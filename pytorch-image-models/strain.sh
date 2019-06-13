python train.py . --model 'tf_efficientnet_b3' --pretrained --lr 0.01 --warmup-lr 0.001 --decay-epochs 20 --num-classes 5 --batch-size 50 --epochs 50 --output './trained' #--initial-checkpoint './trained/train/20190609-141543-tf_efficientnet_b2-224/checkpoint-42.pth.tar'
# --initial-checkpoint './trained/train/20190602-071531-tf_efficientnet_b3-224/model_best.pth.tar'
# --num-classes 5 \
# --lr 0.01
# --pretrained
# --pretrained True \
# --mean 0.5 \
# --std 0.5 \
# --batch-size 16 \
# --epochs 60 \
# --output './trained' 
# --initial-checkpoint './trained/train/20190604-054435-gluon_resnet152_v1d-224/model_best.pth.tar'
# ./trained/train/20190531-133621-gluon_senet154-224/checkpoint-36.pth.tar
#--resume path
# --initial-checkpoint './trained/train/20190531-133621-gluon_senet154-224/checkpoint-36.pth.tar'
# --warmup-lr 0.00001