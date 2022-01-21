his model is simple.
original model consists of Conv, BN, pooling and dense layer.
but my model consists of Conv and pooling, dense layer.
 [ I think how to fold(fusing, merge) conv layer and BN layer? ] 
I think I combine conv weights and bn weights.
my model doesn't exist BN layer but my model is faster than original model.

Let me explain the file composition!
1. main.py is training for model.
2. dataset.py is to load cifar10 datasets.
3. model.py is created by me. 
4. make_weight_model.py is to bring weights and bias to original model.
5. utils.py is funtion to make for new weights.
6. with_BNLAYER_model.py include BN layers (original)

If you finish training to my model, you can see result on tensorboard.
please type code [ %tensorboard --logdir logs ]




breifly, let me show you diffence on two model. 

When I train Batch Normalization Folding model, I use colab gpu and take 546.523 second.
Bacth Normalization Folding model have 2,395,434 params.

When I train original model, I use colab gpu and take 686.361 second. 
Original model have 2,397,226 params.










