### Code & Usage
`src` folder contains codes for training a deep neural network to do image classification on FashionMNIST and CIFAR10/100. You can train models with the `main.py` script, with hyper-parameters being specified as flags

After obtaining the results, to see the comparison, use `draw_comps.py` by specifying the logs folder, and `fig-type` (either "stagewise" or "others"), for example:
```
python draw_comps2.py --logs-folder ./logs/CIFAR10 --fig-type others
```



### Reproducing Results



#### FashionMNIST
```
python ./src/main.py --optim-method SGD --eta0 0.007 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 100 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/FashionMNIST --dataset FashionMNIST --dataroot ./data

python ./src/main.py --optim-method SGD_Stage_Decay --eta0 0.04 --alpha 0.1 --milestones 12000 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 100 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/FashionMNIST --dataset FashionMNIST --dataroot ./data

python ./src/main.py --optim-method SGD_Stage_Decay --eta0 0.04 --alpha 0.1 --milestones 9000 15000 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 100 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/FashionMNIST --dataset FashionMNIST --dataroot ./data

python ./src/main.py --optim-method SGD_ReduceLROnPlateau --eta0 0.04 --alpha 0.5 --patience 3 --threshold 0.001 --nesterov --momentum 0.9 --weight-decay 0.001 --train-epochs 100 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/FashionMNIST --dataset FashionMNIST --dataroot ./data

python ./src/main.py --optim-method SGD_1sqrt_Decay --eta0 0.05 --alpha 0.00653 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 50 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/FashionMNIST --dataset FashionMNIST --dataroot ./data

python ./src/main.py --optim-method SGD_1t_Decay --eta0 0.05 --alpha 0.000384 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 100 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/FashionMNIST --dataset FashionMNIST --dataroot ./data

python ./src/main.py --optim-method SGD_Exp_Decay --eta0 0.05 --alpha 0.999902 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 100 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/FashionMNIST --dataset FashionMNIST --dataroot ./data

python ./src/main.py --optim-method Adam --eta0 0.0009 --weight-decay 0.0001 --train-epochs 100 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/FashionMNIST --dataset FashionMNIST --dataroot ./data

python ./src/main.py --optim-method SGD_Cosine_Decay --eta0 0.05 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 50 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/FashionMNIST --dataset FashionMNIST --dataroot ./data

python ./src/main.py --optim-method SLS-Armijo1 --eta0 0.5 --c 0.1 --train-epochs 100 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/FashionMNIST --dataset FashionMNIST --dataroot ./data

python ./src/main.py --optim-method SGD_ln1_Decay --eta0 0.05 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 50 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/FashionMNISTStep  --log_step_length_folder ./logs/FashionMNISTStep --dataset FashionMNIST --dataroot ./data

```


#### CIFAR10 using Convolutional Neural Network (CNN) model
```
python ./src/main.py --optim-method SGD --eta0 0.07 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR10 --dataset CIFAR10 --dataroot ./data

python ./src/main.py --optim-method SGD_Stage_Decay --eta0 0.1 --alpha 0.1 --milestones 32000 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR10 --dataset CIFAR10 --dataroot ./data

python ./src/main.py --optim-method SGD_Stage_Decay --eta0 0.2 --alpha 0.1 --milestones 32000 40000 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR10 --dataset CIFAR10 --dataroot ./data

python ./src/main.py --optim-method SGD_ReduceLROnPlateau --eta0 0.07 --alpha 0.1 --patience 10 --threshold 0.001 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR10 --dataset CIFAR10 --dataroot ./data

python ./src/main.py --optim-method SGD_1sqrt_Decay --eta0 0.2 --alpha 0.079079 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR10 --dataset CIFAR10 --dataroot ./data

python ./src/main.py --optim-method SGD_1t_Decay --eta0 0.1 --alpha 0.000230 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR10 --dataset CIFAR10 --dataroot ./data

python ./src/main.py --optim-method SGD_Exp_Decay --eta0 0.1 --alpha 0.99991 --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR10 --dataset CIFAR10 --dataroot ./data

python ./src/main.py --optim-method Adam --eta0 0.0009 --weight-decay 0.0001 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR10 --dataset CIFAR10 --dataroot ./data

python ./src/main.py --optim-method SGD_Cosine_Decay --eta0 0.25 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR10 --dataset CIFAR10 --dataroot ./data

python ./src/main.py --optim-method SLS-Armijo0 --eta0 2.5 --c 0.1 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR10 --dataset CIFAR10 --dataroot ./data

python ./src/main.py --optim-method SGD_ln1_Decay --eta0 0.25 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR10 --dataset CIFAR10 --dataroot ./data
```


#### CIFAR100 using DenseNet-BC model
```
python ./src/main.py --optim-method SGD --eta0 0.07 --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 50 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data

python ./src/main.py --optim-method SGD_Stage_Decay --eta0 0.07 --alpha 0.1 --milestones 15000 --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 50 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data

python ./src/main.py --optim-method SGD_Stage_Decay --eta0 0.07 --alpha 0.1 --milestones 12000 18000 --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 50 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data

python ./src/main.py --optim-method SGD_ReduceLROnPlateau --eta0 0.1 --alpha 0.5 --patience 3 --threshold 0.001 --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 50 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data

python ./src/main.py --optim-method SGD_1sqrt_Decay --eta0 0.1 --alpha 0.015 --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 50 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data

python ./src/main.py --optim-method SGD_1t_Decay --eta0 0.8 --alpha 0.004 --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 50 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data

python ./src/main.py --optim-method SGD_Exp_Decay --eta0 0.2 --alpha 0.999744 --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 50 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data

python ./src/main.py --optim-method Adam --eta0 0.0009 --weight-decay 0.0005 --train-epochs 50 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data

python ./src/main.py --optim-method SGD_Cosine_Decay --eta0 0.09 --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 50 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data

python ./src/main.py --optim-method SLS-Armijo2 --eta0 5 --c 0.5 --train-epochs 50 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data

python ./src/main.py --optim-method SGD_ln1_Decay --eta0 0.53 --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 50 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data
```



