This forked repo implements an evalution for Rectified Rejection as an AE
detection methods. The evaluation metric is aligned with [flymin/AEdetection](https://github.com/flymin/AEdetection).
For more details, please check that repo.

The training procedures show respect to the original repo.

We trained the models by ourselves. For the weights, please check
[Google Drive](https://drive.google.com/file/d/1TVvtsRwgj6Tq9HkpKhMqaWSw6uh1s92q/view?usp=sharing).
We use `tools/attack_RR.py` to generate AE samples and `tools/test_RR.py` for
evaluation.

After Download and extract the `.tar` file, the `trained_models` directory
should look as follows (`trained_models/RRAE` comes from executing `tools/attack_RR.py`)
```
trained_models/
├── CIFAR-10
│   ├── PGDAT_densenet169BN_adaptiveT...
│   │   ├── model_best.pth
│   │   ├── model_best_s.pth
│   │   └── output_simple.log
│   └── PGDAT_PreActResNet18_...
│       ├── model_best.pth
│       ├── output.log
│       └── output_simple.log
├── gtsrb
│   └── PGDAT_ResNet18BN_adaptiveT...
│       ├── model_best.pth
│       └── output_simple.log
├── MNIST
│   └── PGDAT_Mnist2LayerNetBN_adaptiveT...
│       ├── model_best.pth
│       └── output_simple.log
└── RRAE
    ├── BIM
    │   ├── cifar10
    │   │   └── RR-cifar10-2021-06-17-11-01-20.log
    │   ├── cifar10_BIMinf_2432.pt
    ....
    └── PGDL2
        └── ...
```

For testing command examples, please check:
[scripts/run.sh](https://github.com/flymin/Rectified-Rejection/blob/main/scripts/run.sh).


Below is the original readme.

---

# Adversarial Training with Rectified Rejection

The code for the paper [Adversarial Training with Rectified Rejection](https://arxiv.org/abs/2105.14785).

## Environment settings and libraries we used in our experiments

This project is tested under the following environment settings:
- OS: Ubuntu 18.04.4
- GPU: Geforce 2080 Ti or Tesla P100
- Cuda: 10.1, Cudnn: v7.6
- Python: 3.6
- PyTorch: >= 1.6.0
- Torchvision: >= 0.6.0

## Acknowledgement
The codes are modifed based on [Rice et al. 2020](https://github.com/locuslab/robust_overfitting), and the model architectures are implemented by [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar).

## Training Commands
Below we provide running commands training the models with the RR module, taking the setting of PGD-AT + RR (ResNet-18) as an example:
```python
python train_cifar.py --model_name PreActResNet18_twobranch_DenseV1 --attack pgd --lr-schedule piecewise \
                                              --epochs 110 --epsilon 8 \
                                              --attack-iters 10 --pgd-alpha 2 \
                                              --fname auto \
                                              --batch-size 128 \
                                              --adaptivetrain --adaptivetrainlambda 1.0 \
                                              --weight_decay 5e-4 \
                                              --twobranch --useBN \
                                              --selfreweightCalibrate \
                                              --dataset 'CIFAR-10' \
                                              --ATframework 'PGDAT' \
                                              --SGconfidenceW
```
The FLAG `--model_name` can be `PreActResNet18_twobranch_DenseV1` (ResNet-18) or `WideResNet_twobranch_DenseV1` (WRN-34-10). For alternating different AT frameworks, we can set the FLAG `--ATframework` to be one of `PGDAT`, `TRADES`, `CCAT`.


## Evaluation Commands
Below we provide running commands for evaluations.

### Evaluating under the PGD attacks
The trained model is saved at `trained_models/model_path`, where the specific name of `model_path` is automatically generated during training. The command for evaluating under PGD attacks is:
```python
python eval_cifar.py --model_name PreActResNet18_twobranch_DenseV1 --evalset test --norm l_inf --epsilon 8 \
                                              --attack-iters 1000 --pgd-alpha 2 \
                                              --fname trained_models/model_path \
                                              --load_epoch -1 \
                                              --dataset 'CIFAR-10' \
                                              --twobranch --useBN \
                                              --selfreweightCalibrate

```


### Evaluating under the adaptive CW attacks
The parameter FLAGs `--binary_search_steps`, `--CW_iter`, `--CW_confidence` can be changed, where `--detectmetric` indicates the rejector that needs to be adaptively evaded.
```python
python eval_cifar_CW.py --model_name PreActResNet18_twobranch_DenseV1 --evalset adaptiveCWtest \
                                              --fname trained_models/model_path \
                                              --load_epoch -1 --seed 2020 \
                                              --binary_search_steps 9 --CW_iter 100 --CW_confidence 0 \
                                              --threatmodel linf --reportmodel linf \
                                              --twobranch --useBN \
                                              --selfreweightCalibrate \
                                              --detectmetric 'RR' \
                                              --dataset 'CIFAR-10'
```

### Evaluating under multi-target and GAMA attacks
The running command for evaluating under multi-target attacks is activated by the FLAG `--evalonMultitarget` as:
```python
python eval_cifar.py --model_name PreActResNet18_twobranch_DenseV1 --evalset test --norm l_inf --epsilon 8 \
                                              --attack-iters 100 --pgd-alpha 2 \
                                              --fname trained_models/model_path \
                                              --load_epoch -1 \
                                              --dataset 'CIFAR-10' \
                                              --twobranch --useBN \
                                              --selfreweightCalibrate \
                                              --evalonMultitarget --restarts 1

```

The running command for evaluating under GAMA attacks is activated by the FLAG `--evalonGAMA_PGD` or `--evalonGAMA_FW` as:
```python
python eval_cifar.py --model_name PreActResNet18_twobranch_DenseV1 --evalset test --norm l_inf --epsilon 8 \
                                              --attack-iters 100 --pgd-alpha 2 \
                                              --fname trained_models/model_path \
                                              --load_epoch -1 \
                                              --dataset 'CIFAR-10' \
                                              --twobranch --useBN \
                                              --selfreweightCalibrate \
                                              --evalonGAMA_FW

```

### Evaluating under CIFAR-10-C
The running command for evaluating on common corruptions in CIFAR-10-C is:
```python
python eval_cifar_CIFAR10-C.py --model_name PreActResNet18_twobranch_DenseV1 \
                                              --fname trained_models/model_path \
                                              --load_epoch -1 \
                                              --dataset 'CIFAR-10' \
                                              --twobranch --useBN \
                                              --selfreweightCalibrate

```
