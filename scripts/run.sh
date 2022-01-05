python tools/test_RR.py --dataset MNIST --ae_path "./trained_models/RRAE/PGD/MNIST_PGDinf_2432.pt;./trained_models/RRAE/PGDL2/MNIST_PGDL2_2432.pt;./trained_models/RRAE/BIM/MNIST_BIMinf_2432.pt;./trained_models/RRAE/BIML2/MNIST_BIML2_2432.pt;./trained_models/RRAE/DF/MNIST_DFinf_10000.pt;./trained_models/RRAE/DFL2/MNIST_DFL2_10000.pt;./trained_models/RRAE/CW/MNIST_CW_10000.pt"

python tools/test_RR.py --dataset cifar10 --ae_path "trained_models/RRAE/PGD/cifar10_PGDinf_2432.pt;trained_models/RRAE/PGDL2/cifar10_PGDL2_2432.pt;trained_models/RRAE/BIM/cifar10_BIMinf_2432.pt;trained_models/RRAE/BIML2/cifar10_BIML2_2432.pt;trained_models/RRAE/DF/cifar10_DFinf_10000.pt;trained_models/RRAE/DFL2/cifar10_DFL2_10000.pt;trained_models/RRAE/CW/cifar10_CW_10000.pt"

python tools/test_RR.py --dataset gtsrb --ae_path "./trained_models/RRAE/PGD/gtsrb_PGDinf_2432.pt;./trained_models/RRAE/PGDL2/gtsrb_PGDL2_2432.pt;./trained_models/RRAE/BIM/gtsrb_BIMinf_2432.pt;./trained_models/RRAE/BIML2/gtsrb_BIML2_2432.pt;./trained_models/RRAE/DF/gtsrb_DFinf_12630.pt;./trained_models/RRAE/DFL2/gtsrb_DFL2_12630.pt;./trained_models/RRAE/CW/gtsrb_CW_12630.pt"

python tools/test_RR.py --dataset MNIST --ae_path "trained_models/RRAE/EADA/MNIST_EADA_10000.pt;trained_models/RRAE/EADAL1/MNIST_EADAL1_10000.pt" \
&&
python tools/test_RR.py --dataset cifar10 --ae_path "trained_models/RRAE/EADA/cifar10_EADA_10000.pt;trained_models/RRAE/EADAL1/cifar10_EADAL1_10000.pt" \
&&
python tools/test_RR.py --dataset gtsrb --ae_path "trained_models/RRAE/EADA/gtsrb_EADA_12630.pt;trained_models/RRAE/EADAL1/gtsrb_EADAL1_12630.pt"

# CWinf
python tools/test_RR.py --dataset MNIST --ae_path "trained_models/RRAE/CWinf/MNIST_CW_10000.pt" \
&&
python tools/test_RR.py --dataset cifar10 --ae_path "trained_models/RRAE/CWinf/cifar10_CW_10000.pt" \
&&
python tools/test_RR.py --dataset gtsrb --ae_path "trained_models/RRAE/CWinf/gtsrb_CW_12630.pt"

# 0.01
python tools/test_RR.py --results_dir ./trained_models/ae_test_0.01 --drop_rate 0.01 --dataset MNIST --ae_path "./trained_models/RRAE/PGD/MNIST_PGDinf_2432.pt;./trained_models/RRAE/PGDL2/MNIST_PGDL2_2432.pt;./trained_models/RRAE/BIM/MNIST_BIMinf_2432.pt;./trained_models/RRAE/BIML2/MNIST_BIML2_2432.pt;./trained_models/RRAE/DF/MNIST_DFinf_10000.pt;./trained_models/RRAE/DFL2/MNIST_DFL2_10000.pt;./trained_models/RRAE/CW/MNIST_CW_10000.pt;trained_models/RRAE/CWinf/MNIST_CW_10000.pt;trained_models/RRAE/EADA/MNIST_EADA_10000.pt;trained_models/RRAE/EADAL1/MNIST_EADAL1_10000.pt" \
&& \
python tools/test_RR.py --results_dir ./trained_models/ae_test_0.01 --drop_rate 0.01 --dataset cifar10 --ae_path "trained_models/RRAE/PGD/cifar10_PGDinf_2432.pt;trained_models/RRAE/PGDL2/cifar10_PGDL2_2432.pt;trained_models/RRAE/BIM/cifar10_BIMinf_2432.pt;trained_models/RRAE/BIML2/cifar10_BIML2_2432.pt;trained_models/RRAE/DF/cifar10_DFinf_10000.pt;trained_models/RRAE/DFL2/cifar10_DFL2_10000.pt;trained_models/RRAE/CW/cifar10_CW_10000.pt;trained_models/RRAE/CWinf/cifar10_CW_10000.pt;trained_models/RRAE/EADA/cifar10_EADA_10000.pt;trained_models/RRAE/EADAL1/cifar10_EADAL1_10000.pt" \
&& \
python tools/test_RR.py --results_dir ./trained_models/ae_test_0.01 --drop_rate 0.01 --dataset gtsrb --ae_path "./trained_models/RRAE/PGD/gtsrb_PGDinf_2432.pt;./trained_models/RRAE/PGDL2/gtsrb_PGDL2_2432.pt;./trained_models/RRAE/BIM/gtsrb_BIMinf_2432.pt;./trained_models/RRAE/BIML2/gtsrb_BIML2_2432.pt;./trained_models/RRAE/DF/gtsrb_DFinf_12630.pt;./trained_models/RRAE/DFL2/gtsrb_DFL2_12630.pt;./trained_models/RRAE/CW/gtsrb_CW_12630.pt;trained_models/RRAE/CWinf/gtsrb_CW_12630.pt;trained_models/RRAE/EADA/gtsrb_EADA_12630.pt;trained_models/RRAE/EADAL1/gtsrb_EADAL1_12630.pt"

# 0.0
python tools/test_RR.py --results_dir ./trained_models/ae_test_0.0 --drop_rate 0.0 --dataset MNIST --ae_path "./trained_models/RRAE/PGD/MNIST_PGDinf_2432.pt;./trained_models/RRAE/PGDL2/MNIST_PGDL2_2432.pt;./trained_models/RRAE/BIM/MNIST_BIMinf_2432.pt;./trained_models/RRAE/BIML2/MNIST_BIML2_2432.pt;./trained_models/RRAE/DF/MNIST_DFinf_10000.pt;./trained_models/RRAE/DFL2/MNIST_DFL2_10000.pt;./trained_models/RRAE/CW/MNIST_CW_10000.pt;trained_models/RRAE/CWinf/MNIST_CW_10000.pt;trained_models/RRAE/EADA/MNIST_EADA_10000.pt;trained_models/RRAE/EADAL1/MNIST_EADAL1_10000.pt" \
&& \
python tools/test_RR.py --results_dir ./trained_models/ae_test_0.0 --drop_rate 0.0 --dataset cifar10 --ae_path "trained_models/RRAE/PGD/cifar10_PGDinf_2432.pt;trained_models/RRAE/PGDL2/cifar10_PGDL2_2432.pt;trained_models/RRAE/BIM/cifar10_BIMinf_2432.pt;trained_models/RRAE/BIML2/cifar10_BIML2_2432.pt;trained_models/RRAE/DF/cifar10_DFinf_10000.pt;trained_models/RRAE/DFL2/cifar10_DFL2_10000.pt;trained_models/RRAE/CW/cifar10_CW_10000.pt;trained_models/RRAE/CWinf/cifar10_CW_10000.pt;trained_models/RRAE/EADA/cifar10_EADA_10000.pt;trained_models/RRAE/EADAL1/cifar10_EADAL1_10000.pt" \
&& \
python tools/test_RR.py --results_dir ./trained_models/ae_test_0.0 --drop_rate 0.0 --dataset gtsrb --ae_path "./trained_models/RRAE/PGD/gtsrb_PGDinf_2432.pt;./trained_models/RRAE/PGDL2/gtsrb_PGDL2_2432.pt;./trained_models/RRAE/BIM/gtsrb_BIMinf_2432.pt;./trained_models/RRAE/BIML2/gtsrb_BIML2_2432.pt;./trained_models/RRAE/DF/gtsrb_DFinf_12630.pt;./trained_models/RRAE/DFL2/gtsrb_DFL2_12630.pt;./trained_models/RRAE/CW/gtsrb_CW_12630.pt;trained_models/RRAE/CWinf/gtsrb_CW_12630.pt;trained_models/RRAE/EADA/gtsrb_EADA_12630.pt;trained_models/RRAE/EADAL1/gtsrb_EADAL1_12630.pt"