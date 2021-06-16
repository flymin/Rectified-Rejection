import argparse
import logging
import sys
import time
import math

from torch.functional import norm
from misc.utils import *
from misc.load_dataset import LoadDataset
from models import *
import torch.nn.functional as F
from torchvision import datasets, transforms
import os
import numpy as np
import torchvision

criterion_kl = nn.KLDivLoss(reduction='batchmean')
lower_limit = 0.
upper_limit = 1.


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, norm,
               adaptive_evidence=False, adaptive_lambda=1.,
               uniform_lambda=False, BNeval=False, twobranch=False,
               twosign=False, normalize=None):
    if BNeval:
        model.eval()
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        # uniform sampling for adaptive lambda
        if uniform_lambda:
            if twosign:
                a_lambda = torch.zeros(
                    y.shape[0]).uniform_(- adaptive_lambda, adaptive_lambda).cuda()
            else:
                a_lambda = torch.zeros(
                    y.shape[0]).uniform_(
                    0., adaptive_lambda).cuda()
        else:
            a_lambda = adaptive_lambda

        for _ in range(attack_iters):
            if twobranch:
                output, output_evi = model(normalize(X + delta))
                evi = output_evi.logsumexp(dim=1)
            else:
                output = model(normalize(X + delta))
                evi = output.logsumexp(dim=1)
            loss = F.cross_entropy(output, y)
            # if apply adaptive attacks for the evidence detection
            if adaptive_evidence:
                loss += (a_lambda * evi).mean()
            grad = torch.autograd.grad(loss, delta)[0]
            if norm == "l_inf":
                d = torch.clamp(
                    delta + alpha * torch.sign(grad),
                    min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(
                    grad.view(grad.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                scaled_g = grad / (g_norm + 1e-10)
                d = (delta + scaled_g * alpha).view(delta.size(0), -1).renorm(
                    p=2, dim=0, maxnorm=epsilon).view_as(delta)
            d = clamp(d, lower_limit - X, upper_limit - X)
            delta.data = d
        if twobranch:
            all_loss = F.cross_entropy(
                model(normalize(X + delta))[0],
                y, reduction='none')
        else:
            all_loss = F.cross_entropy(
                model(normalize(X + delta)),
                y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    if BNeval:
        model.train()
    return max_delta, a_lambda


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='PreActResNet18')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--lr-schedule', default='piecewise', type=str)
    parser.add_argument('--attack', default='pgd', type=str,
                        choices=['pgd', 'free', 'none'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float)
    parser.add_argument('--fgsm-alpha', default=1.25, type=float)
    parser.add_argument('--norm', default='l_inf',
                        type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fname', default='cifar_model', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--weight_decay', default=5e-4,
                        type=float)  # weight decay
    parser.add_argument('--optimizer', default='SGD', type=str)
    # whether use target-mode attack
    parser.add_argument('--target', action='store_true')

    parser.add_argument(
        '--ATframework', default='PGDAT', type=str,
        choices=['PGDAT', 'TRADES', 'CCAT'])
    parser.add_argument('--TRADESlambda', default=1., type=float)
    parser.add_argument('--CCATiter', default=20, type=int)
    parser.add_argument('--CCATrho', default=1, type=int)
    parser.add_argument('--CCATstep', default=1., type=float)
    parser.add_argument('--CCATratio', default=1., type=float)
    parser.add_argument('--CCATscale', default=1., type=float)

    # adaptive attack
    # whether use adaptive term in the attacks
    parser.add_argument('--adaptiveattack', action='store_true')
    parser.add_argument('--adaptiveattacklambda', default=1., type=float)
    # whether use uniform distribution for lambda in adaptive attack
    parser.add_argument('--uniform_lambda', action='store_true')
    # whether use eval mode for BN when crafting adversarial examples
    parser.add_argument('--BNeval', action='store_true')
    parser.add_argument('--twosign', action='store_true')

    # adaptive training
    # whether use adaptive term in train
    parser.add_argument('--adaptivetrain', action='store_true')
    parser.add_argument('--adaptivetrainlambda', default=1., type=float)

    parser.add_argument('--selfreweightCalibrate',
                        action='store_true')  # Calibrate
    parser.add_argument('--temp', default=1., type=float)
    parser.add_argument('--tempC', default=1., type=float)
    # stop gradient for the confidence term
    parser.add_argument('--tempC_trueonly', default=1., type=float)
    # stop gradient for the confidence term
    parser.add_argument('--SGconfidenceW', action='store_true')

    parser.add_argument('--ConfidenceOnly', action='store_true')
    parser.add_argument('--AuxiliaryOnly', action='store_true')

    # two branch for our selfreweightCalibrate (rectified rejection)
    parser.add_argument('--twobranch', action='store_true')
    parser.add_argument('--out_dim', default=10, type=int)
    parser.add_argument('--useBN', action='store_true')
    parser.add_argument('--along', action='store_true')

    # EBD baseline
    # Energy-based Out-of-distribution Detection
    parser.add_argument('--selfreweightNIPS20', action='store_true')
    parser.add_argument('--m_in', default=6, type=float)
    parser.add_argument('--m_out', default=3, type=float)

    # ATRO baseline
    # ATRO https://github.com/MasaKat0/ATRO
    parser.add_argument('--selfreweightATRO', action='store_true')
    parser.add_argument('--ATRO_cost', default=0.3, type=float)
    parser.add_argument('--ATRO_coefficient', default=0.3, type=float)

    # CARL baseline
    # CARL https://github.com/cassidylaidlaw/playing-it-safe
    parser.add_argument('--selfreweightCARL', action='store_true')
    parser.add_argument('--CARL_lambda', default=0.5, type=float)
    parser.add_argument('--CARL_eta', default=0.02, type=float)

    return parser.parse_args()


def main():
    args = get_args()
    args.attack = "pgd"
    args.lr_schedule = "piecewise"
    args.epochs = 110
    args.epsilon = 8
    args.fname = "auto"
    args.adaptivetrain = True
    args.twobranch = True
    args.useBN = True
    args.selfreweightCalibrate = True
    args.ATframework = 'PGDAT'
    args.SGconfidenceW = True
    if args.dataset == 'cifar10':
        args.model_name = 'densenet169'
    elif args.dataset == 'MNIST':
        args.model_name = 'Mnist2LayerNet'
    elif args.dataset == 'gtsrb':
        args.model_name = 'ResNet18'
    else:
        raise NotImplementedError()

    # transfer parameter
    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.benchmark = True

    if args.fname == 'auto':
        names = get_auto_fname(args)
        args.fname = 'trained_models/' + args.dataset + '/' + names
    else:
        args.fname = 'trained_models/' + args.dataset + '/' + args.fname

    if not os.path.exists(args.fname):
        os.makedirs(args.fname)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'output_simple.log')),
            logging.StreamHandler()
        ])

    logger.info(args)

    # Prepare dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])

    if args.dataset == "cifar10":
        cls_norm = [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
        trainset = torchvision.datasets.CIFAR10(
            root='./dataset', train=True, download=True,
            transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root='./dataset', train=False, download=True,
            transform=transform_test)
        num_cla = 10
    else:
        if args.dataset == "MNIST":
            cls_norm = [(0.13), (0.31)]
            num_cla = 10
            img_size = (28, 28)
        elif args.dataset == "gtsrb":
            cls_norm = [(0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629)]
            num_cla = 43
            img_size = (32, 32)
        else:
            raise NotImplementedError()
        trainset = LoadDataset(
            args.dataset, './dataset', train=True, download=False,
            resize_size=img_size, hdf5_path=None, random_flip=True, norm=False)
        testset = LoadDataset(
            args.dataset, './dataset', train=False, download=False,
            resize_size=img_size, hdf5_path=None, random_flip=False, norm=False)

    normalize = torchvision.transforms.Normalize(*cls_norm)
    train_batches = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2,
        pin_memory=True)
    test_batches = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2,
        pin_memory=True)

    if args.selfreweightCalibrate or args.selfreweightATRO or args.selfreweightCARL:
        along = True
        args.out_dim = 1

    # Creat model
    model = eval(
        args.model_name +
        "(num_classes=num_cla, out_dim=args.out_dim, use_BN=args.useBN," +
        "along=along)")

    model = nn.DataParallel(model).cuda()
    model.train()
    params = model.parameters()

    if args.optimizer == 'SGD':
        opt = torch.optim.SGD(params, lr=args.lr_max,
                              momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        opt = torch.optim.Adam(
            params, lr=args.lr_max, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)

    if args.attack == 'free':
        epochs = int(math.ceil(args.epochs / args.attack_iters))
    else:
        epochs = args.epochs

    def lr_schedule(t):
        if t < 100:
            return args.lr_max
        elif t < 105:
            return args.lr_max / 10.
        else:
            return args.lr_max / 100.

    best_test_robust_acc, start_epoch = 0, 0

    criterion = nn.CrossEntropyLoss()
    BCEcriterion = nn.BCELoss(reduction='none')
    # MHRLoss = MaxHingeLossWithRejection(args.ATRO_cost)

    # logger.info('Epoch \t Acc \t Robust Acc \t Evi \t Robust Evi')
    logger.info('Epoch \t Acc \t Robust Acc')
    for epoch in range(start_epoch, epochs):
        model.train()
        for i, (data, target) in enumerate(train_batches):
            X, y = data.cuda(), target.cuda()
            epoch_now = epoch + (i + 1) / len(train_batches)
            lr = lr_schedule(epoch_now)
            opt.param_groups[0].update(lr=lr)

            if args.ATframework == 'PGDAT':  # use pgd attack
                delta, _ = attack_pgd(
                    model, X, y, epsilon, pgd_alpha, args.attack_iters, args.
                    restarts, args.norm, adaptive_evidence=args.adaptiveattack,
                    adaptive_lambda=args.adaptiveattacklambda,
                    uniform_lambda=args.uniform_lambda, BNeval=args.BNeval,
                    twobranch=args.twobranch, twosign=args.twosign,
                    normalize=normalize)
                delta = delta.detach()
            else:
                raise NotImplementedError()

            # whether use two branches
            if args.twobranch:
                robust_output, robust_output_aux = model(
                    normalize(
                        torch.clamp(
                            X + delta,
                            min=lower_limit,
                            max=upper_limit)))
            else:
                raise NotImplementedError()

            # choose between PGDAT, CCAT and TRADES
            if args.ATframework == 'PGDAT':  # calculate loss
                cls_loss = criterion(robust_output, y)
            else:
                raise NotImplementedError()

            if args.adaptivetrain:
                if args.selfreweightCalibrate:  # next here
                    # all temp = 1, robust_output_s and robust_output_s_ should
                    # be the same
                    robust_output_s = torch.softmax(
                        robust_output * args.tempC, dim=1)
                    # predicted label and confidence
                    robust_con_pre, _ = robust_output_s.max(1)

                    robust_output_s_ = torch.softmax(
                        robust_output * args.tempC_trueonly, dim=1)
                    # predicted prob on the true label y
                    robust_con_y = robust_output_s_[
                        torch.tensor(range(X.size(0))), y].detach()

                    if args.SGconfidenceW:  # also here
                        correct_index = torch.where(
                            robust_output.max(1)[1] == y)[0]
                        robust_con_pre[correct_index] = \
                            robust_con_pre[correct_index].detach()

                    # bs, Calibration function A \in [0,1]
                    robust_output_aux = robust_output_aux.sigmoid().squeeze()
                    robust_detector = robust_con_pre * robust_output_aux

                    aux_loss = BCEcriterion(robust_detector, robust_con_y)
                    robust_loss = cls_loss + \
                        args.adaptivetrainlambda * aux_loss.mean(dim=0)

                else:
                    raise NotImplementedError()
            # this iteration
            opt.zero_grad()
            robust_loss.backward()
            opt.step()
            if i % 50 == 0:
                logger.info('\t %d \t %.4f \t %.4f \t %.4f', i, cls_loss.item(),
                            aux_loss.mean().item(), robust_loss.mean().item())
        # end for one epoch

        # start test
        model.eval()
        test_acc = 0
        test_robust_acc = 0
        test_evi_correct = 0
        test_robust_evi_correct = 0
        test_evi_wrong = 0
        test_robust_evi_wrong = 0
        test_n = 0
        for i, (data, target) in enumerate(test_batches):
            X, y = data.cuda(), target.cuda()

            # Random initialization
            delta, _ = attack_pgd(
                model, X, y, epsilon, pgd_alpha, args.attack_iters,
                args.restarts, args.norm, twobranch=args.twobranch,
                normalize=normalize)
            delta = delta.detach()

            if args.twobranch:
                output, output_aux = model(normalize(X))
                robust_output, robust_output_aux = model(
                    normalize(
                        torch.clamp(
                            X + delta,
                            min=lower_limit,
                            max=upper_limit)))

                # predicted label and confidence
                con_pre, _ = torch.softmax(output * args.tempC, dim=1).max(1)
                # predicted label and confidence
                robust_con_pre, _ = torch.softmax(
                    robust_output * args.tempC, dim=1).max(1)

                if args.selfreweightCalibrate:
                    output_aux = output_aux.sigmoid().squeeze()
                    # bs x 1, Calibration function A \in [0,1]
                    robust_output_aux = robust_output_aux.sigmoid().squeeze()
                    test_evi_all = con_pre * output_aux
                    test_robust_evi_all = robust_con_pre * robust_output_aux
                    if args.ConfidenceOnly:
                        test_evi_all = con_pre
                        test_robust_evi_all = robust_con_pre
                    if args.AuxiliaryOnly:
                        test_evi_all = output_aux
                        test_robust_evi_all = robust_output_aux

                else:
                    raise NotImplementedError()

            else:
                raise NotImplementedError()

            # output labels
            labels = torch.where(output.max(1)[1] == y)[0]
            robust_labels = torch.where(robust_output.max(1)[1] == y)[0]

            # accuracy
            test_acc += labels.size(0)
            test_robust_acc += robust_labels.size(0)

            # standard evidence
            test_evi_correct += test_evi_all[labels].sum().item()
            test_evi_wrong += test_evi_all.sum().item() - \
                test_evi_all[labels].sum().item()

            # robust evidence
            test_robust_evi_correct += test_robust_evi_all[robust_labels].sum(
            ).item()
            test_robust_evi_wrong += test_robust_evi_all.sum().item(
            ) - test_robust_evi_all[robust_labels].sum().item()

            test_n += y.size(0)

        logger.info('%d \t %.4f \t %.4f', epoch,
                    test_acc / test_n, test_robust_acc / test_n)

        # save best
        if test_robust_acc / test_n > best_test_robust_acc:
            torch.save({
                'state_dict': model.module.state_dict(),
                'test_robust_acc': test_robust_acc / test_n,
                'test_acc': test_acc / test_n,
            }, os.path.join(args.fname, f'model_best.pth'))
            best_test_robust_acc = test_robust_acc / test_n

    # calculate AUC
    model_dict = torch.load(os.path.join(args.fname, f'model_best.pth'))
    logger.info(f'Resuming at best epoch')

    if 'state_dict' in model_dict.keys():
        model.load_state_dict(model_dict['state_dict'])
    else:
        model.load_state_dict(model_dict)

    model.eval()

    test_acc = 0
    test_robust_acc = 0
    test_n = 0
    test_classes_correct = []
    test_classes_wrong = []
    test_classes_robust_correct = []
    test_classes_robust_wrong = []

    # record con
    test_con_correct = []
    test_robust_con_correct = []
    test_con_wrong = []
    test_robust_con_wrong = []

    # record evi
    test_evi_correct = []
    test_robust_evi_correct = []
    test_evi_wrong = []
    test_robust_evi_wrong = []

    for i, (data, target) in enumerate(test_batches):
        X, y = data.cuda(), target.cuda()

        if not args.target:
            delta, _ = attack_pgd(
                model, X, y, epsilon, pgd_alpha, args.attack_iters, args.
                restarts, args.norm, twobranch=args.twobranch,
                normalize=normalize)
        else:
            raise NotImplementedError()

        delta = delta.detach()

        if args.twobranch:
            output, output_aux = model(normalize(X))
            robust_output, robust_output_aux = model(
                normalize(
                    torch.clamp(
                        X + delta,
                        min=lower_limit,
                        max=upper_limit)))

            # predicted label and confidence
            con_pre, _ = torch.softmax(output * args.tempC, dim=1).max(1)
            # predicted label and confidence
            robust_con_pre, _ = torch.softmax(
                robust_output * args.tempC, dim=1).max(1)

            if args.selfreweightCalibrate:
                output_aux = output_aux.sigmoid().squeeze()
                # bs x 1, Calibration function A \in [0,1]
                robust_output_aux = robust_output_aux.sigmoid().squeeze()
                test_evi_all = con_pre * output_aux
                test_robust_evi_all = robust_con_pre * robust_output_aux
                if args.ConfidenceOnly:
                    test_evi_all = con_pre
                    test_robust_evi_all = robust_con_pre
                if args.AuxiliaryOnly:
                    test_evi_all = output_aux
                    test_robust_evi_all = robust_output_aux

            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

        output_s = F.softmax(output, dim=1)
        out_con, out_pre = output_s.max(1)

        ro_output_s = F.softmax(robust_output, dim=1)
        ro_out_con, ro_out_pre = ro_output_s.max(1)

        # output labels
        labels = torch.where(out_pre == y)[0]
        robust_labels = torch.where(ro_out_pre == y)[0]
        labels_n = torch.where(out_pre != y)[0]
        robust_labels_n = torch.where(ro_out_pre != y)[0]

        # ground labels
        test_classes_correct += y[labels].tolist()
        test_classes_wrong += y[labels_n].tolist()
        test_classes_robust_correct += y[robust_labels].tolist()
        test_classes_robust_wrong += y[robust_labels_n].tolist()

        # accuracy
        test_acc += labels.size(0)
        test_robust_acc += robust_labels.size(0)

        # confidence
        test_con_correct += out_con[labels].tolist()
        test_con_wrong += out_con[labels_n].tolist()
        test_robust_con_correct += ro_out_con[robust_labels].tolist()
        test_robust_con_wrong += ro_out_con[robust_labels_n].tolist()

        # evidence
        test_evi_correct += test_evi_all[labels].tolist()
        test_evi_wrong += test_evi_all[labels_n].tolist()
        test_robust_evi_correct += test_robust_evi_all[robust_labels].tolist(
        )
        test_robust_evi_wrong += test_robust_evi_all[robust_labels_n].tolist()

        test_n += y.size(0)

        print('Finish ', i)

    # confidence
    test_con_correct = torch.tensor(test_con_correct)
    test_robust_con_correct = torch.tensor(test_robust_con_correct)
    test_con_wrong = torch.tensor(test_con_wrong)
    test_robust_con_wrong = torch.tensor(test_robust_con_wrong)

    # evidence
    test_evi_correct = torch.tensor(test_evi_correct)
    test_robust_evi_correct = torch.tensor(test_robust_evi_correct)
    test_evi_wrong = torch.tensor(test_evi_wrong)
    test_robust_evi_wrong = torch.tensor(test_robust_evi_wrong)

    print('### Basic statistics ###')
    logger.info(
        'Clean       | acc: %.4f | con cor: %.3f (%.3f) | con wro: %.3f (%.3f) | evi cor: %.3f (%.3f) | evi wro: %.3f (%.3f)',
        test_acc / test_n, test_con_correct.mean().item(),
        test_con_correct.std().item(),
        test_con_wrong.mean().item(),
        test_con_wrong.std().item(),
        test_evi_correct.mean().item(),
        test_evi_correct.std().item(),
        test_evi_wrong.mean().item(),
        test_evi_wrong.std().item())

    logger.info(
        'Robust      | acc: %.4f | con cor: %.3f (%.3f) | con wro: %.3f (%.3f) | evi cor: %.3f (%.3f) | evi wro: %.3f (%.3f)',
        test_robust_acc / test_n, test_robust_con_correct.mean().item(),
        test_robust_con_correct.std().item(),
        test_robust_con_wrong.mean().item(),
        test_robust_con_wrong.std().item(),
        test_robust_evi_correct.mean().item(),
        test_robust_evi_correct.std().item(),
        test_robust_evi_wrong.mean().item(),
        test_robust_evi_wrong.std().item())

    test_acc = test_acc / test_n
    test_robust_acc = test_robust_acc / test_n
    print('')
    print('### ROC-AUC scores (confidence) ###')
    # clean_clean = calculate_auc_scores(test_con_correct, test_con_wrong)
    # robust_robust = calculate_auc_scores(test_robust_con_correct, test_robust_con_wrong)
    # logger.info('clean_clean: %.3f | robust_robust: %.3f',
    #     clean_clean, robust_robust)
    clean_clean = calculate_auc_scores(test_con_correct, test_con_wrong)
    _, acc95 = calculate_FPR_TPR(
        test_con_correct, test_con_wrong, tpr_ref=0.95)
    _, acc99 = calculate_FPR_TPR(
        test_con_correct, test_con_wrong, tpr_ref=0.99)
    robust_robust = calculate_auc_scores(
        test_robust_con_correct, test_robust_con_wrong)
    _, ro_acc95 = calculate_FPR_TPR(
        test_robust_con_correct, test_robust_con_wrong, tpr_ref=0.95)
    _, ro_acc99 = calculate_FPR_TPR(
        test_robust_con_correct, test_robust_con_wrong, tpr_ref=0.99)
    logger.info('clean_clean: %.3f | robust_robust: %.3f',
                clean_clean, robust_robust)
    logger.info(
        'TPR 95 clean acc: %.4f; 99 clean acc: %.4f | TPR 95 robust acc: %.4f; 99 robust acc: %.4f',
        acc95 - test_acc, acc99 - test_acc, ro_acc95 - test_robust_acc,
        ro_acc99 - test_robust_acc)

    print('')
    print('### ROC-AUC scores (evidence) ###')
    # clean_clean = calculate_auc_scores(test_evi_correct, test_evi_wrong)
    # robust_robust = calculate_auc_scores(test_robust_evi_correct, test_robust_evi_wrong)
    # logger.info('clean_clean: %.3f | robust_robust: %.3f',
    #     clean_clean, robust_robust)
    clean_clean = calculate_auc_scores(test_evi_correct, test_evi_wrong)
    _, acc95 = calculate_FPR_TPR(
        test_evi_correct, test_evi_wrong, tpr_ref=0.95)
    _, acc99 = calculate_FPR_TPR(
        test_evi_correct, test_evi_wrong, tpr_ref=0.99)
    robust_robust = calculate_auc_scores(
        test_robust_evi_correct, test_robust_evi_wrong)
    _, ro_acc95 = calculate_FPR_TPR(
        test_robust_evi_correct, test_robust_evi_wrong, tpr_ref=0.95)
    _, ro_acc99 = calculate_FPR_TPR(
        test_robust_evi_correct, test_robust_evi_wrong, tpr_ref=0.99)
    logger.info('clean_clean: %.3f | robust_robust: %.3f',
                clean_clean, robust_robust)
    logger.info(
        'TPR 95 clean acc: %.4f; 99 clean acc: %.4f | TPR 95 robust acc: %.4f; 99 robust acc: %.4f',
        acc95 - test_acc, acc99 - test_acc, ro_acc95 - test_robust_acc,
        ro_acc99 - test_robust_acc)


if __name__ == "__main__":
    main()
