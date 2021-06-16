import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

#####################
# Auto file name
#####################


def get_auto_fname(args):
    if args.ATframework == 'PGDAT':
        names = 'PGDAT_' + args.model_name
    elif args.ATframework == 'TRADES':
        names = 'TRADES' + str(args.TRADESlambda) + '_' + args.model_name
    elif args.ATframework == 'CCAT':
        names = 'CCAT_ratio' + str(args.CCATratio) + 'iter' + str(args.CCATiter) + 'rho' + str(
            args.CCATrho) + 'step' + str(args.CCATstep) + '_' + args.model_name

    if args.useBN:
        names += 'BN'
    if args.adaptiveattack:
        names += '_adaptiveA' + str(args.adaptiveattacklambda)
        if args.uniform_lambda:
            names += 'uniform'
        if args.twosign:
            names += 'twosign'
    if args.BNeval:
        names += '_BNeval'

    if args.adaptivetrain:
        names += '_adaptiveT' + str(args.adaptivetrainlambda)
        if args.selfreweightCalibrate:
            names += '_selfreweightCalibrate'
            if args.tempC != 1.:
                names += '_tempC' + str(args.tempC)
            if args.tempC_trueonly != 1.:
                names += '_tempCtrueonly' + str(args.tempC_trueonly)
            if args.SGconfidenceW:
                names += '_SGconW'
            if args.ConfidenceOnly:
                names += '_ConfidenceOnly'
            if args.AuxiliaryOnly:
                names += '_AuxiliaryOnly'

        elif args.selfreweightNIPS20:
            names += '_selfreweightNIPS20' + 'mi' + \
                str(args.m_in) + 'mo' + str(args.m_out)

        elif args.selfreweightATRO:
            names += '_selfreweightATRO' + 'cost' + str(
                args.ATRO_cost) + 'coe' + str(args.ATRO_coefficient)

        elif args.selfreweightCARL:
            names += '_selfreweightCARL' + 'lambda' + str(
                args.CARL_lambda) + 'eta' + str(args.CARL_eta)
            #names += '_selfreweightCARL' + 'lambda' + str(args.CARL_lambda)

    if args.weight_decay != 5e-4:
        names = names + '_wd' + str(args.weight_decay)
    if args.epochs != 110:
        names += '_epochs' + str(args.epochs)
    if args.batch_size != 128:
        names += '_bs' + str(args.batch_size)
    if args.epsilon != 8:
        names += '_eps' + str(args.epsilon)
    if args.CCATscale != 1:
        names += '_scale' + str(args.CCATscale)
    names += '_seed' + str(args.seed)
    print('File name: ', names)
    return names


def calculate_auc_scores(correct, wrong):
    labels_all = torch.cat(
        (torch.ones_like(correct),
         torch.zeros_like(wrong)),
        dim=0).cpu().numpy()
    prediction_all = torch.cat((correct, wrong), dim=0).cpu().numpy()
    return roc_auc_score(labels_all, prediction_all)


def calculate_FPR_TPR(correct, wrong, tpr_ref=0.95):
    labels_all = torch.cat(
        (torch.ones_like(correct),
         torch.zeros_like(wrong)),
        dim=0).cpu().numpy()
    prediction_all = torch.cat((correct, wrong), dim=0).cpu().numpy()
    fpr, tpr, thresholds = roc_curve(labels_all, prediction_all)
    index = np.argmin(np.abs(tpr - tpr_ref))
    T = thresholds[index]
    FPR_thred = fpr[index]
    index_c = (torch.where(correct > T)[0]).size(0)
    index_w = (torch.where(wrong > T)[0]).size(0)
    acc = index_c / (index_c + index_w)
    return FPR_thred, acc
