from __future__ import print_function
from math import log
from xml.etree.ElementInclude import default_loader

import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.utils.data.distributed
import torch.distributed as dist

from loader import custom_dataloader

from models.network import get_network

from utils.dir_maker import DirectroyMaker
from utils.AverageMeter import AverageMeter
from utils.color import Colorer
from utils.etc import progress_bar, is_main_process, save_on_master, paser_config_save, set_logging_defaults
from utils.label_dynamic import *

#----------------------------------------------------
#  Etc
#----------------------------------------------------
import os, logging, time
import argparse
import numpy as np
import csv  # [DTSKD-PLOT] 用于记录训练指标到CSV


#----------------------------------------------------
#  Training Setting parser
#----------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Progressive Self-Knowledge Distillation : PS-KD')
    #parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    #######
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_rate', default=0.1, type=float, help='learning rate decay rate')
    parser.add_argument('--lr_decay_schedule', default=[150, 225], nargs='*', type=int, help='when to drop lr')
    #######
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight_decay')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number')
    parser.add_argument('--end_epoch', default=300, type=int, help='number of training epoch to run')

    parser.add_argument('--backbone_weight', default=3.0, type=float)
    parser.add_argument('--b1_weight', default=1.0, type=float)
    parser.add_argument('--b2_weight', default=1.0, type=float)
    parser.add_argument('--b3_weight', default=1.0, type=float)

    parser.add_argument('--ce_weight', default=1.0, type=float)
    parser.add_argument('--kd_weight', default=1.0, type=float)
    parser.add_argument('--coeff_decay', default='linear', type=str, help='coefficient decay')

    parser.add_argument('--cos_max', default=1.0, type=float)
    parser.add_argument('--cos_min', default=0.0, type=float)

    parser.add_argument('--HSKD', type=int, default=0, help='history KD')
    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
    parser.add_argument('--experiments_dir', type=str, default='models',help='Directory name to save the model, log, config')
    parser.add_argument('--experiments_name', type=str, default='baseline')
    parser.add_argument('--classifier_type', type=str, default='ResNet18', help='Select classifier')
    parser.add_argument('--data_path', type=str, default=None, help='download dataset path')
    parser.add_argument('--data_type', type=str, default=None, help='type of dataset')
    parser.add_argument('--alpha_T',default=0.8 ,type=float, help='alpha_T')
    parser.add_argument('--alpha_end_epoch', default=-1, type=int, help='epoch count for alpha_t schedule (default: same as end_epoch)')
    parser.add_argument('--saveckp_freq', default=299, type=int, help='Save checkpoint every x epochs. Last model saving set to 299')
    parser.add_argument('--rank', default=-1, type=int,help='node rank for distributed training')
    parser.add_argument('--world_size', default=1, type=int,help='number of distributed processes')
    parser.add_argument('--dist_backend', default='nccl', type=str,help='distributed backend')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:8080', type=str,help='url used to set up distributed training')
    parser.add_argument('--workers', default=8, type=int,help='url used to set up distributed training')
    parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    parser.add_argument('--resume', type=str, default='', help='load model path')
    parser.add_argument('--random_seed', type=int, default=27)
    parser.add_argument('--tsne', type=int, default=0)
    parser.add_argument('--track_forgetting', type=int, default=0,
                        help='Track per-sample correctness on val set for forgetting analysis')
    parser.add_argument('--ewc_lambda', type=float, default=0.0,
                        help='EWC regularization strength (0=disabled)')
    parser.add_argument('--ewc_freq', type=int, default=5,
                        help='Update Fisher information every N epochs')
    # [Anti-Forgetting SKD] 置信度加权的输出空间 KL 约束
    parser.add_argument('--af_lambda', type=float, default=0.0,
                        help='Anti-Forgetting SKD: KL regularization strength (0=disabled)')
    parser.add_argument('--af_alpha', type=float, default=2.0,
                        help='Anti-Forgetting SKD: exponential scale factor for margin→weight mapping')
    parser.add_argument('--af_tau', type=float, default=0.2,
                        help='Anti-Forgetting SKD: neutral threshold (m_i=tau → w_i=1)')
    parser.add_argument('--noise_rate', type=float, default=0.0,
                        help='Symmetric label noise rate on training set (0.0 = clean, 0.2 = 20%%, 0.4 = 40%%)')
    args = parser.parse_args()
    return check_args(args)


def check_args(args):
    # --epoch
    try:
        assert args.end_epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args
    

#----------------------------------------------------
# find free gpu id from multi-gpu server
#----------------------------------------------------
def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)  


#----------------------------------------------------
#  Adjust_learning_rate & get_learning_rate  
#----------------------------------------------------
def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr

    for milestone in args.lr_decay_schedule:
        lr *= args.lr_decay_rate if epoch >= milestone else 1.
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr

#----------------------------------------------------
#  Top-1 / Top -5 accuracy
#----------------------------------------------------
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


C = Colorer.instance()


def main():
    args = parse_args()
    # args.multiprocessing_distributed = True
    args.multiprocessing_distributed = False

    print(C.green("[!] Start the DTSKD."))

    dir_maker = DirectroyMaker(root = args.experiments_dir,save_model=True, save_log=True,save_config=True)
    model_log_config_dir = dir_maker.experiments_dir_maker(args)
    
    model_dir = model_log_config_dir[0]
    log_dir = model_log_config_dir[1]
    config_dir = model_log_config_dir[2]

    paser_config_save(args,config_dir)
    
    import random
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node,args=(ngpus_per_node,model_dir,log_dir,args))
        print(C.green("[!] Multi/Single Node, Multi-GPU All multiprocessing_distributed Training Done."))
        print(C.underline(C.red2('[Info] Save Model dir:')),C.red2(model_dir))
        print(C.underline(C.red2('[Info] Log dir:')),C.red2(log_dir))
        print(C.underline(C.red2('[Info] Config dir:')),C.red2(config_dir))
    else:
        available_gpu = 0  # 指定gpu id
        # print(C.underline(C.yellow("[Info] Finding Empty GPU {}".format(int(get_freer_gpu())))))
        main_worker(available_gpu, ngpus_per_node, model_dir,log_dir,args)
        print(C.green("[!] All Single GPU Training Done"))
        print(C.underline(C.red2('[Info] Save Model dir:')),C.red2(model_dir))
        print(C.underline(C.red2('[Info] Log dir:')),C.red2(log_dir))
        print(C.underline(C.red2('[Info] Config dir:')),C.red2(config_dir))
        

def main_worker(gpu,ngpus_per_node,model_dir,log_dir,args):
    best_acc = 0

    net = get_network(args)

    args.ngpus_per_node = ngpus_per_node
    args.gpu = gpu
    if args.gpu is not None:
        print(C.underline(C.yellow("[Info] Use GPU : {} for training".format(args.gpu))))
    
    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,world_size=args.world_size, rank=args.rank)
    print(C.green("[!] [Rank {}] Distributed Init Setting Done.".format(args.rank)))
    
    if not torch.cuda.is_available():
        print(C.red2("[Warnning] Using CPU, this will be slow."))
        
    elif args.distributed:
        if args.gpu is not None:
            print(C.green("[!] [Rank {}] Distributed DataParallel Setting Start".format(args.rank)))
            
            torch.cuda.set_device(args.gpu)
            net.cuda(args.gpu)

            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.workers = int((args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
            args.batch_size = int(args.batch_size / args.ngpus_per_node)
            
            print(C.underline(C.yellow("[Info] [Rank {}] Workers: {}".format(args.rank, args.workers))))
            print(C.underline(C.yellow("[Info] [Rank {}] Batch_size: {}".format(args.rank, args.batch_size))))
            
            net = torch.nn.parallel.DistributedDataParallel(net,device_ids=[args.gpu])
            print(C.green("[!] [Rank {}] Distributed DataParallel Setting End".format(args.rank)))
            
        else:
            net.cuda()
            net = torch.nn.parallel.DistributedDataParallel(net)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        net = net.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        net = torch.nn.DataParallel(net).cuda()

    set_logging_defaults(log_dir, args)

    train_loader, valid_loader, train_sampler = custom_dataloader.dataloader(args)

    criterion_CE = nn.CrossEntropyLoss().cuda(args.gpu)
    # criterion_CE = KL_Loss2(temperature=1)
    criterion_KD = KL_Loss(temperature=4)
    if args.HSKD:
        criterion_CE_hskd = KL_Loss2(temperature=1).cuda(args.gpu)
        criterion_KD_hskd = KL_Loss(temperature=4).cuda(args.gpu)
    else:
        criterion_CE_hskd = None
        criterion_KD_hskd = None
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    # resume
    if args.resume != '':

        checkpoint = torch.load(args.resume, map_location='cuda')

        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        args.start_epoch = checkpoint['epoch'] + 1

        if 'best_acc' in checkpoint:
            best_acc = checkpoint['best_acc']

        print(f"Resume from epoch {args.start_epoch}")

    #----------------------------------------------------
    #  Empty matrix for store predictions
    #----------------------------------------------------
    if args.HSKD:
        # [FIX] 如果resume且checkpoint包含预测矩阵，则恢复；否则初始化为零
        if args.resume != '' and 'all_predictions' in checkpoint:
            all_predictions = checkpoint['all_predictions'].cpu()
            b1_predictions = checkpoint['b1_predictions'].cpu()
            b2_predictions = checkpoint['b2_predictions'].cpu()
            b3_predictions = checkpoint['b3_predictions'].cpu()
            print(C.underline(C.yellow("[Info] all_predictions matrix restored from checkpoint, shape {} ".format(all_predictions.shape))))
        else:
            all_predictions = torch.zeros(len(train_loader.dataset), len(train_loader.dataset.classes), dtype=torch.float32)
            b1_predictions = torch.zeros(len(train_loader.dataset), len(train_loader.dataset.classes), dtype=torch.float32)
            b2_predictions = torch.zeros(len(train_loader.dataset), len(train_loader.dataset.classes), dtype=torch.float32)
            b3_predictions = torch.zeros(len(train_loader.dataset), len(train_loader.dataset.classes), dtype=torch.float32)
            print(C.underline(C.yellow("[Info] all_predictions matrix shape {} ".format(all_predictions.shape))))
    else:
        all_predictions = None
        b1_predictions = None
        b2_predictions = None
        b3_predictions = None

    # [MCW-AF v3] Per-sample prediction stability (EMA of correctness)
    # stability[i] ≈ 样本 i 最近被正确分类的频率
    # 用于自适应 AF 强度: 稳定样本强保护, 振荡样本弱/不保护
    if args.HSKD and args.af_lambda > 0.0:
        if args.resume != '' and 'stability' in checkpoint:
            stability = checkpoint['stability'].cpu()
        else:
            stability = torch.zeros(len(train_loader.dataset), dtype=torch.float32)
    else:
        stability = None

    #----------------------------------------------------
    #  load status & Resume Learning
    #----------------------------------------------------
    if args.resume:

        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        else:
            # Map model to be loaded to specified single gpu.
            if args.distributed:
                dist.barrier()
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)

        args.start_epoch = checkpoint['epoch'] + 1
        alpha_t = checkpoint['alpha_t'] if 'alpha_t' in checkpoint else 0.9
        best_acc = checkpoint['best_acc']
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # [FIX] 恢复HSKD历史预测矩阵
        if args.HSKD and 'all_predictions' in checkpoint:
            all_predictions = checkpoint['all_predictions'].cpu()
            b1_predictions = checkpoint['b1_predictions'].cpu()
            b2_predictions = checkpoint['b2_predictions'].cpu()
            b3_predictions = checkpoint['b3_predictions'].cpu()
        print(C.green("[!] [Rank {}] Model loaded".format(args.rank)))

        del checkpoint


    cudnn.benchmark = True
    PI = math.acos(-1.0)

    best_acc=0
    logger = logging.getLogger('best')

    # [DTSKD-PLOT] 创建CSV文件记录每个epoch的指标，用于后续绘图
    csv_path = os.path.join(log_dir, 'metrics.csv')
    csv_mode = 'a' if args.resume else 'w'
    csv_file = open(csv_path, csv_mode, newline='')
    csv_writer = csv.writer(csv_file)
    if not args.resume:
        csv_writer.writerow(['epoch', 'lr', 'alpha_t',
                             'train_loss', 'train_top1', 'train_top5',
                             'val_top1', 'val_top5',
                             'val_b1_top1', 'val_b2_top1', 'val_b3_top1',
                             'af_loss'])

    # [FORGETTING] 初始化 per-sample correctness 存储
    if args.track_forgetting:
        num_val_samples = len(valid_loader.dataset)
        per_sample_correct = torch.zeros(num_val_samples, dtype=torch.bool)
        forgetting_dir = os.path.join(log_dir, 'forgetting')
        os.makedirs(forgetting_dir, exist_ok=True)

    # [EWC] 初始化: 第一个 epoch 后, 用当前参数作为参考点 θ*
    global _ewc_ref_params, _ewc_fisher

    for epoch in range(args.start_epoch, args.end_epoch):

        epoch_start = time.time()

        # if args.tsne:
        out_list = []
        target_list = []

        adjust_learning_rate(optimizer, epoch, args)
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if args.HSKD:
            _alpha_epochs = args.alpha_end_epoch if args.alpha_end_epoch > 0 else args.end_epoch
            if args.coeff_decay == 'linear':
                alpha_t = args.alpha_T * ((epoch + 1) / _alpha_epochs)
                alpha_t = max(0, alpha_t)
                alpha_t = 1 - alpha_t
            elif args.coeff_decay == 'cos':
                ratio = 1.0 * epoch / _alpha_epochs
                scale = (math.cos(ratio * PI) + 1.) / 2
                momentum_label_final = args.cos_min
                momentum_label_range = args.cos_max - args.cos_min
                alpha_t = scale * momentum_label_range + momentum_label_final

        else:
            alpha_t = -1

        all_predictions, b1_predictions, b2_predictions, b3_predictions, stability, train_metrics = train(
                                all_predictions,
                                b1_predictions,
                                b2_predictions,
                                b3_predictions,
                                stability,
                                criterion_CE,
                                criterion_CE_hskd,
                                criterion_KD,
                                criterion_KD_hskd,
                                optimizer,
                                net,
                                epoch,
                                alpha_t,
                                train_loader,
                                args)

        # dist.barrier()
        if args.track_forgetting:
            acc, val_metrics, per_sample_correct = val(
                      net,
                      epoch,
                      valid_loader,
                      out_list,
                      target_list,
                      args,
                      per_sample_correct=per_sample_correct,
                      num_val_samples=num_val_samples)
            # 每个 epoch 保存 per-sample correctness
            torch.save(per_sample_correct.clone(),
                       os.path.join(forgetting_dir, f'correct_epoch_{epoch:04d}.pt'))
        else:
            acc, val_metrics = val(
                      net,
                      epoch,
                      valid_loader,
                      out_list,
                      target_list,
                      args)
        # ===== [EWC] 更新 Fisher 信息和参考参数 =====
        if args.ewc_lambda > 0 and args.track_forgetting:
            update_interval = max(1, args.ewc_freq)
            if epoch % update_interval == 0 and epoch >= args.ewc_freq:
                # 用当前正确分类的验证集样本计算 Fisher
                correct_indices = per_sample_correct.nonzero(as_tuple=True)[0]
                if len(correct_indices) > 0:
                    _ewc_fisher = compute_ewc_fisher(
                        net, valid_loader, criterion_CE, args,
                        num_samples=min(2000, len(correct_indices)))
                    _ewc_ref_params = update_ewc_ref(net)
                    print(f'[EWC] Updated Fisher & Ref Params at epoch {epoch} '
                          f'(on {len(correct_indices)} correct samples)')

        # ===== 仅最后一个 epoch 保存 checkpoint 到输出目录 =====
        if epoch == args.end_epoch - 1:
            checkpoint_dir = os.path.join(args.experiments_dir, args.experiments_name, 'checkpoint')
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_dict = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc': acc,
                'best_acc': best_acc,
                'alpha_t': alpha_t if args.HSKD else -1
            }
            if args.HSKD:
                save_dict['all_predictions'] = all_predictions
                save_dict['b1_predictions'] = b1_predictions
                save_dict['b2_predictions'] = b2_predictions
                save_dict['b3_predictions'] = b3_predictions
            if stability is not None:
                save_dict['stability'] = stability
            torch.save(save_dict, os.path.join(checkpoint_dir, 'latest_checkpoint.pth'))

        # [DTSKD-PLOT] 将当前epoch指标写入CSV
        csv_writer.writerow([epoch,
                             train_metrics['lr'], train_metrics['alpha_t'],
                             train_metrics['loss'], train_metrics['top1'], train_metrics['top5'],
                             val_metrics['val_top1'], val_metrics['val_top5'],
                             val_metrics['val_b1_top1'], val_metrics['val_b2_top1'], val_metrics['val_b3_top1'],
                             train_metrics['af_loss']])
        csv_file.flush()

        epoch_time = time.time() - epoch_start
        print(f"[Epoch {epoch} Timing] {epoch_time:.1f}s total")
        if epoch == args.start_epoch:
            print(f"[Estimate] ~{args.end_epoch * epoch_time / 3600:.1f} hours for {args.end_epoch} epochs at this speed")

        print(f"[Checkpoint Saved] Epoch {epoch}")

        if epoch > 50:
            if acc > best_acc:
                best_acc = acc
                # print('')
                logger.info('-------best acc-------')
                if args.tsne:
                    savepickle([torch.cat(out_list), torch.cat(target_list)], os.path.join(args.experiments_dir, args.experiments_name, 'baseline_res18_logits.pkl'))

    # [DTSKD-PLOT] 关闭CSV文件
    csv_file.close()

    # cleanup()
    print(C.green("[!] [Rank {}] Distroy Distributed process".format(args.rank)))


from torch.cuda.amp import GradScaler as GradScaler

# ============================================================
# EWC (Elastic Weight Consolidation) for SKD
# ============================================================
def compute_ewc_fisher(net, dataloader, criterion_ce, args, num_samples=2000):
    """
    计算 Fisher 信息矩阵对角线，仅对当前正确分类的样本。

    数学: F_i = E_x[(∂L_CE/∂θ_i)^2]
    仅对 ŷ(x) == y 的样本计算，保护"已知正确知识"不被遗忘。

    返回: {name: fisher_diagonal_tensor}
    """
    net.eval()
    fisher = {}
    for name, param in net.named_parameters():
        fisher[name] = torch.zeros_like(param)

    count = 0
    for inputs, targets, _ in dataloader:
        if count >= num_samples:
            break
        if args.gpu is not None:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        net.zero_grad()
        outputs, _, _, _ = net(inputs)
        loss = criterion_ce(outputs, targets)
        loss.backward()

        # 只累积正确分类样本的 Fisher
        _, predicted = outputs.max(1)
        correct_mask = (predicted == targets).float()

        for name, param in net.named_parameters():
            if param.grad is not None:
                # F_i += (∂L/∂θ_i)^2，对正确样本加权
                grad_sq = param.grad.data ** 2
                # 按样本平均（简化：用 batch 内正确样本比例缩放）
                weight = correct_mask.mean()
                fisher[name] += grad_sq * weight

        count += inputs.size(0)

    # 归一化
    for name in fisher:
        fisher[name] /= max(count / inputs.size(0), 1)  # 按 batch 数平均

    net.train()
    return fisher


def ewc_loss(net, fisher, ref_params, args):
    """
    计算 EWC 正则化损失。

    L_EWC = (λ/2) * Σ_i F_i * (θ_i - θ_i*)^2

    使用教师置信度加权: λ_eff = λ * max_c p_t(c|x)
    """
    if args.ewc_lambda <= 0 or not fisher:
        return torch.tensor(0.0).cuda()

    loss = 0.0
    for name, param in net.named_parameters():
        if name in fisher and name in ref_params:
            loss += (fisher[name] * (param - ref_params[name]) ** 2).sum()

    return 0.5 * args.ewc_lambda * loss


def update_ewc_ref(net):
    """保存当前参数作为 EWC 参考点 θ*"""
    ref = {}
    for name, param in net.named_parameters():
        ref[name] = param.data.clone()
    return ref
from torch.cuda.amp import autocast as autocast

scaler = GradScaler()

# EWC 全局状态 (避免修改 train() 函数签名)
_ewc_ref_params = {}
_ewc_fisher = {}

def train(all_predictions,
        b1_predictions,
        b2_predictions,
        b3_predictions,
        stability,
          criterion_CE,
          criterion_CE_hskd,
          criterion_KD,
          criterion_KD_hskd,
          optimizer,
          net,
          epoch,
          alpha_t,
          train_loader,
          args):
    
    train_top1 = AverageMeter()
    train_top5 = AverageMeter()
    train_losses = AverageMeter()
    train_af_losses = AverageMeter()  # Anti-Forgetting 加权 KL 损失
    
    # correct = 0
    # total = 0

    net.train()
    current_LR = get_learning_rate(optimizer)[0]

    for batch_idx, (inputs, targets, input_indices) in enumerate(train_loader):
        optimizer.zero_grad()
        
        if args.gpu is not None:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        # =====================================================================
        # [DTSKD-核心] 历史知识蒸馏 (HSKD: Historical Self-KD) 分支
        # =====================================================================
        if args.HSKD:
            # --- 1. 构造 one-hot 标签 ---
            targets_numpy = targets.cpu().detach().numpy()
            identity_matrix = torch.eye(len(train_loader.dataset.classes))
            targets_one_hot = identity_matrix[targets_numpy]  # [batch, 100]

            # Epoch 0: 用 one-hot 初始化历史预测矩阵 (尚无历史)
            if epoch == 0:
                all_predictions[input_indices] = targets_one_hot
                b1_predictions[input_indices] = targets_one_hot
                b2_predictions[input_indices] = targets_one_hot
                b3_predictions[input_indices] = targets_one_hot

            # --- 2. 生成历史软目标 (EMA 混合) ---
            # soft_target = α_t × one_hot + (1-α_t) × last_epoch_prediction
            # α_t 随 epoch 从 0.9→0.0 衰减，即越训练越依赖上一轮的预测
            soft_targets = (alpha_t * targets_one_hot) + ((1 - alpha_t) * all_predictions[input_indices])
            soft_targets = torch.autograd.Variable(soft_targets).cuda()

            soft_b1_targets = (alpha_t * targets_one_hot) + ((1 -alpha_t) * b1_predictions[input_indices])
            soft_b1_targets = torch.autograd.Variable(soft_b1_targets).cuda()

            soft_b2_targets = (alpha_t * targets_one_hot) + ((1 - alpha_t) * b2_predictions[input_indices])
            soft_b2_targets = torch.autograd.Variable(soft_b2_targets).cuda()

            soft_b3_targets = (alpha_t * targets_one_hot) + ((1 - alpha_t) * b3_predictions[input_indices])
            soft_b3_targets = torch.autograd.Variable(soft_b3_targets).cuda()

            inputs = torch.autograd.Variable(inputs, requires_grad=True)    
            
            # --- 3. 混合精度前向传播 (AMP) ---
            with autocast():
                # 模型输出: backbone, branch1, branch2, branch3 四组预测
                outputs, b1_output, b2_output, b3_output = net(inputs)

                # softmax 用于历史预测更新 & 分布式 all_gather
                softmax_output = F.softmax(outputs, dim=1)
                b1_softmax_out = F.softmax(b1_output, dim=1)
                b2_softmax_out = F.softmax(b2_output, dim=1)
                b3_softmax_out = F.softmax(b3_output, dim=1)

                # ============================================================
                # CE Loss (交叉熵 / 与历史软目标的 KL 散度)
                # 每个输出头分别与自己的历史软目标计算 CE loss
                # backbone_weight=3.0 表示主干损失权重是分支的 3 倍
                # ============================================================
                loss_ce = args.backbone_weight * criterion_CE_hskd(outputs, soft_targets)
                loss_ce += args.b1_weight * criterion_CE_hskd(b1_output, soft_b1_targets)
                loss_ce += args.b2_weight * criterion_CE_hskd(b2_output, soft_b2_targets)
                loss_ce += args.b3_weight * criterion_CE_hskd(b3_output, soft_b3_targets)

                # ============================================================
                # KD Loss (结构知识蒸馏)
                # b1/b2/b3 三个分支对齐 backbone 主干的输出
                # backbone 是"结构教师" (Structural Teacher)
                # 分支是"学生" — 学习 backbone 的软标签分布
                # ============================================================
                loss_kd = criterion_KD_hskd(b1_output, outputs)   # 浅层分支 → backbone
                loss_kd += criterion_KD_hskd(b2_output, outputs)  # 中层分支 → backbone
                loss_kd += criterion_KD_hskd(b3_output, outputs)  # 深层分支 → backbone

                if args.distributed:
                    gathered_prediction = [torch.ones_like(softmax_output) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered_prediction, softmax_output)
                    gathered_prediction = torch.cat(gathered_prediction, dim=0)

                    b1_gathered_pre = [torch.ones_like(b1_softmax_out) for _ in range(dist.get_world_size())]
                    dist.all_gather(b1_gathered_pre, b1_softmax_out)
                    b1_gathered_pre = torch.cat(b1_gathered_pre, dim=0)

                    b2_gathered_pre = [torch.ones_like(b2_softmax_out) for _ in range(dist.get_world_size())]
                    dist.all_gather(b2_gathered_pre, b2_softmax_out)
                    b2_gathered_pre = torch.cat(b2_gathered_pre, dim=0)

                    b3_gathered_pre = [torch.ones_like(b3_softmax_out) for _ in range(dist.get_world_size())]
                    dist.all_gather(b3_gathered_pre, b3_softmax_out)
                    b3_gathered_pre = torch.cat(b3_gathered_pre, dim=0)

                    gathered_indices = [torch.ones_like(input_indices.cuda()) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered_indices, input_indices.cuda())
                    gathered_indices = torch.cat(gathered_indices, dim=0)

        # =====================================================================
        # [DTSKD] 无历史蒸馏 (HSKD=0): 纯结构蒸馏 + 标准CE
        # =====================================================================
        else:
            outputs, b1_output, b2_output, b3_output = net(inputs)

            # 标准交叉熵 (用 one-hot 标签)
            loss_ce = criterion_CE(outputs, targets)

            # 结构蒸馏: b1/b2/b3 学习 backbone 的软标签输出
            loss_kd = criterion_KD(b1_output, outputs)
            loss_kd += criterion_KD(b2_output, outputs)
            loss_kd += criterion_KD(b3_output, outputs)

        # =====================================================================
        # 总损失: loss = ce_weight × CE + kd_weight × KD
        # 默认 ce_weight=0.2, kd_weight=0.8 (蒸馏信号占主导)
        # =====================================================================
        loss = args.ce_weight * loss_ce + args.kd_weight * loss_kd

        # =====================================================================
        # [Anti-Forgetting SKD] 置信度加权的输出空间 KL 约束
        # 核心: 边际置信度 m_i = p_old(y_i) - max_{c≠y_i} p_old(c)
        #       m_i ≤ 0 → w_i = 0 (抽奖机制, 完全放弃错误历史知识)
        #       m_i > 0 → w_i = exp(α·(m_i-τ)) (指数型动态权重, 保护高置信度正确知识)
        #       KL(p_old || p_new) 强制新分布覆盖旧分布的高概率区域
        # =====================================================================
        if args.HSKD and args.af_lambda > 0.0 and epoch > 0:
            # [v3] 更新 per-sample 预测稳定性 (正确分类的 EMA)
            with torch.no_grad():
                correct_mask = (softmax_output.argmax(dim=1) == targets).float().cpu()
                stability[input_indices] = 0.9 * stability[input_indices] + 0.1 * correct_mask
                stab_batch = stability[input_indices].cuda()

                p_old_batch = all_predictions[input_indices].cuda()
                # 式(1): 边际置信度 — 识别"可靠知识"与"噪声"
                p_old_correct = p_old_batch[torch.arange(p_old_batch.size(0)), targets]
                p_old_masked = p_old_batch.clone()
                p_old_masked[torch.arange(p_old_masked.size(0)), targets] = -float('inf')
                p_old_max_wrong = p_old_masked.max(dim=1).values
                m = p_old_correct - p_old_max_wrong
                # 式(2): 动态连续权重 — "抽奖机制" (m_i≤0→0) + 指数型映射 (m_i>0)
                w = torch.where(
                    m > 0,
                    torch.exp(args.af_alpha * (m - args.af_tau)),
                    torch.zeros_like(m)
                )

            # 前向 KL: KL(p_old || p_new) — 用旧分布"检查"新分布
            log_p_new = torch.log(softmax_output + 1e-10)
            kl_per_sample = F.kl_div(log_p_new, p_old_batch, reduction='none').sum(dim=1)

            # 式(3): 加权 KL 均值
            # [v3] w_i 乘 stability_i: 只有稳定正确+高边际置信度的样本才获强保护
            # (1-α_t) 全局调度: 早期弱(自由探索), 后期强(巩固保护)
            af_weights = args.af_lambda * (1.0 - alpha_t) * stab_batch * w
            loss_af = (af_weights * kl_per_sample).mean()
            loss = loss + loss_af
            train_af_losses.update(loss_af.item(), inputs.size(0))

        # [EWC] 弹性权重巩固正则化: L_EWC = (λ/2) * Σ F_i * (θ_i - θ_i*)^2
        if args.ewc_lambda > 0 and _ewc_ref_params and _ewc_fisher:
            loss_ewc = 0.0
            for name, param in net.named_parameters():
                if name in _ewc_fisher and name in _ewc_ref_params:
                    loss_ewc += (_ewc_fisher[name] * (param - _ewc_ref_params[name]) ** 2).sum()
            loss = loss + 0.5 * args.ewc_lambda * loss_ewc

        train_losses.update(loss.item(), inputs.size(0))

        err1, err5 = accuracy(outputs.data, targets, topk=(1, 5))
        train_top1.update(err1.item(), inputs.size(0))
        train_top5.update(err5.item(), inputs.size(0))

        # compute gradient and do SGD step
        # loss.backward()
        # optimizer.step()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # =====================================================================
        # [DTSKD-核心] 更新历史预测矩阵 (用于下一轮的软目标生成)
        # =====================================================================
        if args.HSKD:
            if args.multiprocessing_distributed:
                # 分布式训练: 用 all_gather 收集所有 GPU 上的预测再更新
                for jdx in range(len(gathered_prediction)):
                    all_predictions[gathered_indices[jdx]] = gathered_prediction[jdx]
                    b1_predictions[gathered_indices[jdx]] = b1_gathered_pre[jdx]
                    b2_predictions[gathered_indices[jdx]] = b2_gathered_pre[jdx]
                    b3_predictions[gathered_indices[jdx]] = b3_gathered_pre[jdx]
            else:
                # [BugFix] 单GPU训练: 用当前 epoch 的 softmax 输出更新历史矩阵
                # 原代码缺少此分支，导致 all_predictions 始终为 epoch0 的 one-hot
                # HSKD 退化为普通 CE training — 这是作者 README 提到的已知问题
                with torch.no_grad():
                    all_predictions[input_indices] = softmax_output.cpu()
                    b1_predictions[input_indices] = b1_softmax_out.cpu()
                    b2_predictions[input_indices] = b2_softmax_out.cpu()
                    b3_predictions[input_indices] = b3_softmax_out.cpu()

        af_info = ' | af_loss: {:.3f} (stab={:.2f})'.format(train_af_losses.avg, stability.mean().item()) if args.af_lambda > 0 else ''
        progress_bar(epoch,batch_idx, len(train_loader),args, 'lr: {:.1e} | alpha_t: {:.3f} | top1_acc: {:.3f} | top5_acc: {:.3f}{}'.format(
            current_LR, alpha_t, train_top1.avg, train_top5.avg, af_info))

    # dist.barrier()
    
    logger = logging.getLogger('train')
    af_log = ' [af_loss {:.3f}]'.format(train_af_losses.avg) if args.af_lambda > 0 else ''
    logger.info('[Rank {}] [Epoch {}] [HSKD {}] [lr {:.1e}] [alpht_t {:.3f}] [train_loss {:.3f}] [train_top1_acc {:.3f}] [train_top5_acc {:.3f}]{}'.format(
        args.rank,
        epoch,
        args.HSKD,
        current_LR,
        alpha_t,
        train_losses.avg,
        train_top1.avg,
        train_top5.avg,
        af_log))
    
    # [DTSKD-PLOT] 收集训练指标用于CSV记录
    train_metrics = {
        'loss': train_losses.avg,
        'top1': train_top1.avg,
        'top5': train_top5.avg,
        'lr': current_LR,
        'alpha_t': alpha_t,
        'af_loss': train_af_losses.avg if args.af_lambda > 0 else 0.0
    }
    return all_predictions, b1_predictions, b2_predictions, b3_predictions, stability, train_metrics


def val(
        net,
        epoch,
        val_loader,
        out_list,
        target_list,
        args,
        per_sample_correct=None,
        num_val_samples=None):

    val_top1 = AverageMeter()
    val_top5 = AverageMeter()
    val_b1_top1 = AverageMeter()
    val_b2_top1 = AverageMeter()
    val_b3_top1 = AverageMeter()

    # [FORGETTING] 初始化 per-sample correctness 追踪
    if args.track_forgetting and per_sample_correct is None:
        per_sample_correct = torch.zeros(num_val_samples, dtype=torch.bool)

    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets, indices) in enumerate(val_loader):

            if args.gpu is not None:
                inputs = inputs.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)

            # model output
            outputs, b1_out, b2_out, b3_out = net(inputs)

            # [FORGETTING] 记录每个样本是否被正确分类
            if args.track_forgetting:
                _, predicted = outputs.max(1)
                correct = predicted.eq(targets).cpu()
                for i, idx in enumerate(indices):
                    per_sample_correct[idx] = correct[i]

            #Top1, Top5 Err
            err1, err5 = accuracy(outputs.data, targets, topk=(1, 5))
            val_top1.update(err1.item(), inputs.size(0))
            val_top5.update(err5.item(), inputs.size(0))

            b1_err1, _ = accuracy(b1_out.data, targets, topk=(1, 5))
            val_b1_top1.update(b1_err1.item(), inputs.size(0))

            b2_err1, _ = accuracy(b2_out.data, targets, topk=(1, 5))
            val_b2_top1.update(b2_err1.item(), inputs.size(0))

            b3_err1, _ = accuracy(b3_out.data, targets, topk=(1, 5))
            val_b3_top1.update(b3_err1.item(), inputs.size(0))

            if args.tsne:
                out_list.append(outputs.cpu())
                target_list.append(targets.cpu())

            progress_bar(epoch, batch_idx, len(val_loader), args,'val_top1_acc: {:.3f} | val_top5_acc: {:.3f}'.format(
                        val_top1.avg,
                        val_top5.avg,
                        ))

    # dist.barrier()

    if is_main_process():

        logger = logging.getLogger('val')
        logger.info('[Epoch {}] [val_top1_acc {:.3f}] [val_top5_acc {:.3f}] [val_b1_acc {:.3f}] [val_b2_acc {:.3f}] [val_b3_acc {:.3f}]'.format(
                    epoch,
                    val_top1.avg,
                    val_top5.avg,
                    val_b1_top1.avg,
                    val_b2_top1.avg,
                    val_b3_top1.avg,
                    ))

    # [DTSKD-PLOT] 收集验证指标用于CSV记录
    val_metrics = {
        'val_top1': val_top1.avg,
        'val_top5': val_top5.avg,
        'val_b1_top1': val_b1_top1.avg,
        'val_b2_top1': val_b2_top1.avg,
        'val_b3_top1': val_b3_top1.avg,
    }

    if args.track_forgetting:
        return val_top1.avg, val_metrics, per_sample_correct
    return val_top1.avg, val_metrics


def cleanup():
    dist.destroy_process_group()


import pickle
def savepickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=4)
    print('save %s!' % path)


if __name__ == '__main__':
    main()
