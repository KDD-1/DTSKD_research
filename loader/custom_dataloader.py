#--------------
#torch
#--------------
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

#--------------
# torchvision
#--------------
from torchvision import transforms

#--------------
# etc
#--------------
import os
import numpy as np

#--------------
# utils
#--------------
from loader import custom_datasets
from utils import custom_transform
from utils.color import Colorer

C = Colorer.instance()


def apply_symmetric_label_noise(targets, num_classes, noise_rate, seed=42):
    """
    对称标签噪声: 以 noise_rate 概率将每个样本的标签随机翻转为其他类别。

    Args:
        targets: list/array of original labels
        num_classes: total number of classes
        noise_rate: fraction of labels to corrupt (0.0 ~ 1.0)
        seed: random seed for reproducibility

    Returns:
        noisy_targets: corrupted label array (numpy)
        flip_mask: boolean array, True for flipped samples
    """
    if noise_rate <= 0:
        return np.array(targets), np.zeros(len(targets), dtype=bool)

    rng = np.random.RandomState(seed)
    targets = np.array(targets)
    n = len(targets)

    # 随机选择 noise_rate 比例的样本
    flip_mask = rng.rand(n) < noise_rate
    n_flip = flip_mask.sum()

    # 对选中样本随机翻转到其他类别
    noisy_targets = targets.copy()
    if n_flip > 0:
        flip_indices = np.where(flip_mask)[0]
        for idx in flip_indices:
            # 随机选一个不等于原标签的类别
            other_classes = [c for c in range(num_classes) if c != targets[idx]]
            noisy_targets[idx] = rng.choice(other_classes)

    return noisy_targets, flip_mask

def dataloader(args):
    if args.data_type == 'cifar10':
        
        mean= [x / 255.0 for x in [125.3, 123.0, 113.9]]
        stdv = [x / 255.0 for x in [63.0, 62.1, 66.7]]
        
        print(C.green("[!] [Rank {}] Preparing {} data..".format(args.rank, args.data_type)))
        transform_train = transforms.Compose([
                                              transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=mean, std=stdv),
                                             ])

        transform_val = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=stdv),
                                           ])
        
        trainset = custom_datasets.Custom_CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
        validset = custom_datasets.Custom_CIFAR10(root=args.data_path, train=False, download=True, transform=transform_val)

        # [Label Noise] 对称标签噪声 — 仅污染训练集
        if hasattr(args, 'noise_rate') and args.noise_rate > 0:
            noisy_targets, flip_mask = apply_symmetric_label_noise(
                trainset.targets, num_classes=10, noise_rate=args.noise_rate,
                seed=args.random_seed)
            n_flipped = flip_mask.sum()
            trainset.targets = noisy_targets.tolist()
            print(C.yellow("[!] [Rank {}] Applied {:.0f}% symmetric label noise: "
                           "{}/{} training labels flipped".format(
                args.rank, args.noise_rate * 100, n_flipped, len(trainset.targets))))

        if args.multiprocessing_distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
            print(C.green("[!] [Rank {}] Distributed Sampler Data Loading Done".format(args.rank)))
        else:
            train_sampler = None
            print(C.green("[!] [Rank {}] Data Loading Done".format(args.rank)))
        
        train_loader = torch.utils.data.DataLoader(trainset, pin_memory=True,
                                                   batch_size=args.batch_size,
                                                   sampler=train_sampler,
                                                   shuffle=(train_sampler is None),
                                                   num_workers=args.workers,
                                                   persistent_workers=(args.workers > 0),
                                                   prefetch_factor=2 if args.workers > 0 else None)

        valid_loader = torch.utils.data.DataLoader(validset, pin_memory=True,
                                                   batch_size=args.batch_size,
                                                   sampler=None,
                                                   shuffle=False,
                                                   num_workers=args.workers,
                                                   persistent_workers=(args.workers > 0),
                                                   prefetch_factor=2 if args.workers > 0 else None)

    elif args.data_type == 'cifar100':

        mean = [0.4914, 0.4822, 0.4465]                                 
        stdv = [0.2023, 0.1994, 0.2010]
        
        print(C.green("[!] [Rank {}] Preparing {} data..".format(args.rank, args.data_type)))
        
        transform_train = transforms.Compose([                         
                                              transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=mean, std=stdv),
                                             ])

        transform_val = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=stdv),
                                           ])
        
        trainset = custom_datasets.Custom_CIFAR100(root=args.data_path, train=True, download=True, transform=transform_train)
        validset = custom_datasets.Custom_CIFAR100(root=args.data_path, train=False, download=True, transform=transform_val)

        # [Label Noise] 对称标签噪声 — 仅污染训练集
        if hasattr(args, 'noise_rate') and args.noise_rate > 0:
            noisy_targets, flip_mask = apply_symmetric_label_noise(
                trainset.targets, num_classes=100, noise_rate=args.noise_rate,
                seed=args.random_seed)
            n_flipped = flip_mask.sum()
            trainset.targets = noisy_targets.tolist()
            print(C.yellow("[!] [Rank {}] Applied {:.0f}% symmetric label noise: "
                           "{}/{} training labels flipped".format(
                args.rank, args.noise_rate * 100, n_flipped, len(trainset.targets))))

        if args.multiprocessing_distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
            print(C.green("[!] [Rank {}] Distributed Sampler Data Loading Done".format(args.rank)))
        else:
            train_sampler = None
            print(C.green("[!] [Rank {}] Data Loading Done".format(args.rank)))
        

        train_loader = torch.utils.data.DataLoader(trainset, pin_memory=True,
                                                   batch_size=args.batch_size,
                                                   sampler=train_sampler,
                                                   shuffle=(train_sampler is None),
                                                   num_workers=args.workers,
                                                   persistent_workers=(args.workers > 0),
                                                   prefetch_factor=2 if args.workers > 0 else None)

        valid_loader = torch.utils.data.DataLoader(validset, pin_memory=True,
                                                   batch_size=args.batch_size,
                                                   sampler=None, shuffle=False,
                                                   num_workers=args.workers,
                                                   persistent_workers=(args.workers > 0),
                                                   prefetch_factor=2 if args.workers > 0 else None)
    
    elif args.data_type == 'imagenet':
        mean=[0.485, 0.456, 0.406]
        stdv=[0.229, 0.224, 0.225]
        print(C.green("[!] [Rank {}] Preparing {} data..".format(args.rank, args.data_type)))
        jittering = custom_transform.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        lighting = custom_transform.Lighting(alphastd=0.1,
                                             eigval=[0.2175, 0.0188, 0.0045],
                                             eigvec=[[-0.5675, 0.7192, 0.4009],
                                                     [-0.5808, -0.0045, -0.8140],
                                                     [-0.5836, -0.6948, 0.4203]])
        
        transform_train = transforms.Compose([
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              jittering,
                                              lighting,
                                              transforms.Normalize(mean=mean, std=stdv),
                                             ])
        
        transform_val = transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=stdv),
                                          ])
        
        trainset = custom_datasets.Custom_ImageFolder(os.path.join(args.data_path,'train'), transform=transform_train)
        validset = custom_datasets.Custom_ImageFolder(os.path.join(args.data_path,'val'), transform=transform_val)
        
        if args.multiprocessing_distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
            print(C.green("[!] [Rank {}] Distributed Sampler Data Loading Done".format(args.rank)))
        else:
            train_sampler = None
            print(C.green("[!] [Rank {}] Data Loading Done".format(args.rank)))
        
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   pin_memory=True,
                                                   batch_size=args.batch_size,
                                                   sampler=train_sampler,
                                                   shuffle=(train_sampler is None),
                                                   num_workers=args.workers,
                                                   persistent_workers=(args.workers > 0),
                                                   prefetch_factor=2 if args.workers > 0 else None)

        valid_loader = torch.utils.data.DataLoader(validset,
                                                   pin_memory=True,
                                                   batch_size=args.batch_size,
                                                   sampler=None,
                                                   shuffle=False,
                                                   num_workers=args.workers,
                                                   persistent_workers=(args.workers > 0),
                                                   prefetch_factor=2 if args.workers > 0 else None)
    
    else:
        raise Exception("[!] There is no option for Datatype")    
        
    return train_loader,valid_loader,train_sampler

