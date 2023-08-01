import os
import cv2
from collections import OrderedDict
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader,ConcatDataset,Subset
from tqdm import tqdm
import albumentations as A
from efficientunet import *
import sys
import argparse
# import tensorflow as tf
# from tensorflow import keras
from modules.dataset import SatelliteDataset
# from modules.dataset import SatelliteDatasetSplitTrainValid
from modules.decode import rle_encode
from modules.dice_eval import calculate_dice_scores
from modules.early_stop import *
# import apex.amp as amp
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
# import gc
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 최고 기록 갱신 : clare+상하flip, 언샤픈 마스크 필터+90도 회전
TIME = "_0"
DBDIR = "/hdd_data/datasets/aicon"
DBNAME = "train"
def get_argument():
    parser = argparse.ArgumentParser(description="aicontest")
    parser.add_argument("-t", "--time", type=str, default=TIME, help="invoke time to utilize as a prefix")
    parser.add_argument("-db", "--db_dir", default=DBDIR, type=str, help="dataset directory")
    parser.add_argument("-dname", "--db_name", default=DBNAME, type=str, help="dataset name (csv file)")
    # flags
    parser.add_argument("-val", "--validation", action="store_true", help="split images as validation set")
    parser.add_argument("-es", "--early_stop", action="store_true", help="enable early stopping when validation enabled")
    parser.add_argument("-aug_fr", "--aug_flip_rotation", action="store_true", help="augment train images (flip & rotation)")
    parser.add_argument("-aug_bc", "--aug_bright_contrast", action="store_true", help="augment train images (brightness & contrast)")
    parser.add_argument("-aug_cl", "--aug_clahe", action="store_true", help="augment train images (CLAHE)")
    parser.add_argument("-aug_bl", "--aug_blur", action="store_true", help="augment train images (Blur)")
    parser.add_argument("-aug_em", "--aug_Emboss", action="store_true", help="augment train images (Emboss)")
    # parser.add_argument("-rpt", "--report", action="store_true", help="evaluate test images")
    parser.add_argument("-na", "--no_amp", action="store_true", help="not use amp")
    # values
    parser.add_argument("-nep", "--n_epoch", type=int, default=23, help="number of epochs (maximum when early stopping enabled)")
    parser.add_argument("-b", "--batch_size", type=int, default=192, help="size of batch")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-04, help="learning_rate")
    parser.add_argument("-pf", "--posfix", type=str, default="efun7_all_aug_em_ep23", help="postfix for checkpt")
    parser.add_argument("-efun", "--efunet", type=int, default=7, help="efficientunet B number")
    parser.add_argument("-gn", "--gpu", type=int, default=0, help="gpu number")
    parser.add_argument("-ws", "--world_size", type=int, default=1, help="world_size")
    parser.add_argument("-nw", "--num_workers", type=int, default=32, help="world_size")
    parser.add_argument("-rk", "--rank", type=int, default=0, help="loss function")
    return parser.parse_args()

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    torch.cuda.set_device(args.gpu)

    print(f'Use GPU:{args.gpu} for training')
    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend='nccl',
            init_method='tcp://127.0.0.1:12233',
            world_size=args.world_size,
            rank=args.rank)
    
    csv_file = f'{args.db_dir}/{args.db_name}.csv'
    chkpt_file = f'checkpoint_{args.posfix}.pt'

    if gpu == 0:
        print ('================================')
        print (' Parameters')
        print(args)
        print (f'database csv: {csv_file}')
        print ('================================')

    # Image Preprocessing
    train_val_transform = A.Compose([
        A.RandomCrop(224, 224),
        A.Normalize(),
        ToTensorV2()
    ])

    os.chdir(args.db_dir)
    dataset = SatelliteDataset(csv_file=csv_file, transform=train_val_transform)

    if args.validation:
        num_data = len(dataset)
        indices = list(range(num_data))
        train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, test_indices)
    else:
        train_dataset = dataset

    del dataset

    #### image augmentation
    ## 1: flip & rotation
    if args.aug_flip_rotation:
        aug_trans = A.Compose([
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.Rotate(limit=90, p=1, border_mode=cv2.BORDER_REPLICATE),
            ],p=1),
            A.RandomCrop(224, 224),
            A.Normalize(),
            ToTensorV2()
        ])
        aug_dataset = SatelliteDataset(csv_file=csv_file, transform=aug_trans)
        train_dataset = ConcatDataset([train_dataset, aug_dataset])
        del aug_dataset
        del aug_trans
    ## 2: brightness & contrast
    if args.aug_bright_contrast:
        aug_trans = A.Compose([
            A.OneOf([
                A.RandomBrightness(p=1.0),
                A.RandomContrast(p=1.0),
            ],p=1),
            A.RandomCrop(224, 224),
            A.Normalize(),
            ToTensorV2()
        ])
        aug_dataset = SatelliteDataset(csv_file=csv_file, transform=aug_trans)
        train_dataset = ConcatDataset([train_dataset, aug_dataset])
        del aug_dataset
        del aug_trans
    ## 3: CLAHE
    if args.aug_clahe:
        aug_trans = A.Compose([
            A.CLAHE(p=1,clip_limit=(1,15),tile_grid_size=(8,8)),
            A.RandomCrop(224, 224),
            A.Normalize(),
            ToTensorV2()
        ])
        aug_dataset = SatelliteDataset(csv_file=csv_file, transform=aug_trans)
        train_dataset = ConcatDataset([train_dataset, aug_dataset])
        del aug_dataset
        del aug_trans
    ## 4: BLUR
    if args.aug_blur:
        aug_trans = A.Compose([
            A.Blur(p=1),
            A.RandomCrop(224, 224),
            A.Normalize(),
            ToTensorV2()
        ])
        aug_dataset = SatelliteDataset(csv_file=csv_file, transform=aug_trans)
        train_dataset = ConcatDataset([train_dataset, aug_dataset])
        del aug_dataset
        del aug_trans
    ## 5: Emboss
    if args.aug_Emboss:
        aug_trans = A.Compose([
            A.Emboss(p=1),
            A.RandomCrop(224, 224),
            A.Normalize(),
            ToTensorV2()
        ])
        aug_dataset = SatelliteDataset(csv_file=csv_file, transform=aug_trans)
        train_dataset = ConcatDataset([train_dataset, aug_dataset])
        del aug_dataset
        del aug_trans


    if gpu == 0:
        print(f"Training Data Size: {len(train_dataset)}")
        if args.validation:
            print(f"Validation Data Size: {len(val_dataset)}")

    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    if args.validation:
        val_sampler = DistributedSampler(dataset=val_dataset, shuffle=False)


    train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=int(args.batch_size / args.world_size),
            shuffle=False,
            num_workers=int(args.num_workers / args.world_size),
            sampler=train_sampler,
            pin_memory=True)

    if args.validation:
        val_dataloader = DataLoader(
                dataset=val_dataset,
                batch_size=int(args.batch_size / args.world_size),
                shuffle=False,
                num_workers=int(args.num_workers / args.world_size),
                sampler=val_sampler,
                pin_memory=True)

    if gpu == 0:
        print(f"Training Data Size (batched): {len(train_dataloader)}")
        if args.validation:
            print(f"Validation Data Size (batched): {len(val_dataloader)}")


    # load model
    if args.efunet == 0:
        model = get_efficientunet_b0(out_channels=1, concat_input=True, pretrained=True)
    elif args.efunet == 1:
        model = get_efficientunet_b1(out_channels=1, concat_input=True, pretrained=True)
    elif args.efunet == 2:
        model = get_efficientunet_b2(out_channels=1, concat_input=True, pretrained=True)
    elif args.efunet == 3:
        model = get_efficientunet_b3(out_channels=1, concat_input=True, pretrained=True)
    elif args.efunet == 4:
        model = get_efficientunet_b4(out_channels=1, concat_input=True, pretrained=True)
    elif args.efunet == 5:
        model = get_efficientunet_b5(out_channels=1, concat_input=True, pretrained=True)
    elif args.efunet == 6:
        model = get_efficientunet_b6(out_channels=1, concat_input=True, pretrained=True)
    elif args.efunet == 7:
        model = get_efficientunet_b7(out_channels=1, concat_input=True, pretrained=True)
    if gpu == 0:
        print (f'loading efficientunet b{args.efunet}...')
    model = model.cuda(args.gpu)
    model = DDP(module=model, device_ids=[args.gpu])

    criterion = torch.nn.BCEWithLogitsLoss()
    if gpu == 0:
        print ('Using BCEWithLogitsLoss...')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if gpu == 0:
        print ('train epochs start...')

    if args.early_stop:
        train_losses = []
        valid_losses = []
        avg_train_losses = []
        avg_valid_losses = []
        early_stopping = EarlyStopping(patience = 5, verbose=True, path=chkpt_file)
        # Creates a GradScaler once at the beginning of training.
        scaler = GradScaler()
        for epoch in range(args.n_epoch):
            model.train()
            train_sampler.set_epoch(epoch)
            for images, masks in tqdm(train_dataloader):
                for img, mk in zip(images, masks):
                    img = img.float().to(args.gpu)
                    mk = mk.float().to(args.gpu)
                    optimizer.zero_grad()
                    if args.no_amp:
                        outputs = model(img)
                        loss = criterion(outputs, mk.unsqueeze(1))
                        loss.backward()
                        optimizer.step()
                    else:
                        with autocast():
                            outputs = model(img)
                            loss = criterion(outputs, mk.unsqueeze(1))
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
    
                    train_losses.append(loss.item())               
     
            # torch.cuda.empty_cache()
            # gc.collect()
    
            # valid epochs({epoch}) start...
            with torch.no_grad():
                model.eval()
                for img, mk in tqdm(val_dataloader):
                    for img, mk in zip(images, masks):
                        img = img.float().to(args.gpu)
                        mk = mk.float().to(args.gpu)
                        outputs = model(img)
                        loss = criterion(outputs, mk.unsqueeze(1))
                        valid_losses.append(loss.item())
                
                # print 학습/검증 statistics
                # epoch당 평균 loss 계산
                train_loss = np.average(train_losses)
                valid_loss = np.average(valid_losses)
                avg_train_losses.append(train_loss)
                avg_valid_losses.append(valid_loss)
                
                epoch_len = len(str(args.n_epoch))
    
                if gpu == 0:
                    print_msg = (f'[{epoch:>{epoch_len}}/{args.n_epoch:>{epoch_len}}] ' +
                                 f'train_loss: {train_loss:.5f} ' +
                                 f'valid_loss: {valid_loss:.5f}')
    
                    print(print_msg)
    
                # clear lists to track next epoch
                train_losses = []
                valid_losses = []
    
                # early_stopping는 validation loss가 감소하였는지 확인이 필요하며,
                # 만약 감소하였을경우 현제 모델을 checkpoint로 만든다.
                if gpu == 0:
                    early_stopping(valid_loss, model)
    
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break
    
       # best model이 저장되어있는 last checkpoint를 로드한다.
        model.load_state_dict(torch.load(chkpt_file))
            
        del train_dataloader
    
    else:
        # Creates a GradScaler once at the beginning of training.
        scaler = GradScaler()
        for epoch in range(args.n_epoch):
            model.train()
            train_sampler.set_epoch(epoch)
            epoch_loss = 0
            k = 0
            for images, masks in tqdm(train_dataloader):
                for img, mk in zip(images, masks):
                    img = img.float().to(args.gpu)
                    mk = mk.float().to(args.gpu)
        
                    optimizer.zero_grad()
                    if args.no_amp:
                        outputs = model(img)
                        loss = criterion(outputs, mk.unsqueeze(1))
                        loss.backward()
                        optimizer.step()
                    else:
                        with autocast():
                            outputs = model(img)
                            loss = criterion(outputs, mk.unsqueeze(1))
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
    
                    ltmp = loss.item()
                    epoch_loss += ltmp
        
                    k += 1
            
            if gpu == 0:
                print(f'GPU({gpu}):Epoch {epoch+1}, Loss: {epoch_loss/k}, image_num: {k}')
                print(f'GPU({gpu}):Saving model... ({chkpt_file}-GPU{gpu}-epoch{epoch+1})')
                torch.save(model.state_dict(), f'{chkpt_file}-GPU{gpu}-epoch{epoch+1}')
        
        if gpu == 0:
            print(f'GPU({gpu}):Saving model... ({chkpt_file}-GPU{gpu})')
            torch.save(model.state_dict(), f'{chkpt_file}-GPU{gpu}')
            # print(f'Saving model... ({chkpt_file})')
            # torch.save(model.state_dict(), chkpt_file)
        del train_dataloader
    
    
    if args.validation:
        if gpu == 0:
            print("Dice score evaluation")
            with torch.no_grad():
                model.eval()
                score_sum = 0
                image_num = 0
                output_rle_array = []
                gt_rle_array = []
                for images, masks in tqdm(val_dataloader):
                    for img, mk in zip(images, masks):
                        images = img.float().to(args.gpu)
                        gts = mk.float().to(args.gpu)
            
                        outputs = model(images)
                        outputs = torch.sigmoid(outputs).cpu().numpy()
                        outputs = np.squeeze(outputs, axis=1)
                        outputs = (outputs > 0.35).astype(np.uint8) # Threshold = 0.35
            
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                        
                        for i in range(len(images)):
                            gt_rle_array.append(rle_encode(gts[i].cpu().numpy()))
                            # 모폴로지 연산을 수행하여 마스크 개선
                            output = cv2.morphologyEx(outputs[i], cv2.MORPH_OPEN, kernel)
                            output_rle = rle_encode(output)
                            if output_rle == '': # 예측된 건물 픽셀이 아예 없는 경우 -1
                                output_rle_array.append(-1)
                            else:
                                output_rle_array.append(output_rle)
            
                del val_dataloader
                dice_score = calculate_dice_scores(gt_rle_array, output_rle_array)
                print(f'dice score = {np.mean(dice_score)}')
    
    
    # if args.report:
    #     test_transform = A.Compose([
    #         # A.RandomCrop(224, 224),
    #         # A.Normalize(),
    #         ToTensorV2()
    #     ])
    #     csv_file=f'{args.db_dir}/test.csv'
    #     test_dataset = SatelliteDataset(csv_file=csv_file, transform=test_transform, infer=True)
    #     test_dataloader = DataLoader(test_dataset, batch_size=batch_size_cfg, shuffle=False, num_workers=num_workers_cfg, pin_memory=True) 
    # 
    #     with torch.no_grad():
    #         model.eval()
    #         result = []
    #         for images in tqdm(test_dataloader):
    #             images = images.float().to(args.gpu)
    #             
    #             outputs = model(images)
    #             masks = torch.sigmoid(outputs).cpu().numpy()
    #             masks = np.squeeze(masks, axis=1)
    #             masks = (masks > 0.35).astype(np.uint8) # Threshold = 0.35
    #     
    #             kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #     
    #             for i in range(len(images)):
    #                 # 모폴로지 연산을 수행하여 마스크 개선
    #                 mask = masks[i]
    #                 mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #                 mask_rle = rle_encode(mask)
    #                 if mask_rle == '': # 예측된 건물 픽셀이 아예 없는 경우 -1
    #                     result.append(-1)
    #                 else:
    #                     result.append(mask_rle)
    #     
    #     submit = pd.read_csv(f'{args.db_dir}/sample_submission.csv')
    #     submit['mask_rle'] = result
    #     submit.to_csv('submit_{}.csv'.format(args.time), index=False)
    


if __name__ == '__main__':

    args = get_argument()
    ngpus_per_node = torch.cuda.device_count()
    print (f'ngpu_per_node = {ngpus_per_node}')
    args.world_size = ngpus_per_node * args.world_size
    print (f'world_size = {args.world_size}')
    mp.spawn(
            main_worker,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, args),
            join=True)

