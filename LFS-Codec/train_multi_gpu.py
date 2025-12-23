#coding=utf-8
import logging
import os
import warnings
from collections import defaultdict
import random
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import torchaudio

import customAudioDataset as data
from customAudioDataset import collate_fn
from losses import disc_loss, total_loss, d_axis_distill_loss, t_axis_distill_loss 
#from model import EncodecModel
from msstftd import MultiScaleSTFTDiscriminator
from scheduler import WarmupCosineLrScheduler
from utils_en import (count_parameters, save_master_checkpoint, set_seed,
                   start_dist_train)
from balancer import Balancer
from cal_metrics import calculate_stoi, calculate_pesq, cal_stoi_gpu, cal_pesq_gpu
import wave
# from device_config import device    # no distributed training
from models import loaders
import time
from transformers import AutoModel,  Wav2Vec2FeatureExtractor
from transformers import WavLMModel, Wav2Vec2Processor, AutoProcessor
import torch.nn as nn
import math
from modules.resample import ConvDownsample1d
import itertools
import distrib

warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def loss_norm(loss):
    try:
        norm = 10.0 ** math.floor(math.log10(1 / abs(loss.detach().item())))
    except:
        norm = 0.0
    return norm

def cal_metr(in_wav, out_wav, config):
    total_stoi = 0.0
    total_pesq_wb = 0.0
    B = in_wav.size(0)
    in_wav = in_wav.squeeze(1)
    out_wav = out_wav.squeeze(1)
    for i in range(B):
        sing_inwav_16 = torchaudio.functional.resample(in_wav[i,:], config.model.sample_rate, 16000)
        sing_outwav_16 = torchaudio.functional.resample(out_wav[i,:], config.model.sample_rate, 16000)
        sing_inwav = in_wav[i,:]
        sing_outwav = out_wav[i,:]
        total_stoi += cal_stoi_gpu(sing_inwav, sing_outwav, config.model.sample_rate)
        wb = cal_pesq_gpu(sing_inwav_16, sing_outwav_16, 16000)
        total_pesq_wb += wb
    return total_stoi / B , total_pesq_wb /B

def calculate_loss_distill(feature,semantic_feature,distill_type, config):
    if distill_type == 't_axis':
        return t_axis_distill_loss(feature,semantic_feature,config.distill.lambda_sim)
    else:
        return d_axis_distill_loss(feature,semantic_feature)
# Define train one step function
def train_one_step(epoch,optimizer,optimizer_disc, distill_type, model, Conv, disc_model, trainloader,config,scheduler,disc_scheduler,scaler=None,scaler_disc=None,writer=None,balancer=None,total_step=None,feature_extractor=None,feat_model=None):
    """train one step function

    Args:
        epoch (int): current epoch
        optimizer (_type_) : generator optimizer
        optimizer_disc (_type_): discriminator optimizer
        distill_type: To Distill
        model (_type_): generator model
        disc_model (_type_): discriminator model
        trainloader (_type_): train dataloader
        config (_type_): hydra config file
        scheduler (_type_): adjust generate model learning rate
        disc_scheduler (_type_): adjust discriminator model learning rate
        warmup_scheduler (_type_): warmup learning rate
    """
    model.train()
    Conv.train()
    disc_model.train()
    data_length=len(trainloader)
    # Initialize variables to accumulate losses  
    accumulated_loss_g = 0.0
    accumulated_losses_g = defaultdict(float)
    accumulated_loss_w = 0.0
    accumulated_loss_disc = 0.0
    accumulated_loss_distill = 0.0
    accumulated_loss_t = 0.0
    accumulated_loss_f = 0.0
    accumulated_loss_global = 0.0
    accumulated_loss_W = 0.0

    stoi = 0.0
    # accumulated_pesq_nb =0.0
    pesq_wb = 0.0
    ts_time = 0.0
    te_time = time.time()
    data_time = 0.0
    train_time = 0.0
    # input_wav_16 = torch.zeros(config.datasets.batch_size, int(config.datasets.segment_size/1.5))
    input_wav_16 = torch.zeros((config.datasets.batch_size, int(config.datasets.segment_size/1.5)))
    for idx,input_list in enumerate(trainloader):

        if idx == data_length - 1:  
            continue
        # del input_wav
        input_wav = input_list[0]
        input_global = input_list[1]
        total_step = total_step + 1
        # warmup learning rate, warmup_epoch is defined in config file,default is 5
        ts_time = time.time()
        data_time = (ts_time - te_time)

        input_wav = input_wav.cuda()
        input_wav = input_wav.squeeze(1)
        input_global = input_global.cuda()
        target_layer = 'notavg'

        with torch.no_grad():
            for index in range(config.datasets.batch_size):
                input_wav_16[index] = torchaudio.functional.resample(input_wav[index], 24000, 16000)
            input_values = feature_extractor(input_wav_16, sampling_rate=16000, padding=True, return_tensors="pt").input_values
            input_values = input_values.squeeze(0)
            ouput = feat_model(input_values.cuda(), output_hidden_states=True)
            if target_layer == 'avg':
                rep = torch.mean(torch.stack(ouput.hidden_states), axis=0)
            else:
                rep = ouput.hidden_states[19] 
        input_feat1 = rep.permute(0, 2, 1)
        input_feat = Conv(input_feat1)
            
        input_wav = input_wav.unsqueeze(1)
        input_wav = input_wav.contiguous()#[B, 1, T]: eg. [2, 1, 203760]
        input_global = input_global.contiguous()
        optimizer.zero_grad()
        distill_type = config.distill.distill_type
        if distill_type == 't_axis':
            from functools import partial
            lambda_sim = config.distill.lambda_sim
            distill_loss = partial(t_axis_distill_loss, lambda_sim=lambda_sim)
        else:
            distill_loss = d_axis_distill_loss
        with autocast(enabled=config.common.amp):
            output, loss_w, semantic_feature, global_loss, W_Loss = model(input_wav, input_global) #output: [B, 1, T]: eg. [2, 1, 203760] | loss_w: [1] 
            logits_real, fmap_real = disc_model(input_wav)
            logits_fake, fmap_fake = disc_model(output)
            loss_distill, sim = calculate_loss_distill(input_feat, semantic_feature, distill_type, config)    
            losses_g = total_loss(
                fmap_real, 
                logits_fake, 
                fmap_fake, 
                input_wav, 
                output, 
                sample_rate=config.model.sample_rate,
            ) 
        loss_w_lambda = config.model.commitment_loss_lambda
        loss_distill_lambda = config.model.distill_loss_lambda
        # loss_distill = distill_loss(input_feat, semantic_feature)

        if config.common.amp: 

            losses_g['l_g'] = loss_norm(losses_g['l_g'])*losses_g['l_g']
            losses_g['l_feat'] = loss_norm(losses_g['l_feat'])*losses_g['l_feat']
            losses_g['l_t'] = loss_norm(losses_g['l_t'])*losses_g['l_t']
            losses_g['l_f'] = loss_norm(losses_g['l_f'])*losses_g['l_f']
            loss_g = losses_g['l_g'] + losses_g['l_feat'] + losses_g['l_t'] + losses_g['l_f'] 
            if loss_w.detach().item() >= 0.01:
                loss_w = loss_norm(loss_w)*loss_w
            loss_g_all = loss_w + loss_distill + loss_g

            scaler.scale(loss_g_all).backward()  
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  
            scaler.step(optimizer)  
            scaler.update()   
            # BUG: doesn't this get done later anyway?
            scheduler.step()  
        else:
            # They say they use multiple backwards calls, and lambda_w is 1...
            # https://github.com/facebookresearch/encodec/issues/20
            if balancer is not None:
                balancer.backward(losses_g, output, retain_graph=True)
                # naive loss summation for metrics below
                loss_g = sum([l * balancer.weights[k] for k, l in losses_g.items()])
            else:
                losses_t = loss_norm(losses_g['l_t'])*losses_g['l_t']
                losses_f = loss_norm(losses_g['l_f'])*losses_g['l_f']
                loss_g = (3*losses_g['l_g']) + (3*losses_g['l_feat']) + losses_t*8 + losses_f*8
            
            # if loss_w.detach().item() >= 0.005:
            #     loss_w = loss_norm(loss_w)*loss_w
            loss_g_all = loss_g + 10*loss_w + 12*loss_norm(loss_distill)*loss_distill + global_loss + W_Loss

            loss_g_all.backward()
            optimizer.step()

        # Accumulate losses  
        accumulated_loss_g += loss_g.item()
        for k, l in losses_g.items():
            accumulated_losses_g[k] += l.item()
        accumulated_loss_w += loss_w.item()
        accumulated_loss_distill += loss_distill.item()
        accumulated_loss_t += losses_t.item()*8
        accumulated_loss_f += losses_f.item()*8
        accumulated_loss_global += global_loss.item()
        accumulated_loss_W += W_Loss.item()
        
        if idx % config.common.log_interval == 0:
            stoi, pesq_wb = cal_metr(input_wav, output, config)
        # only update discriminator with probability from paper (configure)
        optimizer_disc.zero_grad()
        train_discriminator = torch.BoolTensor([config.model.train_discriminator 
                               and (epoch >= config.lr_scheduler.warmup_epoch or total_step >= config.lr_scheduler.discri_step)            #调整
                               and random.random() < float(config.model.train_discriminator_rate)]).cuda()
        if dist.is_initialized():
            dist.broadcast(train_discriminator, 0) 

        if train_discriminator:
            with autocast(enabled=config.common.amp):
                logits_real, _ = disc_model(input_wav)
                logits_fake, _ = disc_model(output.detach()) # detach to avoid backpropagation to model
                loss_disc = disc_loss(logits_real, logits_fake) # compute discriminator loss
            if config.common.amp: 
                # loss_disc = loss_norm(loss_disc)*loss_disc
                scaler_disc.scale(loss_disc).backward()
                # torch.nn.utils.clip_grad_norm_(disc_model.parameters(), 1.0)    
                scaler_disc.step(optimizer_disc)  
                scaler_disc.update()  
            else:
                # loss_disc = loss_norm(loss_disc)*loss_disc
                loss_disc.backward() 
                optimizer_disc.step()

            # Accumulate discriminator loss  
            accumulated_loss_disc += loss_disc.item()
        scheduler.step()
        disc_scheduler.step()
        te_time = time.time()
        train_time = (te_time - ts_time)
        if total_step % config.common.save_interval == 0:
            model_to_save = model.module if config.distributed.data_parallel else model
            Conv_to_save = Conv.module if config.distributed.data_parallel else Conv
            disc_model_to_save = disc_model.module if config.distributed.data_parallel else disc_model 
            if not config.distributed.data_parallel or dist.get_rank() == 0:  
                save_master_checkpoint(epoch, total_step, model_to_save, Conv_to_save, optimizer, scheduler, f'{config.checkpoint.save_location}epoch{epoch}_step{total_step}_lr{config.optimization.lr}.pt')  
                save_master_checkpoint(epoch, total_step, disc_model_to_save, Conv_to_save, optimizer_disc, disc_scheduler, f'{config.checkpoint.save_location}epoch{epoch}_step{total_step}_disc_lr{config.optimization.lr}.pt') 
                logger.info(f"save checkpoint at {total_step} step")


        if (not config.distributed.data_parallel or dist.get_rank() == 0) and (idx % config.common.log_interval == 0 or idx == data_length - 2): # idx == data_length - 1
            log_msg = (  
                f"Epoch {epoch} {idx+1}/{data_length}\tAvg loss_G: {accumulated_loss_g / (idx + 1):.4f}\tAvg loss_W: {accumulated_loss_w / (idx + 1):.4f}\tAvg loss_Distill: {accumulated_loss_distill / (idx + 1):.4f}\tAvg loss_global: {accumulated_loss_global / (idx + 1):.4f}\tlr_G: {optimizer.param_groups[0]['lr']:.6e}\tlr_D: {optimizer_disc.param_groups[0]['lr']:.6e}\n"  
            ) 
            writer.add_scalar('Train/Loss_G', accumulated_loss_g / (idx + 1), total_step)  
            for k, l in accumulated_losses_g.items():
                writer.add_scalar(f'Train/{k}', l / (idx + 1), total_step)
                log_msg += f"{k}:{l / (idx + 1) :.4f}\t"
            writer.add_scalar('Train/Loss_W', accumulated_loss_w / (idx + 1), total_step) 
            writer.add_scalar('Train/Loss_Distill', accumulated_loss_distill / (idx + 1), total_step)
            writer.add_scalar('Train/Loss_global', accumulated_loss_global / (idx + 1), total_step) 
            writer.add_scalar('Train/loss_t_norm', accumulated_loss_t / (idx + 1), total_step)
            log_msg += f"Avg loss_t_norm: {accumulated_loss_t / (idx + 1):.4f}\t"
            writer.add_scalar('Train/loss_f_norm', accumulated_loss_f / (idx + 1), total_step)
            log_msg += f"Avg loss_f_norm: {accumulated_loss_f / (idx + 1):.4f}\n"

            writer.add_scalar('Train/loss_w_orth', accumulated_loss_W / (idx + 1), total_step)
            log_msg += f"Avg loss_w_orth: {accumulated_loss_W / (idx + 1):.4f}\n"

            if config.model.train_discriminator and (epoch >= config.lr_scheduler.warmup_epoch or idx >= config.lr_scheduler.discri_step):
                log_msg += f"loss_disc: {accumulated_loss_disc / (idx + 1) :.4f}\n"  
                writer.add_scalar('Train/Loss_Disc', accumulated_loss_disc / (idx + 1), total_step) 
            log_msg += f"stoi: {stoi :.4f}\tpesq_wb: {pesq_wb :.4f}\tSim: {sim :.4f}\n"
            log_msg += f"data_time: {data_time :.4f}\ttrain_time: {train_time :.4f}\n"
            writer.add_scalar('Train/stoi', stoi, total_step) 
            # writer.add_scalar('Train/pesq_nb', accumulated_pesq_nb / (idx + 1), total_step)
            writer.add_scalar('Train/pesq_wb', pesq_wb, total_step)
            writer.add_scalar('Train/Sim', sim, total_step)
            logger.info(log_msg) 
        # if (config.distributed.data_parallel and dist.get_rank() != 0) and (idx % config.common.log_interval == 0 or idx == data_length - 1):
        #     log_msg = f"{dist.get_rank()}_avg_data_time: {total_data_time / (idx+1) :.4f}\t{dist.get_rank()}_avg_train_time: {total_train_time / (idx + 1) :.4f}\n"
        #     logger.info(log_msg)
    return total_step

# @torch.no_grad()  
def test(epoch, distill_type, model, disc_model, testloader, config, writer, feature_extractor=None, feat_model=None):
    with torch.no_grad():  
        model.eval()
        accumulated_stoi = 0.0
        # accumulated_pesq_nb = 0.0
        accumulated_pesq_wb = 0.0
        for idx, input_tulple in enumerate(testloader):
            input_wav = input_tulple[0].unsqueeze(1)
            #input_feat = input_tulple[1].squeeze(1)
            input_wav = input_wav.cuda()
            #input_feat = input_feat.cuda()
            # import pdb;pdb.set_trace()
            output = model(input_wav)
            logits_real, fmap_real = disc_model(input_wav)
            logits_fake, fmap_fake = disc_model(output)
            loss_disc = disc_loss(logits_real, logits_fake) # compute discriminator loss
            losses_g = total_loss(fmap_real, logits_fake, fmap_fake, input_wav, output) 
            # breakpoint()

            stoi, pesq_wb = cal_metr(input_wav, output, config)
            # breakpoint()
            accumulated_stoi += stoi
            # accumulated_pesq_nb += pesq_nb
            accumulated_pesq_wb += pesq_wb
        avg_stoi = accumulated_stoi / len(testloader)
        avg_pesq = accumulated_pesq_wb / len(testloader)
        if not config.distributed.data_parallel or dist.get_rank()==0: 
            log_msg = (f'| TEST | epoch: {epoch} | loss_g: {sum([l.item() for l in losses_g.values()])} | loss_disc: {loss_disc.item():.4f} | avg_stoi: {avg_stoi:.4f} | avg_pesq: {avg_pesq:.4f}') 
            for k, l in losses_g.items():
                writer.add_scalar(f'Test/{k}', l.item(), epoch)  
            writer.add_scalar('Test/Loss_Disc', loss_disc.item(), epoch)
            writer.add_scalar('Test/stoi', avg_stoi, epoch)
            writer.add_scalar('Test/pesq_wb', avg_pesq, epoch)
            logger.info(log_msg)

            # save a sample reconstruction (not cropped)
            input_wav, _ = testloader.dataset.get()
            input_wav = input_wav.cuda()
            output = model(input_wav.unsqueeze(0)).squeeze(0)    
            # summarywriter can't log stereo files 😅 so just save examples
            sp = Path(config.checkpoint.save_folder)
            torchaudio.save(sp/f'GT.wav', input_wav.cpu(), config.model.sample_rate)
            torchaudio.save(sp/f'Reconstruction.wav', output.cpu(), config.model.sample_rate)

def train(local_rank,world_size,config,tmp_file=None):
    """train main function."""
    logger.handlers.clear()
    # set logger
    file_handler = logging.FileHandler(f"{config.checkpoint.save_folder}/train_encodec_bs{config.datasets.batch_size}_lr{config.optimization.lr}.log")
    formatter = logging.Formatter('%(asctime)s: %(levelname)s: [%(filename)s: %(lineno)d]: %(message)s')
    file_handler.setFormatter(formatter)

    # print to screen
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    # add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # set seed
    if config.common.seed is not None:
        set_seed(config.common.seed)

    # set train dataset
    trainset = data.CustomAudioDataset(config=config)
    testset = data.CustomAudioDataset(config=config,mode='test')
    # set encodec model and discriminator model

    model = loaders.get_mimi("", device='cuda')

    disc_model = MultiScaleSTFTDiscriminator(
        in_channels=config.model.channels,
        out_channels=config.model.channels,
        filters=config.model.filters,
        hop_lengths=config.model.disc_hop_lengths,
        win_lengths=config.model.disc_win_lengths,
        n_ffts=config.model.disc_n_ffts,
    )
    Conv = ConvDownsample1d(stride=4,dimension=1024,causal=True,learnt=True)

    # log model, disc model parameters and train mode
    logger.info(model)
    logger.info(disc_model)
    logger.info(config)
    logger.info(f"Mimi Model Parameters: {count_parameters(model)} | Disc Model Parameters: {count_parameters(disc_model)}")
    logger.info(f"model train mode :{model.training} | quantizer train mode :{model.quantizer.training} ")

    # resume training
    resume_epoch = 0
    resume_step = 0
    if config.checkpoint.resume:
        # check the checkpoint_path
        assert config.checkpoint.checkpoint_path != '', "resume path is empty"
        assert config.checkpoint.disc_checkpoint_path != '', "disc resume path is empty"

        model_checkpoint = torch.load(config.checkpoint.checkpoint_path, map_location='cpu')
        disc_model_checkpoint = torch.load(config.checkpoint.disc_checkpoint_path, map_location='cpu')
        model.load_state_dict(model_checkpoint['model_state_dict'])
        Conv.load_state_dict(model_checkpoint['Conv_state_dict'])
        disc_model.load_state_dict(disc_model_checkpoint['model_state_dict'])
        resume_epoch = model_checkpoint['epoch']
        resume_step = model_checkpoint['step']
        if resume_epoch >= config.common.max_epoch:
            raise ValueError(f"resume epoch {resume_epoch} is larger than total epochs {config.common.epochs}")
        logger.info(f"load chenckpoint of model and disc_model, resume from {resume_epoch}")

    train_sampler = None
    test_sampler = None
    if config.distributed.data_parallel:
        # distributed init
        if config.distributed.init_method == "tmp":
            torch.distributed.init_process_group(
                backend='nccl',
                init_method="file://{}".format(tmp_file),
                rank=local_rank,
                world_size=world_size)
        elif config.distributed.init_method == "tcp":
            if "MASTER_ADDR" in os.environ:
                master_addr = os.environ['MASTER_ADDR']
            else:
                master_addr = "localhost"
            if "MASTER_PORT" in os.environ:
                master_port = os.environ["MASTER_PORT"]
            else:
                master_port = 6008

            distributed_init_method = "tcp://%s:%s" % (master_addr, master_port)
            logger.info(f"distributed_init_method : {distributed_init_method}")
            torch.distributed.init_process_group(
                backend='nccl',
                init_method=distributed_init_method,
                rank=local_rank,
                world_size=world_size)

        torch.cuda.set_device(local_rank) 
        torch.cuda.empty_cache()
        # set distributed sampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset)

    model.cuda()
    Conv.cuda()
    disc_model.cuda()

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=config.datasets.batch_size,
        sampler=train_sampler, 
        shuffle=(train_sampler is None), collate_fn=collate_fn,
        pin_memory=config.datasets.pin_memory,
        num_workers= config.datasets.num_workers)
    testloader = torch.utils.data.DataLoader(
        testset,
        # batch_size=config.datasets.batch_size,
        batch_size= 1,  
        sampler=test_sampler, 
        shuffle=False, collate_fn=collate_fn,
        pin_memory=config.datasets.pin_memory,
        num_workers=config.datasets.num_workers)
    logger.info(f"There are {len(trainloader)} data to train the Mimi")
    logger.info(f"There are {len(testloader)} data to test the Mimi")

    # set optimizer and scheduler, warmup scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    params_conv = [p for p in Conv.parameters() if p.requires_grad]
    disc_params = [p for p in disc_model.parameters() if p.requires_grad]
    optimizer = optim.Adam([{'params': itertools.chain(params,params_conv), 'lr': config.optimization.lr}], betas=(0.5, 0.9))
    optimizer_disc = optim.Adam([{'params':disc_params, 'lr': config.optimization.disc_lr}], betas=(0.5, 0.9))
    scheduler = WarmupCosineLrScheduler(optimizer, max_iter=300000, eta_ratio=0.1, warmup_iter= config.lr_scheduler.warmup_step, warmup_ratio=1e-4)
    disc_scheduler = WarmupCosineLrScheduler(optimizer_disc, max_iter=300000, eta_ratio=0.1, warmup_iter= config.lr_scheduler.warmup_step, warmup_ratio=1e-4)

    scaler = GradScaler() if config.common.amp else None
    scaler_disc = GradScaler() if config.common.amp else None  

    if config.checkpoint.resume and 'scheduler_state_dict' in model_checkpoint.keys() and 'scheduler_state_dict' in disc_model_checkpoint.keys(): 
        optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(model_checkpoint['scheduler_state_dict'])
        optimizer_disc.load_state_dict(disc_model_checkpoint['optimizer_state_dict'])
        disc_scheduler.load_state_dict(disc_model_checkpoint['scheduler_state_dict'])
        logger.info(f"load optimizer and disc_optimizer state_dict from {resume_epoch}")

    if config.distributed.data_parallel:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        Conv = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Conv)
        disc_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(disc_model)
        # wrap the model by using DDP
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            find_unused_parameters=config.distributed.find_unused_parameters)
        disc_model = torch.nn.parallel.DistributedDataParallel(
            disc_model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            find_unused_parameters=config.distributed.find_unused_parameters)
        Conv = torch.nn.parallel.DistributedDataParallel(
            Conv,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            find_unused_parameters=config.distributed.find_unused_parameters)
    if not config.distributed.data_parallel or dist.get_rank() == 0:  
        writer = SummaryWriter(log_dir=f'{config.checkpoint.save_folder}/runs')  
        logger.info(f'Saving tensorboard logs to {Path(writer.log_dir).resolve()}')
    else:  
        writer = None  
    start_epoch = max(1,resume_epoch+1) # start epoch is 1 if not resume
    # instantiate loss balancer
    balancer = Balancer(dict(config.balancer.weights)) if hasattr(config, 'balancer') else None
    if balancer:
        logger.info(f'Loss balancer with weights {balancer.weights} instantiated')
    distill_type = config.distill.distill_type
    test(0, distill_type, model, disc_model, testloader, config, writer)
    total_step = max(0, resume_step)

    model_file = ""
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_file)
    feat_model = WavLMModel.from_pretrained(model_file).eval().to('cuda')

    for epoch in range(start_epoch, config.common.max_epoch+1):
        total_step = train_one_step(
            epoch, optimizer, optimizer_disc, distill_type,
            model, Conv, disc_model, trainloader,config,
            scheduler,disc_scheduler,scaler,scaler_disc,writer,balancer,total_step = total_step,feature_extractor = feature_extractor,feat_model = feat_model)
        if epoch % config.common.test_interval == 0:
            test(epoch,distill_type,model,disc_model,testloader,config,writer,feature_extractor = feature_extractor,feat_model = feat_model)

    if config.distributed.data_parallel:
        dist.destroy_process_group()

@hydra.main(config_path='config', config_name='config')
def main(config):
    # set distributed debug, if you encouter some multi gpu bug, please set torch_distributed_debug=True
    if config.distributed.torch_distributed_debug: 
        os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
        os.environ["TORCH_DISTRIBUTED_DEBUG"]="DETAIL"
    if not os.path.exists(config.checkpoint.save_folder):
        os.makedirs(config.checkpoint.save_folder)
    # disable cudnn
    torch.backends.cudnn.enabled = False
    # set distributed
    if config.distributed.data_parallel:  
        world_size = config.distributed.world_size  
        if config.distributed.init_method == "tmp":  
            import tempfile  
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:  
                start_dist_train(train, world_size, config, tmp_file.name)  
        elif config.distributed.init_method == "tcp":  
            start_dist_train(train, world_size, config)  
    else:  
        train(1, 1, config)  # set single gpu train 


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()
