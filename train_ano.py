#!/usr/bin/env python3

import os
import argparse
from copy import deepcopy
from functools import partial
import math
import json
from pathlib import Path

import accelerate
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch import multiprocessing as mp
from torch.utils import data
from torchvision import datasets, transforms, utils
from tqdm.auto import trange, tqdm

import k_diffusion as K
# from k_diffusion.utils import get_dis_thresh, get_gen_thresh

from torchmetrics.functional import auroc #, accuracy, precision, f1_score

import random
import numpy as np
import pandas as pd

GLOBAL_SEED = 42


def seed_everything(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    global GLOBAL_SEED
    GLOBAL_SEED = seed
    # torch.use_deterministic_algorithms(True)

def args_parser():

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', type=str, required=True,
                   help='the configuration file')
    p.add_argument('--search_id', type=int, default=0,
                   help='search id to init parameters')

    p.add_argument('--batch_size', type=int, default=256,
                   help='the batch size')
    p.add_argument('--batch_test', type=int, default=256,
                   help='the test batch size')
    p.add_argument('--n_test_batch', type=int, default=10,
                   help='the test batch groups ')
    p.add_argument('--n_epochs', type=int, default=30,
                   help='number of epoch')
    

    p.add_argument('--dataset', type=str,
                   help='dataset_name')
    p.add_argument('--feat_type', type=str,
                   help='feature data type name')
    p.add_argument('--feat', type=str,
                   help='feature backbone name')
    p.add_argument('--cond_type', type=str,
                   help='conditioning data type name')
    p.add_argument('--cond', type=str,
                   help='conditioning backbone name')
    p.add_argument('--ds_no_scale', action='store_true', default=False,
                   help='dataset scaling flag')
    p.add_argument('--ext', type=str,
                   help='naming extention')
    
    p.add_argument('--grad-accum-steps', type=int, default=1,
                   help='the number of gradient accumulation steps')
    p.add_argument('--grow', type=str,
                   help='the checkpoint to grow from')
    p.add_argument('--grow-config', type=str,
                   help='the configuration file of the model to grow from')
    p.add_argument('--lr', type=float,
                   help='the learning rate')
    
    p.add_argument('--n_sample', type=int,
                   help='evaluate sample size')
    p.add_argument('--n_denoise', type=int, default=10,
                   help='denoising steps')

    p.add_argument('--name', type=str, default='model',
                   help='the name of the run')
    p.add_argument('--num-workers', type=int, default=6,
                   help='the number of data loader workers')
    p.add_argument('--resume', type=str,
                   help='the checkpoint to resume from')
    p.add_argument('--sample-n', type=int, default=64,
                   help='the number of images to sample for demo grids')
                   
    p.add_argument('--seed', type=int,  default=42,
                   help='the random seed')
    p.add_argument('--start-method', type=str, default='fork',
                   choices=['fork', 'forkserver', 'spawn'],
                   help='the multiprocessing start method')
    
    p.add_argument('--save_epoch', action='store_true', default=False,
                   help='save model at each epoch')

    # wandb params
    p.add_argument('--use_wandb', action='store_true', default=False,
                   help='use wandb')
    p.add_argument('--wandb-entity', type=str,
                   help='the wandb entity name')
    p.add_argument('--wandb-group', type=str, 
                   help='the wandb group name')
    p.add_argument('--wandb-project', type=str,
                   help='the wandb project name (specify this to enable wandb)')
    p.add_argument('--wandb-save-model', action='store_true',
                   help='save model to wandb')
    return p.parse_args()


def get_datasets(dataset_name, feat_type, feat_model, cond_type, cond_model, no_scale=False):
    from custom_datasets import feature_dataset
    
    dataset_root: str = '../data'


    train_set = feature_dataset.ClipDataset(
        root_dir=dataset_root,
        dataset_name=dataset_name,
        feat_model=feat_model,
        cond_model=cond_model,
        feat_type=feat_type,
        cond_type=cond_type,
        split='train',
        no_overlap=dataset_name == 'UCFC',
        anomaly=dataset_name == 'UCFC',
        no_scale=no_scale,
        )
    
    test_set = feature_dataset.ClipDataset(
        root_dir=dataset_root,
        dataset_name=dataset_name,
        feat_model=feat_model,
        cond_model=cond_model,
        feat_type=feat_type,
        cond_type=cond_type,
        split='test',
        no_overlap=False,
        no_scale=no_scale,
    )
    return train_set, test_set

def main():
    args = args_parser()

    seed_everything(seed=args.seed)

    mp.set_start_method(args.start_method)
    torch.backends.cuda.matmul.allow_tf32 = True

    ext = f'_{args.ext}' if args.ext else ''

    config = K.config.load_config(open(args.config))
    model_config = config['model']
    dataset_config = config['dataset']
    opt_config = config['optimizer']
    sched_config = config['lr_sched']
    ema_sched_config = config['ema_sched']

    search_params = pd.read_csv('search_params.csv')

    search_id = args.search_id
    params = search_params.iloc[int(search_id)]
    
    print('search id:', search_id)
    print('params:')
    print(params,'\n')


    dataset_name = args.dataset if args.dataset else 'shanghai' # 'shanghai', 'UCFC'
    
    if args.cond:
        dataset_config['cond_model'] = args.cond
    cond_model = dataset_config['cond_model']
    if args.feat:
        dataset_config['feat_model'] = args.feat
    feat_model = dataset_config['feat_model']
    
    if args.cond_type:
        dataset_config['cond_type'] = args.cond_type
    cond_type = dataset_config['cond_type']
    if args.feat_type:
        dataset_config['feat_type'] = args.feat_type
    feat_type = dataset_config['feat_type']


    if feat_model == 'r3d18' or feat_model == 'res18':
        model_config['input_size'] = 512
    elif feat_model == 'rx3d' or feat_model == 'res50':
        model_config['input_size'] = 2048
    else:
        model_config['input_size'] = 256

    if cond_model == 'res18' or cond_model == 'r3d18':
        model_config['mapping_cond_dim'] = 512
    elif cond_model == 'res50' or cond_model == 'rx3d':
        model_config['mapping_cond_dim'] = 2048
    else:
        model_config['mapping_cond_dim'] = 256
    

    cond_size = model_config['mapping_cond_dim']
    feat_size = model_config['input_size']
    train_set, test_set = get_datasets(dataset_name, feat_type, feat_model, cond_type, cond_model, args.ds_no_scale)
    ss_data = train_set.std_data
    
    
    log_dir = f'runs/{dataset_name}/{feat_type}_{feat_model}/{cond_type}_{cond_model}_b{args.batch_size}_{search_id}{ext}'
    K.utils.mkdir(log_dir)

    args.name = f'{cond_type}_{cond_model}_b{args.batch_size}_{search_id}{ext}'
    args.wandb_project = f'{feat_type}_{feat_model}_{dataset_name}'
    args.wandb_entity = '-------'

    n_denoise = args.n_denoise
    
    is_cpu = False
    

    model_config['sigma_max'] = params['sigma_max']
    model_config['sigma_min'] = params["sigma_min"]
    model_config['sigma_data'] = ss_data
    model_config['sigma_sample_density']['mean'] = params['ssd_mean']
    model_config['sigma_sample_density']['std'] = params['ssd_std']

    opt_config['lr'] = params['lr']
    opt_config['weight_decay'] = params['weight_decay']


    if model_config["sampler"] == "lms":
        p_sampler_fn = partial(K.sampling.sample_lms, disable=True)
    elif model_config["sampler"] == "heun":
        p_sampler_fn = partial(K.sampling.sample_heun, disable=True)
    elif model_config["sampler"] == "dpm2":
        p_sampler_fn = partial(K.sampling.sample_dpm_2, disable=True)
    else:
        print('unknown sampler method')
        ValueError('Invalid sampler method')



    # ddp_kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=model_config['skip_stages'] > 0)
    accelerator = accelerate.Accelerator(#kwargs_handlers=[ddp_kwargs],
                                         gradient_accumulation_steps=args.grad_accum_steps,
                                         cpu=is_cpu)
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)

    seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes],
                            generator=torch.Generator().manual_seed(GLOBAL_SEED))
    torch.manual_seed(seeds[accelerator.process_index])



    # inner_model = K.config.make_model_aot(config)
    gvad_model = K.models.ConditionedGVADModel(
        feat_size,
        cond_size,
    )

    # If logging to wandb, initialize the run
    use_wandb = args.use_wandb
    if use_wandb:
        import wandb
        log_config = vars(args)
        log_config['config'] = config
        log_config['parameters'] = K.utils.n_params(gvad_model)
        wandb.init(project=args.wandb_project,
                   name=args.name,
                   entity=args.wandb_entity,
                   group=args.wandb_group,
                   config=log_config, save_code=True)


    if opt_config['type'] == 'adamw':
        opt = optim.AdamW(gvad_model.parameters(),
                          lr=opt_config['lr'] if args.lr is None else args.lr,
                          betas=tuple(opt_config['betas']),
                          eps=opt_config['eps'],
                          weight_decay=opt_config['weight_decay'])
    elif opt_config['type'] == 'sgd':
        opt = optim.SGD(gvad_model.parameters(),
                        lr=opt_config['lr'] if args.lr is None else args.lr,
                        momentum=opt_config.get('momentum', 0.),
                        nesterov=opt_config.get('nesterov', False),
                        weight_decay=opt_config.get('weight_decay', 0.))
    else:
        raise ValueError('Invalid optimizer type')

    if sched_config['type'] == 'inverse':
        sched = K.utils.InverseLR(opt,
                                  inv_gamma=sched_config['inv_gamma'],
                                  power=sched_config['power'],
                                  warmup=sched_config['warmup'])
    elif sched_config['type'] == 'exponential':
        sched = K.utils.ExponentialLR(opt,
                                      num_steps=sched_config['num_steps'],
                                      decay=sched_config['decay'],
                                      warmup=sched_config['warmup'])
    else:
        raise ValueError('Invalid schedule type')

    assert ema_sched_config['type'] == 'inverse'
    ema_sched = K.utils.EMAWarmup(power=ema_sched_config['power'],
                                  max_value=ema_sched_config['max_value'])

    if accelerator.is_main_process:
        try:
            print('Number of items in dataset:', len(train_set))
            print('Number of items in testset:', len(test_set))
        except TypeError:
            pass

    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True,
                               num_workers=args.num_workers, persistent_workers=True)

    indices = np.load(f'indices_{test_set.dataset_name}.npy')
    test_part = args.n_test_batch * 8192
    indices = indices[:test_part]
    test_set = data.Subset(test_set, indices)
    test_dl = data.DataLoader(test_set, args.batch_test, shuffle=False, drop_last=False,
                              num_workers=args.num_workers, persistent_workers=True)

    model, opt, train_dl, test_dl = accelerator.prepare(gvad_model, opt, train_dl, test_dl)

    # if use_wandb:
    #     wandb.watch(model)

    sigma_min = model_config['sigma_min']
    sigma_max = model_config['sigma_max']
    sample_density = K.config.make_sample_density(model_config)

    seed_noise_path = f'seed_noise_{feat_size}.pth'
    if os.path.exists(seed_noise_path):
        seed_noise = torch.load(seed_noise_path, map_location=device)
    else:
        seed_noise = torch.randn([1, feat_size], device=device)
        torch.save(seed_noise, seed_noise_path)

    model = K.config.make_denoiser_wrapper(config)(model)
    model_ema = deepcopy(model)

    state_path = log_dir + '/state.json'
    

    step = 0


    if accelerator.is_main_process:
        metrics_log = K.utils.CSVLogger(f'{log_dir}/metrics.csv', ['epoch', 'step', 'auc', 'n_start', 'th_alpha'])
        loss_log = K.utils.CSVLogger(f'{log_dir}/loss.csv', ['epoch', 'step', 'loss'])


    threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0]

    @torch.no_grad()
    @K.utils.eval_mode(model_ema)
    def evaluate(n_start, epoch):
        
        if accelerator.is_main_process:
            tqdm.write('Evaluating...')
        
        sigmas = K.sampling.get_sigmas_karras(n_denoise, sigma_min, sigma_max, rho=7., device=device)
        sigmas = sigmas[n_start:]
        sample_noise = seed_noise.to(device) * sigmas[0]
        
        def sample_fn(x_real, x_feat):
            x_real = x_real.to(device)
            x_feat = x_feat.to(device)
            x = sample_noise + x_real
            x_0 = p_sampler_fn(model_ema.forward, x, sigmas, extra_args={"cond":x_feat})
            
            g_dists = model_ema.gvad_model.loss(x_real, x_0)
            
            return g_dists

        g_dists, labels = K.evaluation.compute_eval_outs_aot(accelerator, sample_fn, test_dl)
        
        if accelerator.is_main_process:
            n_sample = len(g_dists)
            n_batch_p = math.floor(n_sample / 8192)
            n_batch = math.ceil(n_sample / 8192)
            best_auc = -1.0
            best_th = 0.1
            for t_alpha in threshold_values:
                if n_batch_p < n_batch:
                    # last batch
                    remainders = g_dists[n_batch_p*8192:]
                    g_th_r = model_ema.gvad_model.get_dis_thresh(remainders, t_alpha)
                    g_preds_r = (remainders > g_th_r).float()

                    batched = torch.reshape(g_dists[:n_batch_p*8192], (-1, 8192))
                    g_th = model_ema.gvad_model.get_dis_thresh(batched, t_alpha)
                    g_preds = (batched.T > g_th).T.float().flatten()
                    g_preds = torch.cat([g_preds, g_preds_r])
                else:
                    batched = torch.reshape(g_dists[:n_batch_p*8192], (-1, 8192))
                    g_th = model_ema.gvad_model.get_dis_thresh(batched, t_alpha)
                    g_preds = (batched.T > g_th).T.float().flatten()
                
                auc_val = auroc(g_preds, labels).item()
                if auc_val > best_auc:
                    best_auc = auc_val
                    best_th = t_alpha
                metrics_log.write(epoch, step, auc_val, n_start, t_alpha)
            
            
            d_std, d_mean = torch.std_mean(g_dists)
            d_std, d_mean = d_std.item(), d_mean.item()
            d_min, d_max = torch.min(g_dists).item(), torch.max(g_dists).item()

            print(f'\nAUC_{n_start}: {best_auc:g}, best th: {best_th:g}')
            print(f'dist stats:\nmin: {d_min:g} max: {d_max:g}\nmean: {d_mean:g} std: {d_std:g}')
            
            if use_wandb:
                wandb.log({f'AUCs/auc_{n_start}'  : best_auc,f'thresh/best_alpha_{n_start}' : best_th,
                           f'distance/min_{n_start}': d_min, f'distance/max_{n_start}' : d_max,
                           f'distance/std_{n_start}': d_std, f'distance/mean_{n_start}': d_mean}, step=step)

    def save(epoch):
        accelerator.wait_for_everyone()
        filename = f'{log_dir}/{epoch:04}e.pth'
        if accelerator.is_main_process:
            tqdm.write(f'Saving to {filename}...')
        obj = {
            'model': accelerator.unwrap_model(model.gvad_model).state_dict(),
            'model_ema': accelerator.unwrap_model(model_ema.gvad_model).state_dict(),
            'opt': opt.state_dict(),
            'sched': sched.state_dict(),
            'ema_sched': ema_sched.state_dict(),
            'epoch': epoch,
            'step': step,
        }
        accelerator.save(obj, filename)
        if accelerator.is_main_process:
            state_obj = {'latest_checkpoint': filename}
            json.dump(state_obj, open(state_path, 'w'))
        if args.wandb_save_model and use_wandb:
            wandb.save(filename)


    if n_denoise == 10:
        denoise_list = list(range(10))
    elif n_denoise == 50:
        denoise_list = [0, 5, 11, 16, 22, 27, 33, 38, 44, 49]
    else:
        denoise_list = list(range(n_denoise))

    for i in denoise_list:
        evaluate(i, 0)

    try:
        for epoch in range(args.n_epochs):
            for batch in tqdm(train_dl, disable=not accelerator.is_main_process):
                with accelerator.accumulate(model):
                    reals = batch['data']
                    cond = batch['cond']
                    noise = torch.randn_like(reals)
                    sigma = sample_density([reals.shape[0]], device=device)
                    gloss = model.loss(reals, noise, sigma, cond=cond)  # losses with the batch
                    gloss = accelerator.gather(gloss)
                    loss = gloss.mean()

                    accelerator.backward(loss)
                    opt.step()
                    sched.step()
                    opt.zero_grad()
                    if accelerator.sync_gradients:
                        ema_decay = ema_sched.get_value()
                        K.utils.ema_update(model, model_ema, ema_decay)
                        ema_sched.step()

                if accelerator.is_main_process:
                    if step % 25 == 0:
                        loss_log.write(epoch, step, loss.item())
                        tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss.item():g}')

                    if use_wandb:
                        log_dict = {
                            'epoch': epoch,
                            'loss': loss.item(),
                            'lr': sched.get_last_lr()[0],
                            'ema_decay': ema_decay,
                        }
                        wandb.log(log_dict, step=step)

                step += 1
                
            if args.save_epoch:
                save(epoch=epoch)

            for i in denoise_list:
                evaluate(i, epoch)
            

    except KeyboardInterrupt:
        pass





if __name__ == '__main__':
    print('Hello there!')
    os.environ['WANDB_API_KEY'] = "****************************"
    main()
    print('Obiwan Kenobi.')

