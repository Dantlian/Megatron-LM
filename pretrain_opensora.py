# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain VIT"""

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from functools import partial

import mindspeed.megatron_adaptor
from megatron.training import get_args, get_timers, print_rank_0
from megatron.core.enums import ModelType
from megatron.core import mpu
from megatron.legacy.data.vit_dataset import build_train_valid_datasets
from megatron.training import pretrain
from megatron.training.utils import average_losses_across_data_parallel_group
from megatron.training.arguments import core_transformer_config_from_args
from opensora.acceleration.parallel_states import get_data_parallel_group
from opensora.schedulers.iddpm.diffusion_utils import continuous_gaussian_log_likelihood

from opensora.utils.config_utils import parse_configs, merge_args
from opensora.utils.train_utils import  update_ema
from opensora.utils.misc import to_torch_dtype,requires_grad

from mindspeed.utils import get_batch_on_this_cp_rank
from opensora.datasets import DatasetFromCSV, get_transforms_video, get_transforms_image, prepare_dataloader
from opensora.registry import MODELS, SCHEDULERS, build_module
# from opensora.utils.ckpt_utils import model_sharding

from mmengine.config import Config

scheduler = None
vae = None
text_encoder = None
cfg = {}


def initialize():
    def initialize_models():
        global cfg
        global vae
        global text_encoder

        vae_cfg = cfg['vae']
        text_encoder_cfg = cfg['text_encoder']
        if  mpu.get_tensor_model_parallel_rank() == 0:
            vae = build_module(vae_cfg, MODELS)
            text_encoder = build_module(text_encoder_cfg, MODELS, device=torch.cuda.current_device())
        torch.distributed.barrier()
        if  mpu.get_tensor_model_parallel_rank() != 0:
            vae = build_module(vae_cfg, MODELS)
            text_encoder = build_module(text_encoder_cfg, MODELS, device=torch.cuda.current_device())

        dtype = to_torch_dtype("bf16")
        vae = vae.to(torch.cuda.current_device(), dtype)

        cfg['latent_size'] = vae.get_latent_size(cfg['input_size'])

    def initialize_scheduler():
        global scheduler
        scheduler = build_module(cfg['scheduler'], SCHEDULERS)

    def initialize_config():
        args = get_args()
        global cfg
        config = Config.fromfile(args.config)

        cfg['vae'] = config.vae
        cfg['text_encoder'] = config.text_encoder
        cfg['scheduler'] = config.scheduler
        cfg['model'] = config.model
        cfg['num_frames'] = config.num_frames
        cfg['frame_interval'] = config.frame_interval
        cfg['image_size'] = config.image_size
        cfg['num_workers'] = config.num_workers
        cfg['root'] = config.root
        cfg['use_image_transform'] = config.use_image_transform
        cfg['input_size'] = (cfg['num_frames'], *cfg['image_size'])  
        cfg['micro_batch_size'] = args.micro_batch_size
        
        print("cfg:",cfg)

    initialize_config()
    initialize_scheduler()
    initialize_models()


def initialize_pipeline_tensor_shaped(hidden_size):
    micro_batch_size = cfg['micro_batch_size']
    latent_size = cfg['latent_size']
    text_encoder_maxlen = text_encoder.model_max_length

    setattr(forward_step, 'pipeline_tensor_shapes',
        [(micro_batch_size, text_encoder.output_dim, hidden_size),(micro_batch_size, vae.out_channels, *latent_size),
        (micro_batch_size, 1, text_encoder_maxlen, hidden_size), (micro_batch_size),
        (micro_batch_size, hidden_size * 6), (micro_batch_size, text_encoder_maxlen),
        (micro_batch_size, vae.out_channels, *latent_size),
         (micro_batch_size, vae.out_channels, *latent_size)
         ,(2, 1152)])


def model_sharding(model: torch.nn.Module):
    global_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    for _, param in model.named_parameters():
        padding_size = (world_size - param.numel() % world_size) % world_size
        if padding_size > 0:
            padding_param = torch.nn.functional.pad(param.data.view(-1), [0, padding_size])
        else:
            padding_param = param.data.view(-1)
        splited_params = padding_param.split(padding_param.numel() // world_size)
        splited_params = splited_params[global_rank]
        param.data = splited_params

rank_list= []
def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    dtype = to_torch_dtype("bf16")

    latent_size = cfg['latent_size']
    print("latent_size:",latent_size)
    stdit = build_module(
        cfg['model'],
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        dtype=dtype,
    )

    # ema = deepcopy(stdit).to(torch.float32).to(torch.cuda.current_device())
    # requires_grad(ema, False)

    state_dict  = torch.load('/home/l00618052/Megatron-LM/my_model_state_dict.pth')
    
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()


    hooks = ['final_layer.linear.weight', '']


    def print_func(inputs, prefix):
        if isinstance(inputs, tuple):
            for i in inputs:
                print_func(i, prefix)
        elif isinstance(inputs, torch.Tensor):
            print(prefix, inputs.shape, inputs.dtype, inputs)
        else:
            print(prefix, inputs)
        


    def hook_func(name, module):
        def hook_function(module, inputs, outputs):
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
            if 'blocks' in name:
                return
            global rank_list
            if name + str(pp_rank)+":"+str(vpp_rank) not in rank_list:
                rank_list.append(name + str(pp_rank)+":"+str(vpp_rank))

                print('================================================')
                print(module)
                print_func(inputs, name+' inputs')
                print_func(outputs, name+' outputs')
        return hook_function

    #===============预加载模型测试-START=================
    if pp_size <= 1:
        stdit.load_state_dict(state_dict)
        # def print_grad_info(name):
        #     def hook(grad):
        #         if name.startswith('blocks'):
        #             return
        #         print(f"Parameter Name: {name}, Shape: {grad.shape},\n Gradient: {grad}\n")

        #     return hook

        # for name, param in stdit.named_parameters():
        #     if param.requires_grad:
        #         param.register_hook(print_grad_info(name))

        for name, module in stdit.named_modules():
            if module is not None:
                module.register_forward_hook(hook_func('[forward]:' + name, module))
                module.register_backward_hook(hook_func('[backward]:' + name, module))
                        

        return stdit

    pp_rank = mpu.get_pipeline_model_parallel_rank()
    vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
    if vpp_size is None or vpp_size <=1:
        # pp_rank = 3
        pp_blocks = []
        if pp_size == 4:
            if pp_rank == 0:
                pp_blocks = [0, 1, 2]
            if pp_rank == 1:
                pp_blocks = [3, 4, 5]
            if pp_rank == 2:
                pp_blocks = [6, 7, 8]
            if pp_rank == 3:
                pp_blocks = [9, 10, 11]
        if pp_size == 2:
            if pp_rank == 0:
                pp_blocks = [0, 1, 2, 3, 4, 5]
            if pp_rank == 1:
                pp_blocks = [6, 7, 8, 9, 10, 11]


        # if pp_size == 8:
        #     if pp_rank == 0:
        #         pp_blocks = [0]
        #     if pp_rank == 1:
        #         pp_blocks = [1]
        #     if pp_rank == 2:
        #         pp_blocks = [2]
        #     if pp_rank == 3:
        #         pp_blocks = [3]
        #     if pp_rank == 4:
        #         pp_blocks = [4]
        #     if pp_rank == 5:
        #         pp_blocks = [5]
        #     if pp_rank == 6:
        #         pp_blocks = [6]
        #     if pp_rank == 7:
        #         pp_blocks = [7]

        # if pp_size == 4:
        #     if pp_rank == 0:
        #         pp_blocks = [0, 1]
        #     if pp_rank == 1:
        #         pp_blocks = [2, 3]
        #     if pp_rank == 2:
        #         pp_blocks = [4, 5]
        #     if pp_rank == 3:
        #         pp_blocks = [6, 7]
        # if pp_size == 2:
        #     if pp_rank == 0:
        #         pp_blocks = [0, 1, 2, 3]
        #     if pp_rank == 1:
        #         pp_blocks = [ 4, 5, 6, 7]

    else:
        pp_block_index = pp_size*vpp_rank + pp_rank
        print("pp_rank:",pp_rank," vpp_rank:",vpp_rank," pp_block_index:",pp_block_index)
        pp_blocks = [pp_block_index]

    partial_state_dict = {}
    for name, param in state_dict.items():
        if name.startswith('blocks'):
            for i in range(0, len(pp_blocks)):
                key_parts = name.split(".")
                if(key_parts[1] == str(pp_blocks[i])):
                    key_parts[1] = str(i)
                    key = '.'.join(key_parts)
                    partial_state_dict[key] = param
                    break

        else:
            if name.startswith('pos_embed') or name.startswith('x_embedder') \
                or name.startswith('t_embedder') or name.startswith('t_block') \
                or name.startswith('y_embedder') or name.startswith('pos_embed'):
                if mpu.is_pipeline_first_stage():
                    partial_state_dict[name] = param

            if name.startswith('final_layer') or name.startswith('t_embedder') :
                if mpu.is_pipeline_last_stage():
                    partial_state_dict[name] = param


    stdit.load_state_dict(partial_state_dict, strict=False)
    # torch.save(stdit.state_dict(), 'tmp.pth')

    ema = deepcopy(stdit)
    requires_grad(ema, False)
    stdit.ema = ema
    update_ema(ema, stdit, decay=0, sharded=False)
    ema.eval()
    # model_sharding(ema)
    new_stdit_state_dict = stdit.state_dict()

    # def print_grad_info(name):
    #     def hook(grad, inputs, outputs):
    #         global rank_list
    #         # print(f"rank_list: {rank_list}\n")

    #         pp_rank = mpu.get_pipeline_model_parallel_rank()
    #         vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
    #         # if name.startswith('blocks'):
    #         #     return
    #         if name + str(pp_rank)+":"+str(vpp_rank) not in rank_list:
    #             rank_list.append(name + str(pp_rank)+":"+str(vpp_rank))
    #             print(f"Parameter Name: {name}, Shape: {grad.shape}, pp Rank: {pp_rank}, vpp Rank: {vpp_rank},\n Gradient: {grad}\n Input: {input}\n  Output: {output}\n")
    #     return hook

    # for name, param in stdit.named_parameters():
    #     if param.requires_grad:
    #         param.register_hook(print_grad_info(name))

    # def hook_func(name, module):
    #     def hook_function(module, inputs, outputs):
    #         global rank_list

    #         if 'blocks' in name:
    #             return
    #         print_inputs = inputs
    #         print_outputs = outputs
    #         if name + str(pp_rank)+":"+str(vpp_rank) not in rank_list:
    #             rank_list.append(name + str(pp_rank)+":"+str(vpp_rank))
    #         #     if inputs[0] is None:
    #         #         print_inputs = None
    #         #     else:
    #         #         inputs_num_dims = inputs[0].dim()
    #         #         if inputs_num_dims == 0:
    #         #             print_inputs = inputs[0]
    #         #         else:
    #         #             print_inputs = inputs[0][..., -1]

    #         #     if outputs[0] is None:
    #         #         print_outputs = None
    #         #     else:
    #         #         outputs_num_dims = outputs[0].dim()
    #         #         if outputs_num_dims == 0:
    #         #             print_outputs = outputs[0]
    #         #         else:
    #         #             print_outputs = outputs[0][..., -1]

    #             print(f"MODULE Parameter Name: {name},  \n inputs: {print_inputs}, \n outputs: {print_outputs}, \n pp Rank: {pp_rank}, vpp Rank: {vpp_rank}\n")

    #             # print(f"MODULE Parameter Name: {name}, \n input.shape:{print_inputs.shape}, \n inputs: {print_inputs}, \n outputs.shape:{print_outputs.shape}, \n outputs: {print_outputs}, \n pp Rank: {pp_rank}, vpp Rank: {vpp_rank}\n")
    #     return hook_function


    # def hook_func(name, module):
    #     def hook_function(module, grad_input, grad_output):
    #         global rank_list

    #         # if name.startswith('blocks'):
    #         #     return
    #         if name + str(pp_rank)+":"+str(vpp_rank) not in rank_list:
    #             rank_list.append(name + str(pp_rank)+":"+str(vpp_rank))
    #             if grad_input[0] is None:
    #                 print_inputs = None
    #             else:
    #                 inputs_num_dims = grad_input[0].dim()
    #                 if inputs_num_dims == 0:
    #                     print_inputs = grad_input[0]
    #                 else:
    #                     print_inputs = grad_input[0][..., -1]

    #             if grad_output[0] is None:
    #                 print_outputs = None
    #             else:
    #                 outputs_num_dims = grad_output[0].dim()
    #                 if outputs_num_dims == 0:
    #                     print_outputs = grad_output[0]
    #                 else:
    #                     print_outputs = grad_output[0][..., -1]

    #             print(f"MODULE Parameter Name: {name}, \n inputs: {print_inputs}, \n outputs: {print_outputs}, \n pp Rank: {pp_rank}, vpp Rank: {vpp_rank}\n")
    #     return hook_function


    # for name, module in stdit.named_modules():
    #     if module is not None:
    #         module.register_forward_hook(hook_func('[forward]:' + name, module))
    #         module.register_backward_hook(hook_func('[backward]:' + name, module))
                    
  #===============预加载模型测试-END=================
    initialize_pipeline_tensor_shaped(stdit.hidden_size)
    return stdit


def get_batch_on_this_tp_rank(data_iterator):
    args = get_args()

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(),
                                        group=mpu.get_tensor_model_parallel_group())

    global vae
    global text_encoder

    dtype = to_torch_dtype("bf16")

    if mpu.get_tensor_model_parallel_rank() == 0:
        if data_iterator is not None:
            batch = next(data_iterator)
        else:
            batch = None
        x = batch['video'].to(torch.cuda.current_device(), dtype)
        y = batch['text']

        with torch.no_grad():
            # Prepare visual inputs
            x = vae.encode(x).contiguous()  # [B, C, T, H/P, W/P]  #[2, 4, 16, 32, 32]
            # Prepare text inputs
            encoded_text = text_encoder.encode(y)

        y = encoded_text['y'].contiguous()
        mask = encoded_text['mask'].contiguous()


        _broadcast(x)
        _broadcast(y)
        _broadcast(mask)

    else:
        latent_size = cfg['latent_size']
        micro_batch_size = cfg['micro_batch_size']

        text_encoder_maxlen = text_encoder.model_max_length
        text_encoder_output_dim = text_encoder.output_dim

        x = torch.empty((micro_batch_size, vae.out_channels, *latent_size),
                             dtype=dtype, device=torch.cuda.current_device())
        y = torch.empty((micro_batch_size, 1, text_encoder_maxlen, text_encoder_output_dim), dtype=torch.float32, device=torch.cuda.current_device())
        mask = torch.empty((micro_batch_size, text_encoder_maxlen), dtype=torch.int64, device=torch.cuda.current_device())

        _broadcast(x)
        _broadcast(y)
        _broadcast(mask)

    batch = {
        'x': x,
        'y': y,
        'mask': mask
    }
    return batch


def get_batch(data_iterator):
    """Build the batch."""
    # cfg = parse_configs(training=True)
    device = torch.cuda.current_device()
    # dtype = to_torch_dtype(cfg.dtype)
    dtype = to_torch_dtype("bf16")

    if (mpu.is_pipeline_first_stage()):# or mpu.is_pipeline_last_stage()):
        batch = get_batch_on_this_tp_rank(data_iterator)
        batch =  get_batch_on_this_cp_rank(batch)
        return batch['x'], batch['y'], batch['mask']
    else:
        return None, None, None


def loss_func(x_t, x_0, t, noise, output_tensor):
    loss_dict = scheduler.training_losses(output_tensor, x_t, x_0, t, noise = noise)

    loss1 = float(loss_dict["loss"][0])
    loss2 = float(loss_dict["loss"][1])
    # loss3 = float(loss_dict["loss"][2])
    # loss4 = float(loss_dict["loss"][3])
    print("loss_dict:",[loss1,loss2], "\n")


    loss = loss_dict["loss"].mean()

    print("losss:",float(loss), "\n")
    print(" output_tensor[0][0]:",output_tensor[0][0][0][0][0],"\n")
    print(" x_t[0][0]:",x_t[0][0][0][0][0],"\n")
    print(" x_0[0][0]:",x_0[0][0][0][0][0],"\n")
    print(" noise[0][0]:",noise[0][0][0][0][0],"\n")
    # print(" ttt:",t,"\n")
    # print(" yyy:",y,"\n")
    # print(" t0t0t0:",t0,"\n")


    averaged_loss = average_losses_across_data_parallel_group([loss])

    loss = loss.unsqueeze(0)
    print(" averaged loss:",averaged_loss[0],"\n")
    return loss, {"loss": averaged_loss[0]}

iter_index = 0

def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()

    # Get the batch.
    x, y, mask = get_batch(data_iterator)

    num_timesteps = 1000
    micro_bs = args.micro_batch_size
    dtype = to_torch_dtype("bf16")
    # t = torch.randint(0, num_timesteps, (micro_bs,),device=torch.cuda.current_device(), dtype = dtype)
    # t = torch.clamp(t, 0, num_timesteps-1)

    dp_size = mpu.get_data_parallel_world_size()

    global iter_index
    if mpu.is_pipeline_first_stage():
        iter_index = iter_index + 1

    # x_t = None
    timestep = None
    x_0 = None
    noise = None

    pp_size = mpu.get_pipeline_model_parallel_world_size()

    if mpu.is_pipeline_first_stage():
        torch.manual_seed(1234)
        for i in range(0, iter_index):
            timestep = torch.randint(0, num_timesteps, (micro_bs,),device=torch.cuda.current_device(), dtype = torch.int64)
        # print("timestep_old_000:",timestep)
        timestep = torch.clamp(timestep, 0, num_timesteps-3)
        timestep = timestep.to(dtype = torch.float16)
        print("timestep_old_111:",timestep)
        torch.manual_seed(1234)


        x_0 = x.clone()
        # noise = torch.randn_like(x)

        ########精度对齐使用##########
        torch.manual_seed(1234)
        for i in range(0, iter_index):
            noise = torch.randn_like(x)
        torch.manual_seed(1234)
        ###############################

        noise = noise.to(device=torch.cuda.current_device(), dtype = dtype)
        torch.manual_seed(1234)
        x = scheduler.q_sample(x, timestep, noise=noise)
        torch.manual_seed(1234)
        x_t = x.clone()

    if mpu.get_pipeline_model_parallel_world_size() > 1:
        x, x_t, y, timestep, t0, mask, x_0, noise, t = model(x, timestep, y, x_0, noise, mask)
        output_tensor_wrap = [x, x_t, y, timestep, t0, mask, x_0, noise, t]
    else:
        x = model(x, timestep, y, x_0, noise, mask)
        output_tensor_wrap = [x]

    # timestep = timestep.to(torch.int64)
    # print(" x[0][0]:",x[0][0][0],"\n")
    # print(" x_t[0][0]:",x_t[0][0][0],"\n")
    # print(" x_0[0][0]:",x_0[0][0][0],"\n")
    # print(" noise[0][0]:",noise[0][0][0],"\n")
    # print(" timestep:",timestep,"\n")
    # print(" yyy:",y,"\n")
    # print(" t0t0t0:",t0,"\n")
    

    return output_tensor_wrap, partial(loss_func, x_t, x_0, timestep, noise)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print("args.data_path:",args.data_path[0])
    dataset = DatasetFromCSV(
        args.data_path[0],
        transform=(
            get_transforms_video(cfg['image_size'][0])
            if not cfg['use_image_transform']
            else get_transforms_image(cfg['image_size'][0])
        ),
        num_frames=cfg['num_frames'],
        frame_interval=cfg['frame_interval'],
        root=cfg['root'],
    )

    dataloader = prepare_dataloader(
        dataset,
        batch_size=args.micro_batch_size,
        num_workers=cfg['num_workers'],
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        process_group=mpu.get_data_parallel_group(),
    )
    dataloader.sampler.set_start_index(0)
    # dataloader.sampler.set_start_index(sampler_start_idx)
    # dataloader.sampler.set_epoch(epoch)
    # dataloader_iter = iter(dataloader)

    return iter(dataloader), None, None


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'dataloader_type': 'external',
            'init_func': initialize}
    )