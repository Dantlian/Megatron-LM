# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain VIT"""
import copy
import torch
from copy import deepcopy
from functools import partial

import mindspeed.megatron_adaptor
import mindspeed
from megatron import core
from megatron.training import get_args
from megatron.core.enums import ModelType
from megatron.core import mpu
from megatron.training import pretrain
from megatron.training.utils import average_losses_across_data_parallel_group
from opensora.utils.misc import to_torch_dtype, requires_grad
from opensora.datasets import DatasetFromCSV, get_transforms_video, get_transforms_image, prepare_dataloader
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.train_utils import update_ema
from mmengine.config import Config

scheduler = None
vae = None
text_encoder = None
cfg = {}


cp_param_name = 'pos_embed_temporal'


def initialize():
    def initialize_models():
        global cfg
        global vae
        global text_encoder

        vae_cfg = cfg['vae']
        text_encoder_cfg = cfg['text_encoder']
        if mpu.get_tensor_model_parallel_rank() == 0:
            vae = build_module(vae_cfg, MODELS)
            text_encoder = build_module(text_encoder_cfg, MODELS, device=torch.cuda.current_device())
        torch.distributed.barrier()
        if mpu.get_tensor_model_parallel_rank() != 0:
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

        print("cfg:", cfg)

    initialize_config()
    initialize_scheduler()
    initialize_models()


def initialize_pipeline_tensor_shaped(hidden_size):
    micro_batch_size = cfg['micro_batch_size']
    latent_size = cfg['latent_size']
    text_encoder_maxlen = text_encoder.model_max_length

    setattr(forward_step, 'pipeline_tensor_shapes',
            [(micro_batch_size, text_encoder.output_dim, hidden_size),
             (micro_batch_size, vae.out_channels, *latent_size),
             (micro_batch_size, 1, text_encoder_maxlen, hidden_size), (micro_batch_size),
             (micro_batch_size, hidden_size * 6), (micro_batch_size, text_encoder_maxlen),
             (micro_batch_size, vae.out_channels, *latent_size), (micro_batch_size, vae.out_channels, *latent_size),
             (micro_batch_size, hidden_size)])


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
    args = get_args()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()

    def get_partial_state_dict(init_state_dict, load_state_dict):
        _partial_state_dict = copy.deepcopy(load_state_dict)
        for name, param in init_state_dict.items():
            if name not in load_state_dict:
                assert "name not in load_state_dict"
            if not isinstance(load_state_dict.get(name), torch.Tensor):
                continue
            load_shape = list(load_state_dict.get(name).shape)
            init_shape = list(param.shape)
            seq_dim = 1
            if cp_size > 1 and cp_param_name in name and init_shape[seq_dim] * cp_size == load_shape[seq_dim]:
                _partial_state_dict[name] = torch.chunk(load_state_dict.get(name), cp_size, dim=seq_dim)[
                    mpu.get_context_parallel_rank()]
                # update load_state_dict and load_shape
                load_state_dict[name] = _partial_state_dict[name]
                load_shape = list(load_state_dict.get(name).shape)

            for i in range(len(init_shape)):
                if tp_size > 1 and init_shape[i] * tp_size == load_shape[i]:
                    _partial_state_dict[name] = torch.chunk(load_state_dict.get(name), tp_size, dim=i)[
                        mpu.get_tensor_model_parallel_rank()]
                    break

        return _partial_state_dict

    def get_partial_tensor(load_param_name, load_param, init_param, seq_dim=1):
        if not isinstance(load_param, torch.Tensor) or not isinstance(init_param, torch.Tensor):
            return load_param
        load_shape = list(load_param.shape)
        init_shape = list(init_param.shape)
        _partial_param = load_param
        if cp_size > 1 and cp_param_name in load_param_name and init_shape[seq_dim] * cp_size == load_shape[seq_dim]:
            _partial_param = torch.chunk(load_param, cp_size, dim=seq_dim)[mpu.get_context_parallel_rank()]
            # update load_param and load_shape
            load_param = _partial_param
            load_shape = list(load_param.shape)

        for i in range(len(init_shape)):
            if tp_size > 1 and init_shape[i] * tp_size == load_shape[i]:
                _partial_param = torch.chunk(load_param, tp_size, dim=i)[mpu.get_tensor_model_parallel_rank()]
                break
        return _partial_param

    dtype = to_torch_dtype("bf16")

    latent_size = cfg['latent_size']
    print("latent_size:", latent_size)
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

    pp_size = mpu.get_pipeline_model_parallel_world_size()
    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
    init_state_dict = stdit.state_dict()
    load_state_dict = torch.load(args.load)
    if pp_size <= 1:
        stdit.load_state_dict(get_partial_state_dict(init_state_dict, load_state_dict))
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
    for name, param in load_state_dict.items():
        if name.startswith('blocks'):
            for i in range(0, len(pp_blocks)):
                key_parts = name.split(".")
                if(key_parts[1] == str(pp_blocks[i])):
                    key_parts[1] = str(i)
                    key = '.'.join(key_parts)
                    partial_state_dict[key] = get_partial_tensor(name, param, init_state_dict.get(key))
                    break

        else:
            if name.startswith('pos_embed') or name.startswith('x_embedder') \
                or name.startswith('t_embedder') or name.startswith('t_block') \
                or name.startswith('y_embedder') or name.startswith('pos_embed'):
                if mpu.is_pipeline_first_stage():
                    partial_state_dict[name] = get_partial_tensor(name, param, init_state_dict.get(name))

            if name.startswith('final_layer') or name.startswith('t_embedder') :
                if mpu.is_pipeline_last_stage():
                    partial_state_dict[name] = get_partial_tensor(name, param, init_state_dict.get(name))

    stdit.load_state_dict(partial_state_dict, strict=False)

    # create ema
    model_state_dict = stdit.state_dict()
    ema = build_module(
        cfg['model'],
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        dtype=dtype,
    )
    ema.load_state_dict(model_state_dict)
    ema = ema.to(torch.float32).to(torch.cuda.current_device())
    stdit = stdit.to(torch.cuda.current_device(), dtype)
    requires_grad(ema, False)
    stdit.ema = ema
    stdit.train()
    update_ema(ema, stdit, decay=0, sharded=False)
    ema.eval()
    # model_sharding(ema)


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

    def _preprocessing_without_dp():
        if mpu.get_tensor_model_parallel_rank() == 0:
            if data_iterator is not None:
                batch = next(data_iterator)
            else:
                batch = None
            x = batch['video'].to(torch.cuda.current_device(), dtype)
            # x.shape: [4, 3, 16, 256, 256], [B, C, T, H/P, W/P]
            y = batch['text']

            with torch.no_grad():
                # Prepare visual inputs
                # before vae.encode, x.shape: [4, 3, 16, 256, 256], [B, C, T, H/P, W/P]
                x = vae.encode(x).contiguous()
                # after vae.encode, x.shape: [4, 4, 16, 32, 32]
                # Prepare text inputs
                encoded_text = text_encoder.encode(y)

            y = encoded_text['y'].contiguous()
            # y.shape: [4, 1, 120, 4096]
            mask = encoded_text['mask'].contiguous()
            # mask.shape: [4, 120]

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
            y = torch.empty((micro_batch_size, 1, text_encoder_maxlen, text_encoder_output_dim), dtype=torch.float32,
                            device=torch.cuda.current_device())
            mask = torch.empty((micro_batch_size, text_encoder_maxlen), dtype=torch.int64,
                               device=torch.cuda.current_device())

            _broadcast(x)
            _broadcast(y)
            _broadcast(mask)

        return {
            'x': x,
            'y': y,
            'mask': mask
        }

    def _preprocessing_with_dp():
        tp_cp_size = mindspeed.core.parallel_state.get_tensor_and_context_parallel_world_size()
        tp_cp_rank = mindspeed.core.parallel_state.get_tensor_and_context_parallel_rank()

        if data_iterator is not None:
            batch = next(data_iterator)
        else:
            batch = None
        x = batch['video'].to(torch.cuda.current_device(), dtype)
        # x.shape: [4, 3, 16, 256, 256], [B, C, T, H/P, W/P]
        y = batch['text']
        frame_dim = 2
        num_frames = x.shape[frame_dim]
        frames_per_partition = core.utils.divide(num_frames, tp_cp_size)
        start_frame = tp_cp_rank * frames_per_partition
        end_frame = min(start_frame + frames_per_partition, num_frames)
        x_per_partition = x[:, :, start_frame:end_frame].contiguous()

        with torch.no_grad():
            x_per_partition = vae.encode(x_per_partition).contiguous()
            encoded_text = text_encoder.encode(y)
        y = encoded_text['y'].contiguous()
        # y.shape: [4, 1, 120, 4096]
        mask = encoded_text['mask'].contiguous()
        # mask.shape: [4, 120]

        # Gather results from all GPUs
        gathered_latents_list = [torch.zeros_like(x_per_partition) for _ in range(tp_cp_size)]
        torch.distributed.all_gather(gathered_latents_list, x_per_partition,
                                     group=mindspeed.core.parallel_state.get_tensor_and_context_parallel_group())

        x = torch.cat(gathered_latents_list, dim=frame_dim).contiguous()
        # x.shape:torch.Size([4, 4, 16, 32, 32])
        return {
            'x': x,
            'y': y,
            'mask': mask
        }

    if (args.enable_preprocessing_data_parallelism and
            cfg.get('num_frames') % mindspeed.core.parallel_state.get_tensor_and_context_parallel_world_size() == 0):
        batch = _preprocessing_with_dp()
    else:
        batch = _preprocessing_without_dp()
    return batch


def get_batch(data_iterator):
    """Build the batch."""

    if mpu.is_pipeline_first_stage():
        batch = get_batch_on_this_tp_rank(data_iterator)
        return batch['x'], batch['y'], batch['mask']
    else:
        return None, None, None


def loss_func(x_t, x_0, t, noise, output_tensor):
    loss_dict = scheduler.training_losses(output_tensor, x_t, x_0, t, noise = noise)
    loss = loss_dict["loss"].mean()

    averaged_loss = average_losses_across_data_parallel_group([loss])

    loss = loss.unsqueeze(0)
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

    global iter_index
    if mpu.is_pipeline_first_stage():
        iter_index = iter_index + 1

    timestep = None
    x_0 = None
    noise = None

    if mpu.is_pipeline_first_stage():
        torch.manual_seed(1234)
        for i in range(0, iter_index):
            timestep = torch.randint(0, num_timesteps, (micro_bs,), device=torch.cuda.current_device(),
                                     dtype=torch.int64)
        timestep = torch.clamp(timestep, 0, num_timesteps - 3)
        timestep = timestep.to(dtype=torch.float16)
        torch.manual_seed(1234)

        x_0 = x.clone()

        ######## 精度对齐使用##########
        torch.manual_seed(1234)
        for i in range(0, iter_index):
            noise = torch.randn_like(x)
        torch.manual_seed(1234)
        ###############################

        noise = noise.to(device=torch.cuda.current_device(), dtype=dtype)
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

    return output_tensor_wrap, partial(loss_func, x_t, x_0, timestep, noise)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print("args.data_path:", args.data_path[0])
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
