num_frames = 16
frame_interval = 3
image_size = (256, 256)

# Define dataset
root = "/home/l00618052/ModelZoo-PyTorch/PyTorch/built-in/mlm/OpenSora1.0/MSRVTT/videos/all"
data_path = "CSV_PATH"
use_image_transform = False
num_workers = 1

# Define model
model = dict(
    type="STDiT-XL/2",
    space_scale=0.5,
    time_scale=1.0,
    from_pretrained="/home/l00618052/ModelZoo-PyTorch/PyTorch/built-in/mlm/OpenSora1.0/pretrained_models/PixArt-XL-2-512x512.pth",
    enable_flashattn=False,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="/home/l00618052/ModelZoo-PyTorch/PyTorch/built-in/mlm/OpenSora1.0/sd-vae-ft-ema",
)
text_encoder = dict(
    type="t5",
    from_pretrained="/home/l00618052/ModelZoo-PyTorch/PyTorch/built-in/mlm/OpenSora1.0/DeepFloyd/t5-v1_1-xxl",
    model_max_length=120,
    shardformer=False,  # todo True
)
scheduler = dict(
    type="iddpm",
    timestep_respacing="",
)

# Others
seed = 42
outputs = "outputs"
wandb = False

epochs = 1
log_every = 1
ckpt_every = 10000
load = None
