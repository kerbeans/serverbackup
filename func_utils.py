import argparse
import torch
from typing import Optional
from huggingface_hub import HfFolder,whoami
import os 
from accelerate.logging import get_logger
import json
logger = get_logger(__name__)


MODEL_NAME='runwayml/stable-diffusion-v1-5'#'model/info_unet++_64_1e-06_20'
#MODEL_NAME="model/overfit_20_ep100"
unet_path=None
output_path='model/baseLine'
learning_rate=1e-6
batch_size=8 #train_batch_size
accumulation=16# lr_accumulation_steps
epoch=40





def save_progress(unet,text_encoder,accelerator, args):
    torch.save(accelerator.unwrap_model(unet), os.path.join(f'{args.output_dir}',"temp_Unet"))
    torch.save(accelerator.unwrap_model(text_encoder), os.path.join(f'{args.output_dir}',"temp_Unet"))

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--CLIPimg_path",
        type=str,
        default=None,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=MODEL_NAME,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--unet_path",type=str,default=None)
    parser.add_argument("--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=output_path,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=batch_size
        , help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=epoch)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=accumulation,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=learning_rate,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=True,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default='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', help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=4, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", 4))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"









def ArgParse():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=4)
    parser.add_argument('--epoch',type=int,default=1)
    parser.add_argument('--pretrained_dir',type=str,required=False,default=MODEL_NAME)
    parser.add_argument('--unet_path',type=str,default=unet_path,required=False)
    parser.add_argument('--output_dir',type=str,default=output_path,required=False)
    parser.add_argument('--device',type=str,default='cuda:0')
    parser.add_argument('--resolution',type=int,default=256)
    parser.add_argument('--from_hub',type=None,default=None)
    parser.add_argument('--learning_rate',type=float,default=1e-5 )
    parser.add_argument('--adamb1',type=float,default=0.9)
    parser.add_argument('--adamb2',type=float,default=0.99)
    parser.add_argument('--lr_warmup_steps',type=float,default=500,required= False)
    parser.add_argument("--lr_scheduler",type=str,default="constant_with_warmup",
        help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'))
    parser.add_argument('--accumulation_steps',type=int,default=1)
    
    args=parser.parse_args()
    return args


def freeze_params(params):
    for param in params:
        param.requires_grad=False
        
def unfreeze_params(params):
    for param in params:
        param.requires_grad=True
        
def model_log(filename,info):
    if not os.path.exists(filename):
        print(f'making new file {filename}')
    with open(filename,'a') as f:
        f.write(json.dumps(info)+'\n')
        f.close()
        

    
