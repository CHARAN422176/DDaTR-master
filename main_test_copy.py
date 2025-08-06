import os, json
import torch
from torch import nn
import argparse
import numpy as np
from modules.metrics import compute_scores
from modules.tester import Tester
from models.blip import blip_decoder
from dataset import create_dataset_test 
from dataset import create_sampler 
from dataset import create_loader 
from modules import utils
from transformers import BertTokenizer

def parse_agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json', help='the path to the directory containing the data.')
    parser.add_argument('--image_size', type=int, default=224, help='input image size')
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'], help='the dataset to be used.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')
    parser.add_argument('--load_pretrained', type=str, default=None, help='pretrained path if any')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--gen_max_len', type=int, default=150, help='the maximum token length for text generation.')
    parser.add_argument('--gen_min_len', type=int, default=100, help='the minimum token length for text generation.')
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--monitor_metric', type=str, default='ce_f1', help='the metric to be monitored.')
    parser.add_argument('--init_lr', type=float, default=5e-5, help='.')
    parser.add_argument('--min_lr', type=float, default=5e-6, help='.')
    parser.add_argument('--warmup_lr', type=float, default=5e-7, help='.')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='the weight decay.')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='.')
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cls_weight', type=float, default=4, help='Loss weight of classification branch.')
    args = parser.parse_args()
    return args

def main():
    args = parse_agrs()
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(args.seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # --- Tokenizer with all special tokens BEFORE model creation ---
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_tokens(['[BLA]', '[POS]', '[NEG]', '[UNC]'])

    # --- Dataset not needed for profiling, kept for script consistency ---
    print("Creating dataset...")
    test_dataset = create_dataset_test('generation_%s'%args.dataset_name, tokenizer, args)
    print('number of testing samples: %d'%len(test_dataset))
    samplers = [None]
    test_dataloader = create_loader([test_dataset], samplers, batch_size=[args.batch_size], num_workers=[4], is_trains=[False], collate_fns=[None])[0] 

    # --- Build model (for DDATR/PromptMRG, 14 conditions typically used) ---
    labels_temp = ['[BLA]'] * 14
    prompt_temp = ' '.join(labels_temp)+' '
    model = blip_decoder(args, tokenizer, image_size=args.image_size, prompt=prompt_temp)
    if args.load_pretrained:
        state_dict = torch.load(args.load_pretrained, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print("load checkpoint from {}".format(args.load_pretrained))

    # 4. Prepare dummy forward inputs – must match DDATR forward precisely
    batch_size = 1
    img_size = args.image_size
    seq_len = 110  # Typically use gen_max_len or similar

    dummy_image = torch.randn(batch_size, 3, img_size, img_size)            # main image
    dummy_context_image = torch.randn(batch_size, 3, img_size, img_size)    # context image
    dummy_caption = ['sample caption'] * batch_size                         # list of str
    dummy_cls_labels = torch.zeros(batch_size, 14, dtype=torch.long)        # for 14 diseases
    dummy_context_cls_labels = torch.zeros(batch_size, 14, dtype=torch.long)
    dummy_context_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)  # context BERT ids
    dummy_context_segids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    dummy_context_attmasks = torch.ones(batch_size, seq_len, dtype=torch.long)
    dummy_has_progress = torch.ones(batch_size, dtype=torch.bool)           # progress flag, adjust if model expects scalar
    criterion_cls = nn.CrossEntropyLoss()
    dummy_base_probs = np.ones(14, dtype=np.float32)

    # 5. Patch forward for FLOPs/profiling (optional but helps thop)
    orig_forward = model.forward
    def patched_forward(image, context_image, caption, cls_labels, context_cls_labels, context_ids, context_segids, context_attmasks, has_progress, criterion_cls, base_probs):
        with torch.no_grad():
            return orig_forward(image, context_image, caption, cls_labels, context_cls_labels, context_ids, context_segids, context_attmasks, has_progress, criterion_cls, base_probs)
    model.forward = patched_forward

    # 6. Parameter counting
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 7. Profile FLOPs

    from thop import profile, clever_format

    macs, params = profile(
        model,
        inputs=(
            dummy_image,
            dummy_context_image,
            dummy_caption,
            dummy_cls_labels,
            dummy_context_cls_labels,
            dummy_context_ids,
            dummy_context_segids,
            dummy_context_attmasks,
            dummy_has_progress,
            criterion_cls,
            dummy_base_probs
        )
    )
    macs_cf, params_cf = clever_format([macs, params], "%.3f")
    print(f"FLOPs (thop): {macs_cf}")
    print(f"Parameters (thop): {params_cf}")

    model.forward = orig_forward  # Restore if necessary

if __name__ == '__main__':
    main()
