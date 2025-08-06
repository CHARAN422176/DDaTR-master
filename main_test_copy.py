import os
import torch
import numpy as np
from torch import nn
from transformers import BertTokenizer
from thop import profile, clever_format
from models.blip import blip_decoder
from modules import utils
from dataset import create_dataset_test, create_loader
from modules.metrics import compute_scores
from modules.tester import Tester


def parse_agrs():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/', help='path to image dir')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json', help='path to annotation')
    parser.add_argument('--image_size', type=int, default=224, help='input image size')
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'], help='dataset name')
    parser.add_argument('--threshold', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--load_pretrained', type=str, default=None, help='checkpoint path')
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--gen_max_len', type=int, default=150)
    parser.add_argument('--gen_min_len', type=int, default=100)
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='results/iu_xray')
    parser.add_argument('--monitor_metric', type=str, default='ce_f1')
    parser.add_argument('--init_lr', type=float, default=5e-5)
    parser.add_argument('--min_lr', type=float, default=5e-6)
    parser.add_argument('--warmup_lr', type=float, default=5e-7)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=9233)
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cls_weight', type=float, default=4)
    return parser.parse_args()


def main():
    args = parse_agrs()

    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # Fix seeds
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Tokenizer (add special tokens *before* model instantiation)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_tokens(['[BLA]', '[POS]', '[NEG]', '[UNC]'])

    print(f"Tokenizer size: {len(tokenizer)}")  # Confirm vocab size

    # Create dataset and loader (optional / as per your script)
    print("Creating dataset...")
    test_dataset = create_dataset_test(f'generation_{args.dataset_name}', tokenizer, args)
    print(f'Number of testing samples: {len(test_dataset)}')
    samplers = [None]
    test_dataloader = create_loader(
        [test_dataset], samplers,
        batch_size=[args.batch_size],
        num_workers=[4],
        is_trains=[False],
        collate_fns=[None]
    )[0]

    # Build model
    labels_temp = ['[BLA]'] * 14
    prompt_temp = ' '.join(labels_temp) + ' '
    model = blip_decoder(args, tokenizer, image_size=args.image_size, prompt=prompt_temp)

    if args.load_pretrained:
        state_dict = torch.load(args.load_pretrained, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {args.load_pretrained}")

    model = model.to(device)
    model.eval()

    # Prepare dummy inputs (match your model's forward signature)
    batch_size = 1
    seq_len = args.gen_max_len  # or 110, adjust if different

    dummy_image = torch.randn(batch_size, 3, args.image_size, args.image_size).to(device)
    dummy_context_image = torch.randn(batch_size, 3, args.image_size, args.image_size).to(device)
    dummy_caption = ['this is a dummy caption'] * batch_size
    dummy_cls_labels = torch.zeros(batch_size, 14, dtype=torch.long).to(device)
    dummy_context_cls_labels = torch.zeros(batch_size, 14, dtype=torch.long).to(device)
    dummy_context_ids = torch.zeros(batch_size, seq_len, dtype=torch.long).to(device)
    dummy_context_segids = torch.zeros(batch_size, seq_len, dtype=torch.long).to(device)
    dummy_context_attmasks = torch.ones(batch_size, seq_len, dtype=torch.long).to(device)
    dummy_has_progress = torch.ones(batch_size, dtype=torch.bool).to(device)
    criterion_cls = nn.CrossEntropyLoss()
    dummy_base_probs = np.ones(14, dtype=np.float32)


    # Patch forward for FLOPs profiling to avoid side effects
    orig_forward = model.forward

    def patched_forward(
        image, context_image,
        caption, cls_labels,
        context_cls_labels, context_ids,
        context_segids, context_attmasks,
        has_progress, criterion_cls,
        base_probs
    ):
        with torch.no_grad():
            return orig_forward(
                image, context_image,
                caption, cls_labels,
                context_cls_labels, context_ids,
                context_segids, context_attmasks,
                has_progress, criterion_cls,
                base_probs
            )

    model.forward = patched_forward

    # Parameter counting
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # FLOPs profiling
    macs, params = profile(
        model,
        inputs=(
            dummy_image, dummy_context_image,
            dummy_caption, dummy_cls_labels,
            dummy_context_cls_labels, dummy_context_ids,
            dummy_context_segids, dummy_context_attmasks,
            dummy_has_progress, criterion_cls, dummy_base_probs
        )
    )
    macs_cf, params_cf = clever_format([macs, params], "%.3f")

    print(f"FLOPs (thop): {macs_cf}")
    print(f"Parameters (thop): {params_cf}")

    # Restore original forward
    model.forward = orig_forward


if __name__ == '__main__':
    main()