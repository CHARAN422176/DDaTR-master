import os
import torch
import time
import numpy as np
from torch import nn
from transformers import BertTokenizer
from models.blip import blip_decoder
from modules import utils


def parse_agrs():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--gen_max_len', type=int, default=150)
    parser.add_argument('--load_pretrained', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def main():
    args = parse_agrs()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_tokens(['[BLA]', '[POS]', '[NEG]', '[UNC]'])

    # Dummy prompt
    labels_temp = ['[BLA]'] * 14
    prompt_temp = ' '.join(labels_temp) + ' '

    # Model
    model = blip_decoder(args, tokenizer, image_size=args.image_size, prompt=prompt_temp)
    if args.load_pretrained:
        state_dict = torch.load(args.load_pretrained, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {args.load_pretrained}")

    model = model.to(device)
    model.eval()

    # Dummy inputs
    batch_size = 1
    seq_len = args.gen_max_len

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

    # Warm-up
    for _ in range(5):
        _ = model(
            dummy_image, dummy_context_image,
            dummy_caption, dummy_cls_labels,
            dummy_context_cls_labels, dummy_context_ids,
            dummy_context_segids, dummy_context_attmasks,
            dummy_has_progress, criterion_cls, dummy_base_probs
        )

    # Measure inference time
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(20):
        _ = model(
            dummy_image, dummy_context_image,
            dummy_caption, dummy_cls_labels,
            dummy_context_cls_labels, dummy_context_ids,
            dummy_context_segids, dummy_context_attmasks,
            dummy_has_progress, criterion_cls, dummy_base_probs
        )

    torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / 20
    print(f"Average inference time per sample: {avg_time:.6f} seconds")


if __name__ == '__main__':
    main()
