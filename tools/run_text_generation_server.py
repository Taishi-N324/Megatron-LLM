# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate GPT"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import torch

import megatron.training
from megatron import print_rank_0
from megatron.core import mpu
from megatron.checkpointing import load_checkpoint
import megatron.initialize
import megatron
from megatron.model import GPTModel
from megatron.text_generation_server import MegatronServer
import megatron.text_generation
from megatron.model import ModelType


def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    print_rank_0('building GPT model ...')
    model = GPTModel(num_tokentypes=0,
                     parallel_output=False,
                     pre_process=pre_process,
                     post_process=post_process)
    return model


def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')
    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=0,
                       help='Top k sampling.')
    group.add_argument("--out_seq_length", type=int, default=1024,
                       help='Size of the output generated text.')
    return parser


if __name__ == "__main__":
    megatron.initialize.initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True})
    args = megatron.get_args()
    padded_vocab_size = args.padded_vocab_size
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    # Set up model and load checkpoint
    model_type = ModelType.encoder_or_decoder
    model = megatron.training.get_model(model_provider, model_type, wrap_with_ddp=False, args=args)

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]
    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        server = MegatronServer(model)
        server.run("0.0.0.0")

    while True:
        choice = torch.cuda.LongTensor(1)
        torch.distributed.broadcast(choice, 0)
        if choice[0].item() == 0:
            try:
                megatron.text_generation.generate_and_post_process(model, args=args)
            except ValueError as ve:
                pass
        elif choice[0].item() == 1:
            try:
                megatron.text_generation.beam_search_and_post_process(model,
                                                                      padded_vocab_size=padded_vocab_size,
                                                                      args=args)
            except ValueError as ve:
                pass
