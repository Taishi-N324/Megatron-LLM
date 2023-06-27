# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Run multi-stage dialogue prompting (MSDP)."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(__file__), os.path.pardir), os.path.pardir)))
from megatron import get_args
import megatron.initialize


def get_tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='tasks')

    # parameters for the knowledgeable dialogue generation
    group.add_argument('--task', type=str, required=True,
                       help='Task name.')
    group.add_argument("--sample_input_file", type=str, default=None,
                       help='Get input from file instead of interactive mode, '
                       'each line is an input.')
    group.add_argument("--sample_output_file", type=str, default=None,
                       help='Output file got from --sample_input_file')
    group.add_argument('--prompt_file', type=str, default=None,
                       help='prompting file')
    group.add_argument('--prompt_type', type=str, default=None,
                       choices=['knowledge', 'response'],
                       help='prompt type (knowledge or response)')
    group.add_argument('--num_prompt_examples', type=int, default=10,
                       help='number of prompt examples')
    group.add_argument('--guess_file', type=str, default=None,
                       help='datapath for generated sentences')
    group.add_argument('--answer_file', type=str, default=None,
                       help='datapath for golden sentences')
    group.add_argument('--out_seq_length', type=int, default=100,
                       help='output sequence length')
    group.add_argument('--api_prompt', default=False, action="store_true",
                       help='setup model api for prompting')
    group.add_argument('--megatron_api_url', type=str, default=None,
                       help='url of the megatron api')
    return parser


if __name__ == '__main__':
    megatron.initialize.initialize_megatron(extra_args_provider=get_tasks_args)
    args = get_args()

    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for downstream tasks.")
        exit()

    if args.task == 'MSDP-PROMPT':
        from tasks.msdp.prompt import main

    elif args.task == 'MSDP-EVAL-F1':
        from tasks.msdp.evaluate import main

    else:
        raise NotImplementedError('Task {} is not implemented.'.format(
            args.task))

    main()
