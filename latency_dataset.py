# HAT: Hardware-Aware Transformers for Efficient Natural Language Processing
# Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan and Song Han
# The 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
# Paper: https://arxiv.org/abs/2005.14187
# Project page: https://hanruiwang.me/project_pages/hat/

import torch
import time
import pdb

import numpy as np

from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from tqdm import tqdm


def main(args):
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Print args
    print(args)

    # Setup task
    task = tasks.setup_task(args)

    # Build model
    model = task.build_model(args)
    print(model)

    # specify the length of the dummy input for profile
    # for iwslt, the average length is 23, for wmt, that is 30
    dummy_sentence_length_dict = {'iwslt': 23, 'wmt': 30}
    if 'iwslt' in args.arch:
        dummy_sentence_length = dummy_sentence_length_dict['iwslt']
    elif 'wmt' in args.arch:
        dummy_sentence_length = dummy_sentence_length_dict['wmt']
    else:
        raise NotImplementedError

    dummy_src_tokens = [2] + [7] * (dummy_sentence_length - 1)
    dummy_prev = [7] * (dummy_sentence_length - 1) + [2]

    # for latency predictor: latency dataset generation
    with open(args.lat_dataset_path, 'w') as fid:
        src_tokens_test = torch.tensor([dummy_src_tokens], dtype=torch.long)
        src_lengths_test = torch.tensor([dummy_sentence_length])
        prev_output_tokens_test_with_beam = torch.tensor([dummy_prev] * args.beam, dtype=torch.long)
        if args.latcpu:
            model.cpu()
            print('Measuring model latency on CPU for dataset generation...')
        elif args.latgpu:
            model.cuda()
            src_tokens_test = src_tokens_test.cuda()
            src_lengths_test = src_lengths_test.cuda()
            prev_output_tokens_test_with_beam = prev_output_tokens_test_with_beam.cuda()
            src_tokens_test.get_device()
            print('Measuring model latency on GPU for dataset generation...')
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

        feature_info = utils.get_feature_info()
        fid.write(','.join(feature_info) + ',')
        latency_info = ['latency_mean_encoder', 'latency_mean_decoder', 'latency_std_encoder', 'latency_std_decoder']
        fid.write(','.join(latency_info) + '\n')

        for i in range(args.lat_dataset_size):
            print(i)
            config_sam = utils.sample_configs(utils.get_all_choices(args), reset_rand_seed=False, super_decoder_num_layer=args.decoder_layers)

            features = utils.get_config_features(config_sam)
            fid.write(','.join(map(str, features)) + ',')

            model.set_sample_config(config_sam)

            # dry runs
            for _ in range(5):
                encoder_out_test = model.encoder(src_tokens=src_tokens_test, src_lengths=src_lengths_test)

            encoder_latencies = []
            print('Measuring encoder for dataset generation...')
            for _ in tqdm(range(args.latiter)):
                if args.latgpu:
                    start.record()
                elif args.latcpu:
                    start = time.time()

                model.encoder(src_tokens=src_tokens_test, src_lengths=src_lengths_test)

                if args.latgpu:
                    end.record()
                    torch.cuda.synchronize()
                    encoder_latencies.append(start.elapsed_time(end))
                    if not args.latsilent:
                        print('Encoder one run on GPU (for dataset generation): ', start.elapsed_time(end))

                elif args.latcpu:
                    end = time.time()
                    encoder_latencies.append((end - start) * 1000)
                    if not args.latsilent:
                        print('Encoder one run on CPU (for dataset generation): ', (end - start) * 1000)

            # only use the 10% to 90% latencies to avoid outliers
            encoder_latencies.sort()
            encoder_latencies = encoder_latencies[int(args.latiter * 0.1): -max(1, int(args.latiter * 0.1))]
            print(f'Encoder latency for dataset generation: Mean: {np.mean(encoder_latencies)} ms; \t Std: {np.std(encoder_latencies)} ms')

            bsz = 1
            new_order = torch.arange(bsz).view(-1, 1).repeat(1, args.beam).view(-1).long()
            if args.latgpu:
                new_order = new_order.cuda()

            encoder_out_test_with_beam = model.encoder.reorder_encoder_out(encoder_out_test, new_order)

            # dry runs
            for _ in range(5):
                model.decoder(prev_output_tokens=prev_output_tokens_test_with_beam,
                                   encoder_out=encoder_out_test_with_beam)

            # decoder is more complicated because we need to deal with incremental states and auto regressive things
            decoder_iterations_dict = {'iwslt': 23, 'wmt': 30}
            if 'iwslt' in args.arch:
                decoder_iterations = decoder_iterations_dict['iwslt']
            elif 'wmt' in args.arch:
                decoder_iterations = decoder_iterations_dict['wmt']

            decoder_latencies = []
            print('Measuring decoder for dataset generation...')
            for _ in tqdm(range(args.latiter)):
                if args.latgpu:
                    start.record()
                elif args.latcpu:
                    start = time.time()
                incre_states = {}
                for k_regressive in range(decoder_iterations):
                    model.decoder(prev_output_tokens=prev_output_tokens_test_with_beam[:, :k_regressive + 1],
                                       encoder_out=encoder_out_test_with_beam, incremental_state=incre_states)
                if args.latgpu:
                    end.record()
                    torch.cuda.synchronize()
                    decoder_latencies.append(start.elapsed_time(end))
                    if not args.latsilent:
                        print('Decoder one run on GPU (for dataset generation): ', start.elapsed_time(end))

                elif args.latcpu:
                    end = time.time()
                    decoder_latencies.append((end - start) * 1000)
                    if not args.latsilent:
                        print('Decoder one run on CPU (for dataset generation): ', (end - start) * 1000)

            # only use the 10% to 90% latencies to avoid outliers
            decoder_latencies.sort()
            decoder_latencies = decoder_latencies[int(args.latiter * 0.1): -max(1, int(args.latiter * 0.1))]

            print(decoder_latencies)
            print(f'Decoder latency for dataset generation: Mean: {np.mean(decoder_latencies)} ms; \t Std: {np.std(decoder_latencies)} ms')

            lats = [np.mean(encoder_latencies), np.mean(decoder_latencies), np.std(encoder_latencies), np.std(decoder_latencies)]
            fid.write(','.join(map(str, lats)) + '\n')

def cli_main():
    parser = options.get_training_parser()

    parser.add_argument('--latgpu', action='store_true', help='measure SubTransformer latency on GPU')
    parser.add_argument('--latcpu', action='store_true', help='measure SubTransformer latency on CPU')
    parser.add_argument('--latiter', type=int, default=300, help='how many iterations to run when measure the latency')
    parser.add_argument('--latsilent', action='store_true', help='keep silent when measure latency')

    parser.add_argument('--lat-dataset-path', type=str, default='./latency_dataset/lat.tmp', help='the path to write latency dataset')
    parser.add_argument('--lat-dataset-size', type=int, default=200, help='number of data points for the dataset')

    options.add_generation_args(parser)

    args = options.parse_args_and_arch(parser)

    if args.latcpu:
        args.cpu = True
        args.fp16 = False

    if args.pdb:
        pdb.set_trace()

    main(args)

if __name__ == '__main__':
    cli_main()
