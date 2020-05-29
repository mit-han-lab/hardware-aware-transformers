# HAT: Hardware-Aware Transformers for Efficient Natural Language Processing
# Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan and Song Han
# The 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
# Paper: https://arxiv.org/abs/2005.14187
# Project page: https://hanruiwang.me/project_pages/hat/

import contextlib
import copy
import importlib.util
import math
import os
import sys
import time
import random
import warnings
import collections
import numpy as np

from tqdm import tqdm
from typing import Callable, List
from collections import defaultdict

import torch
import torch.nn.functional as F

from fairseq.modules import gelu, gelu_accurate



def load_ensemble_for_inference(filenames, task, model_arg_overrides=None):
    from fairseq import checkpoint_utils
    deprecation_warning(
        'utils.load_ensemble_for_inference is deprecated. '
        'Please use checkpoint_utils.load_model_ensemble instead.'
    )
    return checkpoint_utils.load_model_ensemble(
        filenames, arg_overrides=model_arg_overrides, task=task,
    )


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {
                key: _apply(value)
                for key, value in x.items()
            }
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):

    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)


INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda: 0)


def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__

    # assign a unique ID to each module instance, so that incremental state is
    # not shared across module instances
    if not hasattr(module_instance, '_fairseq_instance_id'):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._fairseq_instance_id = INCREMENTAL_STATE_INSTANCE_ID[module_name]

    return '{}.{}.{}'.format(module_name, module_instance._fairseq_instance_id, key)


def get_incremental_state(module, incremental_state, key):
    """Helper for getting incremental state for an nn.Module."""
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]


def set_incremental_state(module, incremental_state, key, value):
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value


def load_align_dict(replace_unk):
    if replace_unk is None:
        align_dict = None
    elif isinstance(replace_unk, str) and len(replace_unk) > 0:
        # Load alignment dictionary for unknown word replacement if it was passed as an argument.
        align_dict = {}
        with open(replace_unk, 'r') as f:
            for line in f:
                cols = line.split()
                align_dict[cols[0]] = cols[1]
    else:
        # No alignment dictionary provided but we still want to perform unknown word replacement by copying the
        # original source word.
        align_dict = {}
    return align_dict


def print_embed_overlap(embed_dict, vocab_dict):
    embed_keys = set(embed_dict.keys())
    vocab_keys = set(vocab_dict.symbols)
    overlap = len(embed_keys & vocab_keys)
    print("| Found {}/{} types in embedding file.".format(overlap, len(vocab_dict)))


def parse_embedding(embed_path):
    """Parse embedding text file into a dictionary of word and embedding tensors.

    The first line can have vocabulary size and dimension. The following lines
    should contain word and embedding separated by spaces.

    Example:
        2 5
        the -0.0230 -0.0264  0.0287  0.0171  0.1403
        at -0.0395 -0.1286  0.0275  0.0254 -0.0932
    """
    embed_dict = {}
    with open(embed_path) as f_embed:
        next(f_embed)  # skip header
        for line in f_embed:
            pieces = line.rstrip().split(" ")
            embed_dict[pieces[0]] = torch.Tensor([float(weight) for weight in pieces[1:]])
    return embed_dict


def load_embedding(embed_dict, vocab, embedding):
    for idx in range(len(vocab)):
        token = vocab[idx]
        if token in embed_dict:
            embedding.weight.data[idx] = embed_dict[token]
    return embedding


def replace_unk(hypo_str, src_str, alignment, align_dict, unk):
    from fairseq import tokenizer
    # Tokens are strings here
    hypo_tokens = tokenizer.tokenize_line(hypo_str)
    src_tokens = tokenizer.tokenize_line(src_str) + ['<eos>']
    for i, ht in enumerate(hypo_tokens):
        if ht == unk:
            src_token = src_tokens[alignment[i]]
            # Either take the corresponding value in the aligned dictionary or just copy the original value.
            hypo_tokens[i] = align_dict.get(src_token, src_token)
    return ' '.join(hypo_tokens)


def post_process_prediction(hypo_tokens, src_str, alignment, align_dict, tgt_dict, remove_bpe=None):
    hypo_str = tgt_dict.string(hypo_tokens, remove_bpe)
    if align_dict is not None:
        hypo_str = replace_unk(hypo_str, src_str, alignment, align_dict, tgt_dict.unk_string())
    if align_dict is not None or remove_bpe is not None:
        # Convert back to tokens for evaluating with unk replacement or without BPE
        # Note that the dictionary can be modified inside the method.
        hypo_tokens = tgt_dict.encode_line(hypo_str, add_if_not_exist=True)
    return hypo_tokens, hypo_str, alignment


def make_positions(tensor, padding_idx, onnx_trace=False):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
        torch.cumsum(mask, dim=1).type_as(mask) * mask
    ).long() + padding_idx


def strip_pad(tensor, pad):
    return tensor[tensor.ne(pad)]


def buffered_arange(max):
    if not hasattr(buffered_arange, 'buf'):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


def convert_padding_direction(src_tokens, padding_idx, right_to_left=False, left_to_right=False):
    assert right_to_left ^ left_to_right
    pad_mask = src_tokens.eq(padding_idx)
    if not pad_mask.any():
        # no padding, return early
        return src_tokens
    if left_to_right and not pad_mask[:, 0].any():
        # already right padded
        return src_tokens
    if right_to_left and not pad_mask[:, -1].any():
        # already left padded
        return src_tokens
    max_len = src_tokens.size(1)
    range = buffered_arange(max_len).type_as(src_tokens).expand_as(src_tokens)
    num_pads = pad_mask.long().sum(dim=1, keepdim=True)
    if right_to_left:
        index = torch.remainder(range - num_pads, max_len)
    else:
        index = torch.remainder(range + num_pads, max_len)
    return src_tokens.gather(1, index)


def item(tensor):
    if hasattr(tensor, 'item'):
        return tensor.item()
    if hasattr(tensor, '__getitem__'):
        return tensor[0]
    return tensor


def clip_grad_norm_(tensor, max_norm):
    grad_norm = item(torch.norm(tensor))
    if grad_norm > max_norm > 0:
        clip_coef = max_norm / (grad_norm + 1e-6)
        tensor.mul_(clip_coef)
    return grad_norm


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def resolve_max_positions(*args):
    """Resolve max position constraints from multiple sources."""

    def map_value_update(d1, d2):
        updated_value = copy.deepcopy(d1)
        for key in d2:
            if key not in updated_value:
                updated_value[key] = d2[key]
            else:
                updated_value[key] = min(d1[key], d2[key])
        return updated_value

    def nullsafe_min(l):
        minim = None
        for item in l:
            if minim is None:
                minim = item
            elif item is not None and item < minim:
                minim = item
        return minim

    max_positions = None
    for arg in args:
        if max_positions is None:
            max_positions = arg
        elif arg is not None:
            if isinstance(arg, float) or isinstance(arg, int):
                max_positions = min(max_positions, arg)
            elif isinstance(arg, dict):
                max_positions = map_value_update(max_positions, arg)
            else:
                max_positions = tuple(
                    map(nullsafe_min, zip(max_positions, arg))
                )

    return max_positions


def import_user_module(args):
    module_path = getattr(args, 'user_dir', None)
    if module_path is not None:
        module_path = os.path.abspath(args.user_dir)
        if not os.path.exists(module_path):
            fairseq_rel_path = os.path.join(os.path.dirname(__file__), '..', args.user_dir)
            if os.path.exists(fairseq_rel_path):
                module_path = fairseq_rel_path
        module_parent, module_name = os.path.split(module_path)

        if module_name not in sys.modules:
            sys.path.insert(0, module_parent)
            importlib.import_module(module_name)
            sys.path.pop(0)


def softmax(x, dim, onnx_trace=False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


def log_softmax(x, dim, onnx_trace=False):
    if onnx_trace:
        return F.log_softmax(x.float(), dim=dim)
    else:
        return F.log_softmax(x, dim=dim, dtype=torch.float32)


def get_perplexity(loss):
    try:
        return '{:.2f}'.format(math.pow(2, loss))
    except OverflowError:
        return float('inf')


def deprecation_warning(message, stacklevel=3):
    # don't use DeprecationWarning, since it's ignored by default
    warnings.warn(message, stacklevel=stacklevel)


def get_activation_fn(activation: str) -> Callable:
    """ Returns the activation function corresponding to `activation` """
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return gelu
    elif activation == 'gelu_fast':
        deprecation_warning('--activation-fn=gelu_fast has been renamed to gelu_accurate')
        return gelu_accurate
    elif activation == 'gelu_accurate':
        return gelu_accurate
    elif activation == 'tanh':
        return torch.tanh
    elif activation == 'linear':
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


def get_available_activation_fns() -> List:
    return [
        'relu',
        'gelu',
        'gelu_fast',  # deprecated
        'gelu_accurate',
        'tanh',
        'linear',
    ]

def handle_save_path(args):
    if args.save_dir is None:
        dirname = args.configs[len('configs' + os.sep):-len('.yml')]
        args.save_dir = os.path.join('checkpoints', dirname)
    if args.tensorboard_logdir is None or args.tensorboard_logdir == '':
        os.makedirs(args.save_dir, exist_ok=True)
        args.tensorboard_logdir = os.path.join(args.save_dir, 'tensorboard')

@contextlib.contextmanager
def eval(model):
    is_training = model.training
    model.eval()
    yield
    model.train(is_training)


def has_parameters(module):
    try:
        next(module.parameters())
        return True
    except StopIteration:
        return False


def sample_configs(choices, reset_rand_seed, rand_seed=0, super_decoder_num_layer=6):
    if reset_rand_seed:
        random.seed(rand_seed)
    config = {
        'encoder': {},
        'decoder': {}
    }

    direct_select = ['embed_dim',
                    'layer_num']
    for v in direct_select:
        for part in ['encoder', 'decoder']:
            config[part][part+'_'+v] = random.choice(choices[part][part+'_'+v])

    # encoder
    encoder_ffn_embed_dim = []
    encoder_self_attention_heads = []
    for _ in range(config['encoder']['encoder_layer_num']):
        encoder_ffn_embed_dim.append(random.choice(choices['encoder']['encoder_ffn_embed_dim']))
        encoder_self_attention_heads.append(random.choice(choices['encoder']['encoder_self_attention_heads']))

    config['encoder']['encoder_ffn_embed_dim'] = encoder_ffn_embed_dim
    config['encoder']['encoder_self_attention_heads'] = encoder_self_attention_heads


    decoder_ffn_embed_dim = []
    decoder_self_attention_heads = []
    decoder_ende_attention_heads = []
    for _ in range(config['decoder']['decoder_layer_num']):
        decoder_ffn_embed_dim.append(random.choice(choices['decoder']['decoder_ffn_embed_dim']))
        decoder_self_attention_heads.append(random.choice(choices['decoder']['decoder_self_attention_heads']))
        decoder_ende_attention_heads.append(random.choice(choices['decoder']['decoder_ende_attention_heads']))
    decoder_arbitrary_ende_attn_all = []

    # every decoder layer need arbitrary_ende_attn setting, even if the layer will not be used
    for _ in range(super_decoder_num_layer):
        decoder_arbitrary_ende_attn_all.append(random.choice(choices['decoder']['decoder_arbitrary_ende_attn']))

    config['decoder']['decoder_ffn_embed_dim'] = decoder_ffn_embed_dim
    config['decoder']['decoder_self_attention_heads'] = decoder_self_attention_heads
    config['decoder']['decoder_ende_attention_heads'] = decoder_ende_attention_heads
    config['decoder']['decoder_arbitrary_ende_attn'] = decoder_arbitrary_ende_attn_all



    return config


def get_subtransformer_config(args):

    config = {
        'encoder': {
            'encoder_embed_dim': args.encoder_embed_dim_subtransformer,
            'encoder_layer_num': args.encoder_layer_num_subtransformer,
            'encoder_ffn_embed_dim': args.encoder_ffn_embed_dim_all_subtransformer,
            'encoder_self_attention_heads': args.encoder_self_attention_heads_all_subtransformer,
        },
        'decoder': {
            'decoder_embed_dim': args.decoder_embed_dim_subtransformer,
            'decoder_layer_num': args.decoder_layer_num_subtransformer,
            'decoder_ffn_embed_dim': args.decoder_ffn_embed_dim_all_subtransformer,
            'decoder_self_attention_heads': args.decoder_self_attention_heads_all_subtransformer,
            'decoder_ende_attention_heads': args.decoder_ende_attention_heads_all_subtransformer,
            'decoder_arbitrary_ende_attn': args.decoder_arbitrary_ende_attn_all_subtransformer
        }
    }

    return config



def get_all_choices(args):

    all_choices = {
        'encoder': {
            'encoder_embed_dim': args.encoder_embed_choice,
            'encoder_layer_num': args.encoder_layer_num_choice,
            'encoder_ffn_embed_dim': args.encoder_ffn_embed_dim_choice,
            'encoder_self_attention_heads': args.encoder_self_attention_heads_choice,

        },
        'decoder': {
            'decoder_embed_dim': args.decoder_embed_choice,
            'decoder_layer_num': args.decoder_layer_num_choice,
            'decoder_ffn_embed_dim': args.decoder_ffn_embed_dim_choice,
            'decoder_self_attention_heads': args.decoder_self_attention_heads_choice,
            'decoder_ende_attention_heads': args.decoder_ende_attention_heads_choice,
            'decoder_arbitrary_ende_attn': args.decoder_arbitrary_ende_attn_choice
        }
    }

    return all_choices

def get_feature_info():
    return ['encoder_embed_dim', 'encoder_layer_num', 'encoder_ffn_embed_dim_avg', 'encoder_self_attention_heads_avg', 'decoder_embed_dim', 'decoder_layer_num', 'decoder_ffn_embed_dim_avg', 'decoder_self_attention_heads_avg', 'decoder_ende_attention_heads_avg', 'decoder_arbitrary_ende_attn_avg']

def get_config_features(config):

    features = []

    features.append(config['encoder']['encoder_embed_dim'])

    encoder_layer_num = config['encoder']['encoder_layer_num']
    features.append(encoder_layer_num)

    encoder_ffn_embed_dim_mean = np.mean(config['encoder']['encoder_ffn_embed_dim'][:encoder_layer_num])
    features.append(encoder_ffn_embed_dim_mean)

    encoder_self_attention_heads_mean = np.mean(config['encoder']['encoder_self_attention_heads'][:encoder_layer_num])
    features.append(encoder_self_attention_heads_mean)


    features.append(config['decoder']['decoder_embed_dim'])

    decoder_layer_num = config['decoder']['decoder_layer_num']
    features.append(decoder_layer_num)

    decoder_ffn_embed_dim_mean = np.mean(config['decoder']['decoder_ffn_embed_dim'][:decoder_layer_num])
    features.append(decoder_ffn_embed_dim_mean)

    decoder_self_attention_heads_mean = np.mean(config['decoder']['decoder_self_attention_heads'][:decoder_layer_num])
    features.append(decoder_self_attention_heads_mean)

    decoder_ende_attention_heads_mean = np.mean(config['decoder']['decoder_ende_attention_heads'][:decoder_layer_num])
    features.append(decoder_ende_attention_heads_mean)

    arbitrary_ende_attn_trans = []
    for i in range(decoder_layer_num):
        if config['decoder']['decoder_arbitrary_ende_attn'][i] == -1:
            arbitrary_ende_attn_trans.append(1)
        elif config['decoder']['decoder_arbitrary_ende_attn'][i] == 1:
            arbitrary_ende_attn_trans.append(2)
        elif config['decoder']['decoder_arbitrary_ende_attn'][i] == 2:
            arbitrary_ende_attn_trans.append(3)

    features.append(np.mean(arbitrary_ende_attn_trans))

    return features


def get_represent_configs(args):
    # largest Subtransformer
    largest_arbitrary1 = {
        'encoder': {
            'encoder_embed_dim': max(args.encoder_embed_choice),
            'encoder_layer_num': max(args.encoder_layer_num_choice),
            'encoder_ffn_embed_dim': [max(args.encoder_ffn_embed_dim_choice)] * args.encoder_layers,
            'encoder_self_attention_heads': [max(args.encoder_self_attention_heads_choice)] * args.encoder_layers,

        },
        'decoder': {
            'decoder_embed_dim': max(args.decoder_embed_choice),
            'decoder_layer_num': max(args.decoder_layer_num_choice),
            'decoder_ffn_embed_dim': [max(args.decoder_ffn_embed_dim_choice)] * args.decoder_layers,
            'decoder_self_attention_heads': [max(args.decoder_self_attention_heads_choice)] * args.decoder_layers,
            'decoder_ende_attention_heads': [max(args.decoder_ende_attention_heads_choice)] * args.decoder_layers,
            'decoder_arbitrary_ende_attn': [1] * args.decoder_layers
        }
    }

    # smallest Subtransformer
    smallest_arbitrary1 = {
        'encoder': {
            'encoder_embed_dim': min(args.encoder_embed_choice),
            'encoder_layer_num': min(args.encoder_layer_num_choice),
            'encoder_ffn_embed_dim': [min(args.encoder_ffn_embed_dim_choice)] * args.encoder_layers,
            'encoder_self_attention_heads': [min(args.encoder_self_attention_heads_choice)] * args.encoder_layers,

        },
        'decoder': {
            'decoder_embed_dim': min(args.decoder_embed_choice),
            'decoder_layer_num': min(args.decoder_layer_num_choice),
            'decoder_ffn_embed_dim': [min(args.decoder_ffn_embed_dim_choice)] * args.decoder_layers,
            'decoder_self_attention_heads': [min(args.decoder_self_attention_heads_choice)] * args.decoder_layers,
            'decoder_ende_attention_heads': [min(args.decoder_ende_attention_heads_choice)] * args.decoder_layers,
            'decoder_arbitrary_ende_attn': [1] * args.decoder_layers
        }
    }

    if args.train_subtransformer:
        subtransformer = get_subtransformer_config(args)
        return {'subtransformer': subtransformer}

    else:
        return {'largest_arbitrary1': largest_arbitrary1, 'smallest_arbitrary1': smallest_arbitrary1}


def measure_latency(args, model, dummy_src_tokens, dummy_prev):
    # latency measurement
    assert not (args.latcpu and args.latgpu)

    model_test = copy.copy(model)
    model_test.set_sample_config(get_subtransformer_config(args))
    src_tokens_test = torch.tensor([dummy_src_tokens], dtype=torch.long)
    src_lengths_test = torch.tensor([30])
    prev_output_tokens_test_with_beam = torch.tensor([dummy_prev] * args.beam, dtype=torch.long)

    if args.latcpu:
        model_test.cpu()
        print('| Measuring model latency on CPU...')
    elif args.latgpu:
        # model_test.cuda()
        src_tokens_test = src_tokens_test.cuda()
        src_lengths_test = src_lengths_test.cuda()
        prev_output_tokens_test_with_beam = prev_output_tokens_test_with_beam.cuda()
        src_tokens_test.get_device()
        print('| Measuring model latency on GPU...')
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

    # dry runs
    for _ in range(5):
        encoder_out_test = model_test.encoder(src_tokens=src_tokens_test, src_lengths=src_lengths_test)

    encoder_latencies = []
    print('| Measuring encoder...')
    for _ in tqdm(range(args.latiter)):
        if args.latgpu:
            start.record()
        elif args.latcpu:
            start = time.time()

        model_test.encoder(src_tokens=src_tokens_test, src_lengths=src_lengths_test)

        if args.latgpu:
            end.record()
            torch.cuda.synchronize()
            encoder_latencies.append(start.elapsed_time(end))
            if not args.latsilent:
                print('| Encoder one run on GPU: ', start.elapsed_time(end))

        elif args.latcpu:
            end = time.time()
            encoder_latencies.append((end - start) * 1000)
            if not args.latsilent:
                print('| Encoder one run on CPU: ', (end - start) * 1000)

    # only use the 10% to 90% latencies to avoid outliers
    print(f'| Encoder latencies: {encoder_latencies}')
    encoder_latencies.sort()
    encoder_latencies = encoder_latencies[int(args.latiter * 0.1): -int(args.latiter * 0.1)]
    print(f'| Encoder latency: Mean: {np.mean(encoder_latencies)} ms; \t Std: {np.std(encoder_latencies)} ms')

    # beam to the batch dimension
    # encoder_out_test_with_beam = encoder_out_test.repeat(1, args.beam)
    bsz = 1
    new_order = torch.arange(bsz).view(-1, 1).repeat(1, args.beam).view(-1).long()
    if args.latgpu:
        new_order = new_order.cuda()

    encoder_out_test_with_beam = model_test.encoder.reorder_encoder_out(encoder_out_test, new_order)

    # dry runs
    for _ in range(5):
        model_test.decoder(prev_output_tokens=prev_output_tokens_test_with_beam,
                           encoder_out=encoder_out_test_with_beam)

    # decoder is more complicated because we need to deal with incremental states and auto regressive things
    decoder_iterations_dict = {'iwslt': 23, 'wmt': 30}
    if 'iwslt' in args.arch:
        decoder_iterations = decoder_iterations_dict['iwslt']
    elif 'wmt' in args.arch:
        decoder_iterations = decoder_iterations_dict['wmt']

    decoder_latencies = []
    print('| Measuring decoder...')
    for _ in tqdm(range(args.latiter)):
        if args.latgpu:
            start.record()
        elif args.latcpu:
            start = time.time()
        incre_states = {}
        for k_regressive in range(decoder_iterations):
            model_test.decoder(prev_output_tokens=prev_output_tokens_test_with_beam[:, :k_regressive + 1],
                               encoder_out=encoder_out_test_with_beam, incremental_state=incre_states)
        if args.latgpu:
            end.record()
            torch.cuda.synchronize()
            decoder_latencies.append(start.elapsed_time(end))
            if not args.latsilent:
                print('| Decoder one run on GPU: ', start.elapsed_time(end))

        elif args.latcpu:
            end = time.time()
            decoder_latencies.append((end - start) * 1000)
            if not args.latsilent:
                print('| Decoder one run on CPU: ', (end - start) * 1000)

    # only use the 10% to 90% latencies to avoid outliers
    decoder_latencies.sort()
    decoder_latencies = decoder_latencies[int(args.latiter * 0.1): -int(args.latiter * 0.1)]

    print(f'| Decoder latencies: {decoder_latencies}')
    print(f'| Decoder latency: Mean: {np.mean(decoder_latencies)} ms; \t Std: {np.std(decoder_latencies)} ms\n')

    print(f"| Overall Latency: {np.mean(encoder_latencies) + np.mean(decoder_latencies)}")


def log_arch_info(stats, config:dict):
    for k, v in config.items():
        for kk, vv in v.items():
            if kk not in ['encoder_ffn_embed_dim', 'decoder_ffn_embed_dim', 'encoder_self_attention_heads', 'decoder_self_attention_heads', 'decoder_ende_attention_heads', 'decoder_arbitrary_ende_attn']:
                stats[kk] = vv
            elif kk in ['encoder_ffn_embed_dim', 'decoder_ffn_embed_dim', 'encoder_self_attention_heads', 'decoder_self_attention_heads', 'decoder_ende_attention_heads', 'decoder_arbitrary_ende_attn']:
                for kkk in range(len(vv)):
                    stats[kk + '_' + str(kkk)] = vv[kkk]

    return stats


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('train_loss')
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('train_loss')
    stats['ppl'] = get_perplexity(nll_loss.avg)
    stats['wps'] = trainer.get_meter('wps')
    stats['ups'] = trainer.get_meter('ups')
    stats['wpb'] = trainer.get_meter('wpb')
    stats['bsz'] = trainer.get_meter('bsz')
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = trainer.get_meter('gnorm')
    stats['clip'] = trainer.get_meter('clip')
    stats['oom'] = trainer.get_meter('oom')
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = trainer.get_meter('loss_scale')
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = trainer.get_meter('train_wall')
    return stats


def get_valid_stats(trainer, args, extra_meters=None):
    from fairseq import checkpoint_utils
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('valid_loss')
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = stats['loss']
    stats['ppl'] = get_perplexity(nll_loss.avg)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        key = 'best_{0}'.format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min

        current_metric = None
        if args.best_checkpoint_metric == 'loss':
            current_metric = stats['loss'].avg
        elif args.best_checkpoint_metric in extra_meters:
            current_metric = extra_meters[args.best_checkpoint_metric].avg
        elif args.best_checkpoint_metric in stats:
            current_metric = stats[args.best_checkpoint_metric]
        else:
            raise ValueError("best_checkpoint_metric not found in logs")

        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            current_metric,
        )
    return stats
