# From Github: https://github.com/RayeRen/multilingual-kd-pytorch/blob/a369a3edb08e255ba024cf76b00cc5a8d057bd2c/fairseq/fed_utils.py
import glob
import hashlib
import os
import torch
from tqdm import tqdm
from fairseq import utils, distributed_utils
import numpy as np
import ujson as json

from fairseq.data.indexed_dataset import IndexedDatasetBuilder, IndexedCachedDataset

FED_VERSION_FN = 'fed_version.v3.idx'


def dist2topk(out_dist, k):
    topk_prob, topk_idx = torch.topk(out_dist, k, dim=-1)
    topk_prob = topk_prob.view(-1, k)  # (B x T) x k
    topk_prob = topk_prob / topk_prob.sum(1, keepdim=True)
    topk_idx = topk_idx.view(-1, k)  # (B x T) x k
    return topk_idx, topk_prob


def output2topk(output, k):
    topk_outp, topk_idx = torch.topk(output, k, dim=-1)
    topk_outp = topk_outp.view(-1, k)  # (B x T) x k
    topk_idx = topk_idx.view(-1, k)  # (B x T) x k
    return topk_idx, topk_outp


def get_sample_key(ids):
    if not hasattr(get_sample_key, 'sample_key_cache'):
        get_sample_key.sample_key_cache = {}
    ids_str = ','.join([str(id) for id in sorted(ids)])
    if ids_str not in get_sample_key.sample_key_cache:
        hash_object = hashlib.md5(ids_str.encode())
        get_sample_key.sample_key_cache[ids_str] = hash_object.hexdigest()
    return get_sample_key.sample_key_cache[ids_str]


class TeacherOutputDatasetBuilder(IndexedDatasetBuilder):
    def add_item(self, data):
        # +1 for Lua compatibility
        data = np.array(data, dtype=self.dtype)
        bytes = self.out_file.write(data)
        self.data_offsets.append(self.data_offsets[-1] + bytes / self.element_size)
        for s in data.shape:
            self.sizes.append(s)
        self.dim_offsets.append(self.dim_offsets[-1] + len(data.shape))


class TeacherOutputDataset(IndexedCachedDataset):
    dtype2size = {
        float: 8,
        int: 4,
    }

    def __init__(self, prefix):
        self.cache_index = {}
        super().__init__(prefix, fix_lua_indexing=False)

    @staticmethod
    def save_bin(prefix, data_list, dtype=np.float):
        bin_path = prefix + '.bin'
        idx_path = prefix + '.idx'
        builder = TeacherOutputDatasetBuilder(bin_path, dtype)
        for d in data_list:
            builder.add_item(d)
        builder.finalize(idx_path)

    @staticmethod
    def get_builder(prefix, data_list, dtype=np.float):
        bin_path = prefix + '.bin'
        idx_path = prefix + '.idx'
        builder = TeacherOutputDatasetBuilder(bin_path, dtype)
        return builder, idx_path

    def __getitem__(self, i):
        self.check_index(i)
        tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
        a = np.empty(tensor_size, dtype=self.dtype)

        if i in self.cache:
            np.copyto(a, self.cache[i])
        else:
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            self.data_file.readinto(a)
            self.cache[i] = a

        item = torch.from_numpy(a)
        if self.dtype == np.int32 or self.dtype == np.int or self.dtype == np.int64:
            item = item.long()
        else:
            item = item.float()
        return item


def gen_attn(args, task, trainer):
    trainer.model.eval()
    itr = task.get_batch_iterator(
        dataset=task.dataset('train'),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences_valid,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            trainer.get_model().max_positions(),
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=8,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
    ).next_epoch_itr(shuffle=False)

    outputs = {'output': [None for _ in range(len(task.dataset('train')))], 'attn': [None for _ in range(len(task.dataset('train')))]}
    save_interval = 0
    for sample in tqdm(itr, mininterval=5):
        with torch.no_grad():
            if sample is None or len(sample) == 0:
                continue
            sample = utils.move_to_cuda(sample)

            bs, srclen = sample['net_input']['src_tokens'].shape
            trainer.model.make_generation_fast_(need_attn=True)
            output, attns = trainer.model(**sample['net_input'])
            non_padding_mask = sample['target'].ne(task.target_dictionary.pad()).cpu()
            _, tgtlen = sample['target'].shape

            # for output
            topk_idx, topk_v = output2topk(output, args.distill_topk)
            topk_x_shape = (bs, tgtlen, args.distill_topk)
            topk_idx, topk_v = topk_idx.view(*topk_x_shape).cpu().numpy(), topk_v.view(*topk_x_shape).cpu().numpy()
            non_padding_mask = non_padding_mask.view(*topk_x_shape[:2]).cpu().numpy().astype(bool)
            for b in range(bs):
                outputs['output'][sample['id'][b].item()] = \
                    topk_idx[b, non_padding_mask[b]].tolist(), \
                    topk_v[b, non_padding_mask[b]].tolist()
            
            for b in range(bs):
                outputs['attn'][sample['id'][b].item()] = [attn[b].cpu().numpy().tolist() for attn in attns['attn']]
            save_interval += 1
            # if save_interval == 1000:

    return outputs


