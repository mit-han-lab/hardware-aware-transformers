# HAT: Hardware-Aware Transformers for Efficient Natural Language Processing
# Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan and Song Han
# The 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
# Paper: https://arxiv.org/abs/2005.14187
# Project page: https://hanruiwang.me/project_pages/hat/

import torch
import random

import numpy as np
import fairseq.utils as utils

from fairseq import progress_bar
from latency_predictor import LatencyPredictor


class Converter(object):
    def __init__(self, args):
        self.args = args
        self.super_encoder_layer_num = args.encoder_layers
        self.super_decoder_layer_num = args.decoder_layers

        self.encoder_embed_choice = args.encoder_embed_choice
        self.decoder_embed_choice = args.decoder_embed_choice

        self.encoder_ffn_embed_dim_choice = args.encoder_ffn_embed_dim_choice
        self.decoder_ffn_embed_dim_choice = args.decoder_ffn_embed_dim_choice

        self.encoder_layer_num_choice = args.encoder_layer_num_choice
        self.decoder_layer_num_choice = args.decoder_layer_num_choice

        self.encoder_self_attention_heads_choice = args.encoder_self_attention_heads_choice
        self.decoder_self_attention_heads_choice = args.decoder_self_attention_heads_choice
        self.decoder_ende_attention_heads_choice = args.decoder_ende_attention_heads_choice

        self.decoder_arbitrary_ende_attn_choice = args.decoder_arbitrary_ende_attn_choice


    def config2gene(self, config):
        gene = []

        sample_encoder_layer_num = config['encoder']['encoder_layer_num']

        gene.append(config['encoder']['encoder_embed_dim'])
        gene.append(sample_encoder_layer_num)

        for i in range(self.super_encoder_layer_num):
            if i < sample_encoder_layer_num:
                gene.append(config['encoder']['encoder_ffn_embed_dim'][i])
            else:
                gene.append(config['encoder']['encoder_ffn_embed_dim'][0])

        for i in range(self.super_encoder_layer_num):
            if i < sample_encoder_layer_num:
                gene.append(config['encoder']['encoder_self_attention_heads'][i])
            else:
                gene.append(config['encoder']['encoder_self_attention_heads'][0])



        sample_decoder_layer_num = config['decoder']['decoder_layer_num']

        gene.append(config['decoder']['decoder_embed_dim'])
        gene.append(sample_decoder_layer_num)

        for i in range(self.super_decoder_layer_num):
            if i < sample_decoder_layer_num:
                gene.append(config['decoder']['decoder_ffn_embed_dim'][i])
            else:
                gene.append(config['decoder']['decoder_ffn_embed_dim'][0])

        for i in range(self.super_decoder_layer_num):
            if i < sample_decoder_layer_num:
                gene.append(config['decoder']['decoder_self_attention_heads'][i])
            else:
                gene.append(config['decoder']['decoder_self_attention_heads'][0])

        for i in range(self.super_decoder_layer_num):
            if i < sample_decoder_layer_num:
                gene.append(config['decoder']['decoder_ende_attention_heads'][i])
            else:
                gene.append(config['decoder']['decoder_ende_attention_heads'][0])


        for i in range(self.super_decoder_layer_num):
            gene.append(config['decoder']['decoder_arbitrary_ende_attn'][i])

        return gene

    def gene2config(self, gene):

        config = {
            'encoder': {
                'encoder_embed_dim': None,
                'encoder_layer_num': None,
                'encoder_ffn_embed_dim': None,
                'encoder_self_attention_heads': None,
            },
            'decoder': {
                'decoder_embed_dim': None,
                'decoder_layer_num': None,
                'decoder_ffn_embed_dim': None,
                'decoder_self_attention_heads': None,
                'decoder_ende_attention_heads': None,
                'decoder_arbitrary_ende_attn': None
            }
        }
        current_index = 0


        config['encoder']['encoder_embed_dim'] = gene[current_index]
        current_index += 1

        config['encoder']['encoder_layer_num'] = gene[current_index]
        current_index += 1

        config['encoder']['encoder_ffn_embed_dim'] = gene[current_index: current_index + self.super_encoder_layer_num]
        current_index += self.super_encoder_layer_num

        config['encoder']['encoder_self_attention_heads'] = gene[current_index: current_index + self.super_encoder_layer_num]
        current_index += self.super_encoder_layer_num


        config['decoder']['decoder_embed_dim'] = gene[current_index]
        current_index += 1

        config['decoder']['decoder_layer_num'] = gene[current_index]
        current_index += 1

        config['decoder']['decoder_ffn_embed_dim'] = gene[current_index: current_index + self.super_decoder_layer_num]
        current_index += self.super_decoder_layer_num

        config['decoder']['decoder_self_attention_heads'] = gene[current_index: current_index + self.super_decoder_layer_num]
        current_index += self.super_decoder_layer_num

        config['decoder']['decoder_ende_attention_heads'] = gene[current_index: current_index + self.super_decoder_layer_num]
        current_index += self.super_decoder_layer_num

        config['decoder']['decoder_arbitrary_ende_attn'] = gene[current_index: current_index + self.super_decoder_layer_num]


        return config


    def get_gene_choice(self):
        gene_choice = []

        gene_choice.append(self.encoder_embed_choice)
        gene_choice.append(self.encoder_layer_num_choice)

        for i in range(self.super_encoder_layer_num):
            gene_choice.append(self.encoder_ffn_embed_dim_choice)

        for i in range(self.super_encoder_layer_num):
            gene_choice.append(self.encoder_self_attention_heads_choice)


        gene_choice.append(self.decoder_embed_choice)
        gene_choice.append(self.decoder_layer_num_choice)

        for i in range(self.super_decoder_layer_num):
            gene_choice.append(self.decoder_ffn_embed_dim_choice)

        for i in range(self.super_decoder_layer_num):
            gene_choice.append(self.decoder_self_attention_heads_choice)

        for i in range(self.super_decoder_layer_num):
            gene_choice.append(self.decoder_ende_attention_heads_choice)

        for i in range(self.super_decoder_layer_num):
            gene_choice.append(self.decoder_arbitrary_ende_attn_choice)


        return gene_choice



class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Evolution(object):
    def __init__(self, args, trainer, task, epoch_iter):
        self.population_size = args.population_size
        self.args = args
        self.parent_size = args.parent_size
        self.mutation_size = args.mutation_size
        self.mutation_prob = args.mutation_prob
        self.crossover_size = args.crossover_size
        assert self.population_size == self.parent_size + self.mutation_size + self.crossover_size
        self.converter = Converter(args)
        self.gene_choice = self.converter.get_gene_choice()
        self.gene_len = len(self.gene_choice)
        self.evo_iter = args.evo_iter
        self.trainer = trainer
        self.task=task
        self.epoch_iter = epoch_iter
        self.latency_predictor = LatencyPredictor(
            feature_norm=args.feature_norm,
            lat_norm=args.lat_norm,
            ckpt_path=args.ckpt_path
        )
        self.latency_predictor.load_ckpt()
        self.latency_constraint = args.latency_constraint

        self.best_config = None


    def run_evo_search(self):
        popu = self.random_sample(self.population_size)

        all_scores_list = []

        for i in range(self.evo_iter):
            print(f"| Start Iteration {i}:")
            popu_scores = self.get_scores(popu)
            print(f"| Iteration {i}, Lowest loss: {min(popu_scores)}")

            sorted_ind = np.array(popu_scores).argsort()[:self.parent_size]

            self.best_config = self.converter.gene2config(popu[sorted_ind[0]])
            print(f"| Config for lowest loss model: {self.best_config}")
            print(f"| Predicted latency for lowest loss model: {self.latency_predictor.predict_lat(self.converter.gene2config(popu[sorted_ind[0]]))}")

            parents_popu = [popu[m] for m in sorted_ind]

            parents_score = [popu_scores[m] for m in sorted_ind]
            all_scores_list.append(parents_score)

            mutate_popu = []

            k = 0
            while k < self.mutation_size:
                mutated_gene = self.mutate(random.choices(parents_popu)[0])
                if self.satisfy_constraints(mutated_gene):
                    mutate_popu.append(mutated_gene)
                    k += 1

            crossover_popu = []

            k = 0
            while k < self.crossover_size:
                crossovered_gene = self.crossover(random.sample(parents_popu, 2))
                if self.satisfy_constraints(crossovered_gene):
                    crossover_popu.append(crossovered_gene)
                    k += 1

            popu = parents_popu + mutate_popu + crossover_popu

        return self.best_config


    def crossover(self, genes):
        crossovered_gene = []
        for i in range(self.gene_len):
            if np.random.uniform() < 0.5:
                crossovered_gene.append(genes[0][i])
            else:
                crossovered_gene.append(genes[1][i])

        return crossovered_gene


    def mutate(self, gene):
        mutated_gene = []
        for i in range(self.gene_len):
            if np.random.uniform() < self.mutation_prob:
                mutated_gene.append(random.choices(self.gene_choice[i])[0])
            else:
                mutated_gene.append(gene[i])

        return mutated_gene


    def get_scores(self, genes):
        configs = []
        for gene in genes:
            configs.append(self.converter.gene2config(gene))

        scores = validate_all(self.args, self.trainer, self.task, self.epoch_iter, configs)

        return scores

    def satisfy_constraints(self, gene):
        satisfy = True

        config = self.converter.gene2config(gene)

        if self.latency_predictor.predict_lat(config) > self.latency_constraint:
            satisfy = False

        return satisfy


    def random_sample(self, sample_num):
        popu = []
        i = 0
        while i < sample_num:
            samp_gene = []
            for k in range(self.gene_len):
                samp_gene.append(random.choices(self.gene_choice[k])[0])

            if self.satisfy_constraints(samp_gene):
                popu.append(samp_gene)
                i += 1

        return popu



def test():
    config = {
        'encoder': {
            'encoder_embed_dim': 512,
            'encoder_layer_num': 4,
            'encoder_ffn_embed_dim': [1024, 1025, 1026, 1027],
            'encoder_self_attention_heads': [4, 5, 6, 7],
        },
        'decoder': {
            'decoder_embed_dim': 512,
            'decoder_layer_num': 5,
            'decoder_ffn_embed_dim': [2048, 2049, 2050, 2051, 2052],
            'decoder_self_attention_heads': [4, 6, 7, 8, 9],
            'decoder_ende_attention_heads': [3, 4, 5, 6, 7],
            'decoder_arbitrary_ende_attn': [1, 2, 3, 4, 5, 6, 7]
        }
    }

    args = Namespace(encoder_layers=6,
                     decoder_layers=7,
                     encoder_embed_choice=[768, 512],
                     decoder_embed_choice=[768, 512],
                     encoder_ffn_embed_dim_choice=[3072, 2048],
                     decoder_ffn_embed_dim_choice=[3072, 2048],
                     encoder_layer_num_choice=[6, 5],
                     decoder_layer_num_choice=[6, 5, 4, 3],
                     encoder_self_attention_heads_choice=[8, 4],
                     decoder_self_attention_heads_choice=[8, 4],
                     decoder_ende_attention_heads_choice=[8],
                     decoder_arbitrary_ende_attn_choice=[1, 2]
                     )



    converter = Converter(args)
    gene_get = converter.config2gene(config)

    print(gene_get)
    print(len(gene_get))

    config_get = converter.gene2config(gene_get)

    print(config_get)

    print(len(converter.get_gene_choice()))
    print(converter.get_gene_choice())


def validate_all(args, trainer, task, epoch_itr, configs):
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_losses = []
        # Initialize data iterator
    def get_itr():
        itr = task.get_batch_iterator(
            dataset=task.dataset('valid'),
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format('valid'),
        )
        return progress


    for config in configs:
        trainer.set_sample_config(config)
        progress = get_itr()

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        valid_cnt = 0
        for sample in progress:
            valid_cnt += 1
            if valid_cnt > args.valid_cnt_max:
                break
            trainer.valid_step(sample)

        valid_losses.append(trainer.get_meter('valid_loss').avg)

    return valid_losses


if __name__=='__main__':
    test()
