# HAT: Hardware Aware Transformers for Efficient Natural Language Processing [[paper]](https://arxiv.org/abs/2005.14187) [[website]](https://hat.mit.edu) [[video]](https://youtu.be/N_tH1jIbqCw)


```
@inproceedings{hanruiwang2020hat,
    title     = {HAT: Hardware-Aware Transformers for Efficient Natural Language Processing},
    author    = {Wang, Hanrui and Wu, Zhanghao and Liu, Zhijian and Cai, Han and Zhu, Ligeng and Gan, Chuang and Han, Song},
    booktitle = {Annual Conference of the Association for Computational Linguistics},
    year      = {2020}
} 
```


## Overview
We release the PyTorch code and 50 pre-trained models for HAT: Hardware-Aware Transformers. Within a Transformer supernet (SuperTransformer), we efficiently search for a specialized fast model (SubTransformer) for each hardware with latency feedback. The search cost is reduced by over 10000×.
![teaser](https://hanruiwang.me/project_pages/hat/assets/teaser.jpg)

HAT Framework overview:
![overview](https://hanruiwang.me/project_pages/hat/assets/overview.jpg)

HAT models achieve up to 3× speedup and 3.7× smaller model size with no performance loss.
![results](https://hanruiwang.me/project_pages/hat/assets/results.jpg)


## Usage

### Installation
To install from source and develop locally:

```bash
git clone https://github.com/mit-han-lab/hardware-aware-transformers.git
cd hardware-aware-transformers
pip install --editable .
```

### Data Preparation

| Task | task_name | Train | Valid | Test | 
|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| WMT'14 En-De | wmt14.en-de | [WMT'16](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) | newstest2013 | newstest2014 | 
| WMT'14 En-Fr | wmt14.en-fr | [WMT'14](http://statmt.org/wmt14/translation-task.html#Download) | newstest2012&2013 | newstest2014 | 
| WMT'19 En-De | wmt19.en-de | [WMT'19](http://www.statmt.org/wmt19/translation-task.html#download) | newstest2017 | newstest2018 | 
| IWSLT'14 De-En | iwslt14.de-en | [IWSLT'14 train set](https://wit3.fbk.eu/archive/2014-01/texts/de/en/de-en.tgz) | IWSLT'14 valid set | IWSLT14.TED.dev2010 <br> IWSLT14.TEDX.dev2012 <br> IWSLT14.TED.tst2010 <br> IWSLT14.TED.tst2011 <br> IWSLT14.TED.tst2012 |  

To download and preprocess data, run:
```bash
bash configs/[task_name]/preprocess.sh
```

If you find preprocessing time-consuming, you can directly download the preprocessed data we provide:
```bash
bash configs/[task_name]/get_preprocessed.sh
```


### Testing
We provide pre-trained models (SubTransformers) on the Machine Translation tasks for evaluations. The #Params and FLOPs do not count in the embedding lookup table and the last output layers because they are dependent on tasks.

| Task | Hardware | Latency | #Params<br>(M) | FLOPs<br>(G) | BLEU | Sacre<br>BLEU | model_name | Link |
|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------|:-----------:|:-----------|:-----------:|
| WMT'14 En-De | Raspberry Pi ARM Cortex-A72 CPU | 3.5s <br> 4.0s <br> 4.5s <br> 5.0s <br> 6.0s <br> 6.9s | 25.22 <br> 29.42 <br> 35.72 <br> 36.77 <br> 44.13 <br> 48.33 | 1.53 <br> 1.78 <br> 2.19 <br> 2.26 <br> 2.70 <br> 3.02 | 25.8 <br> 26.9 <br> 27.6 <br> 27.8 <br> 28.2 <br> 28.4 | 25.6 <br> 26.6 <br> 27.1 <br> 27.2 <br> 27.6 <br> 27.8 | HAT_wmt14ende_raspberrypi@<!-- -->3.5s_bleu@<!-- -->25.8 <br> HAT_wmt14ende_raspberrypi@<!-- -->4.0s_bleu@<!-- -->26.9 <br> HAT_wmt14ende_raspberrypi@<!-- -->4.5s_bleu@<!-- -->27.6 <br> HAT_wmt14ende_raspberrypi@<!-- -->5.0s_bleu@<!-- -->27.8 <br> HAT_wmt14ende_raspberrypi@<!-- -->6.0s_bleu@<!-- -->28.2 <br> HAT_wmt14ende_raspberrypi@<!-- -->6.9s_bleu@<!-- -->28.4 | [link](https://www.dropbox.com/s/pmfwwg1d1kmfdh5/HAT_wmt14ende_raspberrypi@3.5s_bleu@25.8.pt?dl=0) <br> [link](https://www.dropbox.com/s/ko0i65k1664p74u/HAT_wmt14ende_raspberrypi@4.0s_bleu@26.9.pt?dl=0) <br> [link](https://www.dropbox.com/s/f4y6u9cbcdykeha/HAT_wmt14ende_raspberrypi@4.5s_bleu@27.6.pt?dl=0) <br> [link](https://www.dropbox.com/s/av5vycafxo57x6w/HAT_wmt14ende_raspberrypi@5.0s_bleu@27.8.pt?dl=0) <br> [link](https://www.dropbox.com/s/ywedqumq91a4ekn/HAT_wmt14ende_raspberrypi@6.0s_bleu@28.2.pt?dl=0) <br> [link](https://www.dropbox.com/s/x7iucaotbeald3q/HAT_wmt14ende_raspberrypi@6.9s_bleu@28.4.pt?dl=0) |
| WMT'14 En-De | Intel Xeon E5-2640 CPU | 137.9ms <br> 204.2ms <br> 278.7ms <br> 340.2ms <br> 369.6ms <br> 450.9ms | 30.47 <br> 35.72 <br> 40.97 <br> 46.23 <br> 51.48 <br> 56.73 | 1.87 <br> 2.19 <br> 2.54 <br> 2.86 <br> 3.21 <br> 3.53 | 25.8 <br> 27.6 <br> 27.9 <br> 28.1 <br> 28.2 <br> 28.5 | 25.6 <br> 27.1 <br> 27.3 <br> 27.5 <br> 27.6 <br> 27.9 | HAT_wmt14ende_xeon@<!-- -->137.9ms_bleu@<!-- -->25.8 <br> HAT_wmt14ende_xeon@<!-- -->204.2ms_bleu@<!-- -->27.6 <br> HAT_wmt14ende_xeon@<!-- -->278.7ms_bleu@<!-- -->27.9 <br> HAT_wmt14ende_xeon@<!-- -->340.2ms_bleu@<!-- -->28.1 <br> HAT_wmt14ende_xeon@<!-- -->369.6ms_bleu@<!-- -->28.2 <br> HAT_wmt14ende_xeon@<!-- -->450.9ms_bleu@<!-- -->28.5 | [link](https://www.dropbox.com/s/bvq3y6igoyxe1t5/HAT_wmt14ende_xeon@137.9ms_bleu@25.8.pt?dl=0) <br> [link](https://www.dropbox.com/s/yg12xz504uw2g1s/HAT_wmt14ende_xeon@204.2ms_bleu@27.6.pt?dl=0) <br> [link](https://www.dropbox.com/s/l5ljas8zyg9ik65/HAT_wmt14ende_xeon@278.7ms_bleu@27.9.pt?dl=0) <br> [link](https://www.dropbox.com/s/fkp61h8jbyt524i/HAT_wmt14ende_xeon@340.2ms_bleu@28.1.pt?dl=0) <br> [link](https://www.dropbox.com/s/3mv3oaddeb132np/HAT_wmt14ende_xeon@369.6ms_bleu@28.2.pt?dl=0) <br> [link](https://www.dropbox.com/s/bjldda9nzj7cpni/HAT_wmt14ende_xeon@450.9ms_bleu@28.5.pt?dl=0) |
| WMT'14 En-De | Nvidia TITAN Xp GPU | 57.1ms <br> 91.2ms <br> 126.0ms <br> 146.7ms <br> 208.1ms | 30.47 <br> 35.72 <br> 40.97 <br> 51.20 <br> 49.38 | 1.87 <br> 2.19 <br> 2.54 <br> 3.17 <br> 3.09 <br> | 25.8 <br> 27.6 <br> 27.9 <br> 28.1 <br> 28.5 | 25.6 <br> 27.1 <br> 27.3 <br> 27.5 <br> 27.8 | HAT_wmt14ende_titanxp@<!-- -->57.1ms_bleu@<!-- -->25.8 <br> HAT_wmt14ende_titanxp@<!-- -->91.2ms_bleu@<!-- -->27.6 <br> HAT_wmt14ende_titanxp@<!-- -->126.0ms_bleu@<!-- -->27.9 <br> HAT_wmt14ende_titanxp@<!-- -->146.7ms_bleu@<!-- -->28.1 <br> HAT_wmt14ende_titanxp@<!-- -->208.1ms_bleu@<!-- -->28.5 | [link](https://www.dropbox.com/s/71w5t0qidsxqe1e/HAT_wmt14ende_titanxp@57.1ms_bleu@25.8.pt?dl=0) <br> [link](https://www.dropbox.com/s/j0hnmxw6xz6tskh/HAT_wmt14ende_titanxp@91.2ms_bleu@27.6.pt?dl=0) <br> [link](https://www.dropbox.com/s/pyetdnbz1zvcfg5/HAT_wmt14ende_titanxp@126.0ms_bleu@27.9.pt?dl=0) <br> [link](https://www.dropbox.com/s/ixn832oai2k44j9/HAT_wmt14ende_titanxp@146.7ms_bleu@28.1.pt?dl=0) <br> [link](https://www.dropbox.com/s/owpdwmqwpn9jw14/HAT_wmt14ende_titanxp@208.1ms_bleu@28.5.pt?dl=0) |
| WMT'14 En-Fr | Raspberry Pi ARM Cortex-A72 CPU | 4.3s <br> 5.3s <br> 5.8s <br> 6.9s <br> 7.8s <br> 9.1s | 25.22 <br> 35.72 <br> 36.77 <br> 44.13 <br> 49.38 <br> 56.73 | 1.53 <br> 2.23 <br> 2.26 <br> 2.70 <br> 3.09 <br> 3.57 | 38.8 <br> 40.1 <br> 40.6 <br> 41.1 <br> 41.4 <br> 41.8 | 36.0 <br> 37.3 <br> 37.8 <br> 38.3 <br> 38.5 <br> 38.9 | HAT_wmt14enfr_raspberrypi@<!-- -->4.3s_bleu@<!-- -->38.8 <br> HAT_wmt14enfr_raspberrypi@<!-- -->5.3s_bleu@<!-- -->40.1 <br> HAT_wmt14enfr_raspberrypi@<!-- -->5.8s_bleu@<!-- -->40.6 <br> HAT_wmt14enfr_raspberrypi@<!-- -->6.9s_bleu@<!-- -->41.1 <br> HAT_wmt14enfr_raspberrypi@<!-- -->7.8s_bleu@<!-- -->41.4 <br> HAT_wmt14enfr_raspberrypi@<!-- -->9.1s_bleu@<!-- -->41.8 | [link](https://www.dropbox.com/s/ku97fwz1oj1a112/HAT_wmt14enfr_raspberrypi@4.3s_bleu@38.8.pt?dl=0) <br> [link](https://www.dropbox.com/s/9noopb605fqmjpl/HAT_wmt14enfr_raspberrypi@5.3s_bleu@40.1.pt?dl=0) <br> [link](https://www.dropbox.com/s/vmdkh0ctpdac7gr/HAT_wmt14enfr_raspberrypi@5.8s_bleu@40.6.pt?dl=0) <br> [link](https://www.dropbox.com/s/dbo9abn5pnb6qgz/HAT_wmt14enfr_raspberrypi@6.9s_bleu@41.1.pt?dl=0) <br> [link](https://www.dropbox.com/s/x8tsbxbwkk64ejg/HAT_wmt14enfr_raspberrypi@7.8s_bleu@41.4.pt?dl=0) <br> [link](https://www.dropbox.com/s/zbsbl5e96y3t5zl/HAT_wmt14enfr_raspberrypi@9.1s_bleu@41.8.pt?dl=0) |
| WMT'14 En-Fr | Intel Xeon E5-2640 CPU | 154.7ms <br> 208.8ms <br> 329.4ms <br> 394.5ms <br> 442.0ms | 30.47 <br> 35.72 <br> 44.13 <br> 51.48 <br> 56.73 | 1.84 <br> 2.23 <br> 2.70 <br> 3.28 <br> 3.57 | 39.1 <br> 40.0 <br> 41.1 <br> 41.4 <br> 41.7 | 36.3 <br> 37.2 <br> 38.2 <br> 38.5 <br> 38.8 | HAT_wmt14enfr_xeon@<!-- -->154.7ms_bleu@<!-- -->39.1 <br> HAT_wmt14enfr_xeon@<!-- -->208.8ms_bleu@<!-- -->40.0 <br> HAT_wmt14enfr_xeon@<!-- -->329.4ms_bleu@<!-- -->41.1 <br> HAT_wmt14enfr_xeon@<!-- -->394.5ms_bleu@<!-- -->41.4 <br> HAT_wmt14enfr_xeon@<!-- -->442.0ms_bleu@<!-- -->41.7 | [link](https://www.dropbox.com/s/6xswl0oesuvmqk5/HAT_wmt14enfr_xeon@154.7ms_bleu@39.1.pt?dl=0) <br> [link](https://www.dropbox.com/s/ot3zt8nenda54j7/HAT_wmt14enfr_xeon@208.8ms_bleu@40.0.pt?dl=0) <br> [link](https://www.dropbox.com/s/epe2lvus4l40v9o/HAT_wmt14enfr_xeon@329.4ms_bleu@41.1.pt?dl=0) <br> [link](https://www.dropbox.com/s/qnt2qzkb3i054c6/HAT_wmt14enfr_xeon@394.5ms_bleu@41.4.pt?dl=0) <br> [link](https://www.dropbox.com/s/79zcolb53jbhchk/HAT_wmt14enfr_xeon@442.0ms_bleu@41.7.pt?dl=0) |
| WMT'14 En-Fr | Nvidia TITAN Xp GPU | 69.3ms <br> 94.9ms <br> 132.9ms <br> 168.3ms <br> 208.3ms | 30.47 <br> 35.72 <br> 40.97 <br> 46.23 <br> 51.48 | 1.84 <br> 2.23 <br> 2.51 <br> 2.90 <br> 3.25 | 39.1 <br> 40.0 <br> 40.7 <br> 41.1 <br> 41.7 | 36.3 <br> 37.2 <br> 37.8 <br> 38.3 <br> 38.8 | HAT_wmt14enfr_titanxp@<!-- -->69.3ms_bleu@<!-- -->39.1 <br> HAT_wmt14enfr_titanxp@<!-- -->94.9ms_bleu@<!-- -->40.0 <br> HAT_wmt14enfr_titanxp@<!-- -->132.9ms_bleu@<!-- -->40.7 <br> HAT_wmt14enfr_titanxp@<!-- -->168.3ms_bleu@<!-- -->41.1 <br> HAT_wmt14enfr_titanxp@<!-- -->208.3ms_bleu@<!-- -->41.7 | [link](https://www.dropbox.com/s/hvy255ls277onjw/HAT_wmt14enfr_titanxp@69.3ms_bleu@39.1.pt?dl=0) <br> [link](https://www.dropbox.com/s/rvfv99jbh4n7qys/HAT_wmt14enfr_titanxp@94.9ms_bleu@40.0.pt?dl=0) <br> [link](https://www.dropbox.com/s/u6u3u40pr4f5mzh/HAT_wmt14enfr_titanxp@132.9ms_bleu@40.7.pt?dl=0) <br> [link](https://www.dropbox.com/s/wlbbmnrl61dx4z7/HAT_wmt14enfr_titanxp@168.3ms_bleu@41.1.pt?dl=0) <br> [link](https://www.dropbox.com/s/e41lnsktn5bb2fz/HAT_wmt14enfr_titanxp@208.3ms_bleu@41.7.pt?dl=0) |
| WMT'19 En-De | Nvidia TITAN Xp GPU | 55.7ms <br> 93.2ms <br> 134.5ms <br> 176.1ms <br> 204.5ms <br> 237.8ms | 36.89 <br> 42.28 <br> 40.97 <br> 46.23 <br> 51.48 <br> 56.73 | 2.27 <br> 2.63 <br> 2.54 <br> 2.86 <br> 3.18 <br> 3.53 | 42.4 <br> 44.4 <br> 45.4 <br> 46.2 <br> 46.5 <br> 46.7 | 41.9 <br> 43.9 <br> 44.7 <br> 45.6 <br> 45.7 <br> 46.0 | HAT_wmt19ende_titanxp@<!-- -->55.7ms_bleu@<!-- -->42.4 <br> HAT_wmt19ende_titanxp@<!-- -->93.2ms_bleu@<!-- -->44.4 <br> HAT_wmt19ende_titanxp@<!-- -->134.5ms_bleu@<!-- -->45.4 <br> HAT_wmt19ende_titanxp@<!-- -->176.1ms_bleu@<!-- -->46.2 <br> HAT_wmt19ende_titanxp@<!-- -->204.5ms_bleu@<!-- -->46.5 <br> HAT_wmt19ende_titanxp@<!-- -->237.8ms_bleu@<!-- -->46.7 | [link](https://www.dropbox.com/s/6pokem8orb75ldh/HAT_wmt19ende_titanxp@55.7ms_bleu@42.4.pt?dl=0) <br> [link](https://www.dropbox.com/s/zgcd70pzf1sle4z/HAT_wmt19ende_titanxp@93.2ms_bleu@44.4.pt?dl=0) <br> [link](https://www.dropbox.com/s/mm827rst6a144zy/HAT_wmt19ende_titanxp@134.5ms_bleu@45.4.pt?dl=0) <br> [link](https://www.dropbox.com/s/y0vov0n9zt50n9c/HAT_wmt19ende_titanxp@176.1ms_bleu@46.2.pt?dl=0) <br> [link](https://www.dropbox.com/s/w1si4mgf1e3l8oj/HAT_wmt19ende_titanxp@204.5ms_bleu@46.5.pt?dl=0) <br> [link](https://www.dropbox.com/s/rljih3t0hglp39a/HAT_wmt19ende_titanxp@237.8ms_bleu@46.7.pt?dl=0) |
| IWSLT'14 De-En | Nvidia TITAN Xp GPU | 45.6ms <br> 74.5ms <br> 109.0ms <br> 137.8ms <br> 168.8ms | 16.82 <br> 19.98 <br> 23.13 <br> 27.33 <br> 31.54 | 0.78 <br> 0.93 <br> 1.13 <br> 1.32 <br> 1.52 | 33.4 <br> 34.2 <br> 34.5 <br> 34.7 <br> 34.8 | 32.5 <br> 33.3 <br> 33.6 <br> 33.8 <br> 33.9 | HAT_iwslt14deen_titanxp@<!-- -->45.6ms_bleu@<!-- -->33.4 <br> HAT_iwslt14deen_titanxp@<!-- -->74.5ms_bleu@<!-- -->34.2 <br> HAT_iwslt14deen_titanxp@<!-- -->109.0ms_bleu@<!-- -->34.5 <br> HAT_iwslt14deen_titanxp@<!-- -->137.8ms_bleu@<!-- -->34.7 <br> HAT_iwslt14deen_titanxp@<!-- -->168.8ms_bleu@<!-- -->34.8 | [link](https://www.dropbox.com/s/ntj1gfskish8vz3/HAT_iwslt14deen_titanxp@45.6ms_bleu@33.4.pt?dl=0) <br> [link](https://www.dropbox.com/s/gjq46181s3xbz0k/HAT_iwslt14deen_titanxp@74.5ms_bleu@34.2.pt?dl=0) <br> [link](https://www.dropbox.com/s/fg3r3tk2vjg0diq/HAT_iwslt14deen_titanxp@109.0ms_bleu@34.5.pt?dl=0) <br> [link](https://www.dropbox.com/s/3j5vu5jh71xwec1/HAT_iwslt14deen_titanxp@137.8ms_bleu@34.7.pt?dl=0) <br> [link](https://www.dropbox.com/s/5xy9hdjuc5c6sw5/HAT_iwslt14deen_titanxp@168.8ms_bleu@34.8.pt?dl=0) |



#### Download models:
```bash
python download_model.py --model-name=[model_name]
# for example
python download_model.py --model-name=HAT_wmt14ende_raspberrypi@3.5s_bleu@25.8
# to download all models
python download_model.py --download-all
```

#### Test BLEU (SacreBLEU) score:
```bash
bash configs/[task_name]/test.sh \
    [model_file] \
    configs/[task_name]/subtransformer/[model_name].yml \
    [normal|sacre]
# for example
bash configs/wmt14.en-de/test.sh \
    ./downloaded_models/HAT_wmt14ende_raspberrypi@3.5s_bleu@25.8.pt \
    configs/wmt14.en-de/subtransformer/HAT_wmt14ende_raspberrypi@3.5s_bleu@25.8.yml \
    normal
# another example
bash configs/iwslt14.de-en/test.sh \
    ./downloaded_models/HAT_iwslt14deen_titanxp@137.8ms_bleu@34.7.pt \
    configs/iwslt14.de-en/subtransformer/HAT_iwslt14deen_titanxp@137.8ms_bleu@34.7.yml \
    sacre
```

#### Test Latency, model size and FLOPs
To profile the latency, model size and FLOPs (FLOPs profiling needs [torchprofile](https://github.com/mit-han-lab/torchprofile.git)), you can run the commands below. By default, only the model size is profiled:
```bash
python train.py \
    --configs=configs/[task_name]/subtransformer/[model_name].yml \
    --sub-configs=configs/[task_name]/subtransformer/common.yml \
    [--latgpu|--latcpu|--profile-flops]
# for example
python train.py \
    --configs=configs/wmt14.en-de/subtransformer/HAT_wmt14ende_raspberrypi@3.5s_bleu@25.8.yml \
    --sub-configs=configs/wmt14.en-de/subtransformer/common.yml --latcpu
# another example
python train.py \
    --configs=configs/iwslt14.de-en/subtransformer/HAT_iwslt14deen_titanxp@137.8ms_bleu@34.7.yml \
    --sub-configs=configs/iwslt14.de-en/subtransformer/common.yml --profile-flops
```


### Training

#### 1. Train a SuperTransformer
The SuperTransformer is a supernet that contains many SubTransformers with weight-sharing.
By default, we train WMT tasks on 8 GPUs. Please adjust `--update-freq` according to GPU numbers (`128/x` for x GPUs). Note that for IWSLT, we only train on one GPU with `--update-freq=1`. 
```bash
python train.py --configs=configs/[task_name]/supertransformer/[search_space].yml
# for example
python train.py --configs=configs/wmt14.en-de/supertransformer/space0.yml
# another example
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --configs=configs/wmt14.en-fr/supertransformer/space0.yml --update-freq=32
```
In the `--configs` file, SuperTransformer model architecture, SubTransformer search space and training settings are specified.

We also provide pre-trained SuperTransformers for the four tasks as below. To download, run `python download_model.py --model-name=[model_name]`.

| Task | search_space | model_name | Link |
|:-----------:|:-----------:|:-----------|:-----------:|
| WMT'14 En-De | space0 | HAT_wmt14ende_super_space0 | [link](https://www.dropbox.com/s/pkdddxvvpw9a4vq/HAT_wmt14ende_super_space0.pt?dl=0) |
| WMT'14 En-Fr | space0 | HAT_wmt14enfr_super_space0 | [link](https://www.dropbox.com/s/asegvw9qzpxui6a/HAT_wmt14enfr_super_space0.pt?dl=0) |
| WMT'19 En-De | space0 | HAT_wmt19ende_super_space0 | [link](https://www.dropbox.com/s/uc0lw6jdep1vazc/HAT_wmt19ende_super_space0.pt?dl=0) |
| IWSLT'14 De-En | space1 | HAT_iwslt14deen_super_space1 | [link](https://www.dropbox.com/s/yv0mn8ns36gxkhs/HAT_iwslt14deen_super_space1.pt?dl=0) |


#### 2. Evolutionary Search
The second step of HAT is to perform an evolutionary search in the trained SuperTransformer with a hardware latency constraint in the loop. We train a latency predictor to get fast and accurate latency feedback.

##### 2.1 Generate a latency dataset
```bash
python latency_dataset.py --configs=configs/[task_name]/latency_dataset/[hardware_name].yml
# for example
python latency_dataset.py --configs=configs/wmt14.en-de/latency_dataset/cpu_raspberrypi.yml
```
`hardware_name` can be `cpu_raspberrypi`, `cpu_xeon` and `gpu_titanxp`. The `--configs` file contains the design space in which we sample models to get (model_architecture, real_latency) data pairs.

We provide the datasets we collect in the [latency_dataset](./latency_dataset) folder.

##### 2.2 Train a latency predictor
Then train a predictor with collected dataset:
```bash
python latency_predictor.py --configs=configs/[task_name]/latency_predictor/[hardware_name].yml
# for example
python latency_predictor.py --configs=configs/wmt14.en-de/latency_predictor/cpu_raspberrypi.yml
```
The `--configs` file contains the predictor's model architecture and training settings.
We provide pre-trained predictors in [latency_dataset/predictors](./latency_dataset/predictors) folder.

##### 2.3 Run evolutionary search with a latency constraint
```bash
python evo_search.py --configs=[supertransformer_config_file].yml --evo-configs=[evo_settings].yml
# for example
python evo_search.py --configs=configs/wmt14.en-de/supertransformer/space0.yml --evo-configs=configs/wmt14.en-de/evo_search/wmt14ende_titanxp.yml
```
The `--configs` file points to the SuperTransformer training config file. `--evo-configs` file includes evolutionary search settings, and also specifies the desired latency constraint `latency-constraint`. Note that the `feature-norm` and `lat-norm` here should be the same as those when training the latency predictor. `--write-config-path` specifies the location to write out the searched SubTransformer architecture. 


#### 3. Train a Searched SubTransformer
Finally, we train the search SubTransformer from scratch:
```bash
python train.py --configs=[subtransformer_architecture].yml --sub-configs=configs/[task_name]/subtransformer/common.yml
# for example
python train.py --configs=configs/wmt14.en-de/subtransformer/wmt14ende_titanxp@200ms.yml --sub-configs=configs/wmt14.en-de/subtransformer/common.yml
```

`--configs` points to the `--write-config-path` in step 2.3. `--sub-configs` contains training settings for the SubTransformer.

After training a SubTransformer, you can test its performance with the methods in [Testing](#testing) section.

### Dependencies
* Python >= 3.6
* [PyTorch](http://pytorch.org/) >= 1.0.0
* configargparse >= 0.14
* New model training requires NVIDIA GPUs and [NCCL](https://github.com/NVIDIA/nccl)

## Related works on efficient deep learning

[MicroNet for Efficient Language Modeling](https://arxiv.org/abs/2005.07877)

[Lite Transformer with Long-Short Range Attention](https://arxiv.org/abs/2004.11886)

[AMC: AutoML for Model Compression and Acceleration on Mobile Devices](https://arxiv.org/abs/1802.03494)

[Once-for-All: Train One Network and Specialize it for Efficient Deployment](https://arxiv.org/abs/1908.09791)

[ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://arxiv.org/abs/1812.00332)

## Contact
If you have any questions, feel free to contact [Hanrui Wang](https://hanruiwang.me) through Email ([hanrui@mit.edu](mailto:hanrui@mit.edu)) or Github issues. Pull requests are highly welcomed! 

## Licence

This repository is released under the MIT license. See [LICENSE](./LICENSE) for more information.

## Acknowledgements

We are thankful to [fairseq](https://github.com/pytorch/fairseq) as the backbone of this repo.
