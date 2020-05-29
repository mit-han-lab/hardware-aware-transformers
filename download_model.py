# HAT: Hardware-Aware Transformers for Efficient Natural Language Processing
# Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan and Song Han
# The 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
# Paper: https://arxiv.org/abs/2005.14187
# Project page: https://hanruiwang.me/project_pages/hat/

import argparse
import os
from tqdm import tqdm

url_dict = {
    'HAT_wmt14ende_raspberrypi@3.5s_bleu@25.8': 'https://www.dropbox.com/s/pmfwwg1d1kmfdh5/HAT_wmt14ende_raspberrypi@3.5s_bleu@25.8.pt?dl=0',
    'HAT_wmt14ende_raspberrypi@4.0s_bleu@26.9': 'https://www.dropbox.com/s/ko0i65k1664p74u/HAT_wmt14ende_raspberrypi@4.0s_bleu@26.9.pt?dl=0',
    'HAT_wmt14ende_raspberrypi@4.5s_bleu@27.6': 'https://www.dropbox.com/s/f4y6u9cbcdykeha/HAT_wmt14ende_raspberrypi@4.5s_bleu@27.6.pt?dl=0',
    'HAT_wmt14ende_raspberrypi@5.0s_bleu@27.8': 'https://www.dropbox.com/s/av5vycafxo57x6w/HAT_wmt14ende_raspberrypi@5.0s_bleu@27.8.pt?dl=0',
    'HAT_wmt14ende_raspberrypi@6.0s_bleu@28.2': 'https://www.dropbox.com/s/ywedqumq91a4ekn/HAT_wmt14ende_raspberrypi@6.0s_bleu@28.2.pt?dl=0',
    'HAT_wmt14ende_raspberrypi@6.9s_bleu@28.4': 'https://www.dropbox.com/s/x7iucaotbeald3q/HAT_wmt14ende_raspberrypi@6.9s_bleu@28.4.pt?dl=0',
    'HAT_wmt14ende_xeon@137.9ms_bleu@25.8': 'https://www.dropbox.com/s/bvq3y6igoyxe1t5/HAT_wmt14ende_xeon@137.9ms_bleu@25.8.pt?dl=0',
    'HAT_wmt14ende_xeon@204.2ms_bleu@27.6': 'https://www.dropbox.com/s/yg12xz504uw2g1s/HAT_wmt14ende_xeon@204.2ms_bleu@27.6.pt?dl=0',
    'HAT_wmt14ende_xeon@278.7ms_bleu@27.9': 'https://www.dropbox.com/s/l5ljas8zyg9ik65/HAT_wmt14ende_xeon@278.7ms_bleu@27.9.pt?dl=0',
    'HAT_wmt14ende_xeon@340.2ms_bleu@28.1': 'https://www.dropbox.com/s/fkp61h8jbyt524i/HAT_wmt14ende_xeon@340.2ms_bleu@28.1.pt?dl=0',
    'HAT_wmt14ende_xeon@369.6ms_bleu@28.2': 'https://www.dropbox.com/s/3mv3oaddeb132np/HAT_wmt14ende_xeon@369.6ms_bleu@28.2.pt?dl=0',
    'HAT_wmt14ende_xeon@450.9ms_bleu@28.5': 'https://www.dropbox.com/s/bjldda9nzj7cpni/HAT_wmt14ende_xeon@450.9ms_bleu@28.5.pt?dl=0',
    'HAT_wmt14ende_titanxp@57.1ms_bleu@25.8': 'https://www.dropbox.com/s/71w5t0qidsxqe1e/HAT_wmt14ende_titanxp@57.1ms_bleu@25.8.pt?dl=0',
    'HAT_wmt14ende_titanxp@91.2ms_bleu@27.6': 'https://www.dropbox.com/s/j0hnmxw6xz6tskh/HAT_wmt14ende_titanxp@91.2ms_bleu@27.6.pt?dl=0',
    'HAT_wmt14ende_titanxp@126.0ms_bleu@27.9': 'https://www.dropbox.com/s/pyetdnbz1zvcfg5/HAT_wmt14ende_titanxp@126.0ms_bleu@27.9.pt?dl=0',
    'HAT_wmt14ende_titanxp@146.7ms_bleu@28.1': 'https://www.dropbox.com/s/ixn832oai2k44j9/HAT_wmt14ende_titanxp@146.7ms_bleu@28.1.pt?dl=0',
    'HAT_wmt14ende_titanxp@208.1ms_bleu@28.5': 'https://www.dropbox.com/s/owpdwmqwpn9jw14/HAT_wmt14ende_titanxp@208.1ms_bleu@28.5.pt?dl=0',
    'HAT_wmt14enfr_raspberrypi@4.3s_bleu@38.8': 'https://www.dropbox.com/s/ku97fwz1oj1a112/HAT_wmt14enfr_raspberrypi@4.3s_bleu@38.8.pt?dl=0',
    'HAT_wmt14enfr_raspberrypi@5.3s_bleu@40.1': 'https://www.dropbox.com/s/9noopb605fqmjpl/HAT_wmt14enfr_raspberrypi@5.3s_bleu@40.1.pt?dl=0',
    'HAT_wmt14enfr_raspberrypi@5.8s_bleu@40.6': 'https://www.dropbox.com/s/vmdkh0ctpdac7gr/HAT_wmt14enfr_raspberrypi@5.8s_bleu@40.6.pt?dl=0',
    'HAT_wmt14enfr_raspberrypi@6.9s_bleu@41.1': 'https://www.dropbox.com/s/dbo9abn5pnb6qgz/HAT_wmt14enfr_raspberrypi@6.9s_bleu@41.1.pt?dl=0',
    'HAT_wmt14enfr_raspberrypi@7.8s_bleu@41.4': 'https://www.dropbox.com/s/x8tsbxbwkk64ejg/HAT_wmt14enfr_raspberrypi@7.8s_bleu@41.4.pt?dl=0',
    'HAT_wmt14enfr_raspberrypi@9.1s_bleu@41.8': 'https://www.dropbox.com/s/zbsbl5e96y3t5zl/HAT_wmt14enfr_raspberrypi@9.1s_bleu@41.8.pt?dl=0',
    'HAT_wmt14enfr_xeon@154.7ms_bleu@39.1': 'https://www.dropbox.com/s/6xswl0oesuvmqk5/HAT_wmt14enfr_xeon@154.7ms_bleu@39.1.pt?dl=0',
    'HAT_wmt14enfr_xeon@208.8ms_bleu@40.0': 'https://www.dropbox.com/s/ot3zt8nenda54j7/HAT_wmt14enfr_xeon@208.8ms_bleu@40.0.pt?dl=0',
    'HAT_wmt14enfr_xeon@329.4ms_bleu@41.1': 'https://www.dropbox.com/s/epe2lvus4l40v9o/HAT_wmt14enfr_xeon@329.4ms_bleu@41.1.pt?dl=0',
    'HAT_wmt14enfr_xeon@394.5ms_bleu@41.4': 'https://www.dropbox.com/s/qnt2qzkb3i054c6/HAT_wmt14enfr_xeon@394.5ms_bleu@41.4.pt?dl=0',
    'HAT_wmt14enfr_xeon@442.0ms_bleu@41.7': 'https://www.dropbox.com/s/79zcolb53jbhchk/HAT_wmt14enfr_xeon@442.0ms_bleu@41.7.pt?dl=0',
    'HAT_wmt14enfr_titanxp@69.3ms_bleu@39.1': 'https://www.dropbox.com/s/hvy255ls277onjw/HAT_wmt14enfr_titanxp@69.3ms_bleu@39.1.pt?dl=0',
    'HAT_wmt14enfr_titanxp@94.9ms_bleu@40.0': 'https://www.dropbox.com/s/rvfv99jbh4n7qys/HAT_wmt14enfr_titanxp@94.9ms_bleu@40.0.pt?dl=0',
    'HAT_wmt14enfr_titanxp@132.9ms_bleu@40.7': 'https://www.dropbox.com/s/u6u3u40pr4f5mzh/HAT_wmt14enfr_titanxp@132.9ms_bleu@40.7.pt?dl=0',
    'HAT_wmt14enfr_titanxp@168.3ms_bleu@41.1': 'https://www.dropbox.com/s/wlbbmnrl61dx4z7/HAT_wmt14enfr_titanxp@168.3ms_bleu@41.1.pt?dl=0',
    'HAT_wmt14enfr_titanxp@208.3ms_bleu@41.7': 'https://www.dropbox.com/s/e41lnsktn5bb2fz/HAT_wmt14enfr_titanxp@208.3ms_bleu@41.7.pt?dl=0',
    'HAT_wmt19ende_titanxp@55.7ms_bleu@42.4': 'https://www.dropbox.com/s/6pokem8orb75ldh/HAT_wmt19ende_titanxp@55.7ms_bleu@42.4.pt?dl=0',
    'HAT_wmt19ende_titanxp@93.2ms_bleu@44.4': 'https://www.dropbox.com/s/zgcd70pzf1sle4z/HAT_wmt19ende_titanxp@93.2ms_bleu@44.4.pt?dl=0',
    'HAT_wmt19ende_titanxp@134.5ms_bleu@45.4': 'https://www.dropbox.com/s/mm827rst6a144zy/HAT_wmt19ende_titanxp@134.5ms_bleu@45.4.pt?dl=0',
    'HAT_wmt19ende_titanxp@176.1ms_bleu@46.2': 'https://www.dropbox.com/s/y0vov0n9zt50n9c/HAT_wmt19ende_titanxp@176.1ms_bleu@46.2.pt?dl=0',
    'HAT_wmt19ende_titanxp@204.5ms_bleu@46.5': 'https://www.dropbox.com/s/w1si4mgf1e3l8oj/HAT_wmt19ende_titanxp@204.5ms_bleu@46.5.pt?dl=0',
    'HAT_wmt19ende_titanxp@237.8ms_bleu@46.7': 'https://www.dropbox.com/s/rljih3t0hglp39a/HAT_wmt19ende_titanxp@237.8ms_bleu@46.7.pt?dl=0',
    'HAT_iwslt14deen_titanxp@45.6ms_bleu@33.4': 'https://www.dropbox.com/s/ntj1gfskish8vz3/HAT_iwslt14deen_titanxp@45.6ms_bleu@33.4.pt?dl=0',
    'HAT_iwslt14deen_titanxp@74.5ms_bleu@34.2': 'https://www.dropbox.com/s/gjq46181s3xbz0k/HAT_iwslt14deen_titanxp@74.5ms_bleu@34.2.pt?dl=0',
    'HAT_iwslt14deen_titanxp@109.0ms_bleu@34.5': 'https://www.dropbox.com/s/fg3r3tk2vjg0diq/HAT_iwslt14deen_titanxp@109.0ms_bleu@34.5.pt?dl=0',
    'HAT_iwslt14deen_titanxp@137.8ms_bleu@34.7': 'https://www.dropbox.com/s/3j5vu5jh71xwec1/HAT_iwslt14deen_titanxp@137.8ms_bleu@34.7.pt?dl=0',
    'HAT_iwslt14deen_titanxp@168.8ms_bleu@34.8': 'https://www.dropbox.com/s/5xy9hdjuc5c6sw5/HAT_iwslt14deen_titanxp@168.8ms_bleu@34.8.pt?dl=0',
    'HAT_wmt14ende_super_space0': 'https://www.dropbox.com/s/pkdddxvvpw9a4vq/HAT_wmt14ende_super_space0.pt?dl=0',
    'HAT_wmt14enfr_super_space0': 'https://www.dropbox.com/s/asegvw9qzpxui6a/HAT_wmt14enfr_super_space0.pt?dl=0',
    'HAT_wmt19ende_super_space0': 'https://www.dropbox.com/s/uc0lw6jdep1vazc/HAT_wmt19ende_super_space0.pt?dl=0',
    'HAT_iwslt14deen_super_space1': 'https://www.dropbox.com/s/yv0mn8ns36gxkhs/HAT_iwslt14deen_super_space1.pt?dl=0'
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='HAT_wmt14ende_raspberrypi@3.5s_bleu@25.8', help='which model to download, please see README.md for details')
    parser.add_argument('--save-path', type=str, default='./downloaded_models')
    parser.add_argument('--download-all', action='store_true')

    args = parser.parse_args()

    print(args)
    os.makedirs(args.save_path, exist_ok=True)

    # dropbox python api requires access tokens, so use wget
    if not args.download_all:
        cmd = 'wget -O ' + args.save_path + '/' + args.model_name + '.pt ' + url_dict[args.model_name]
        os.system(cmd)
    else: # download all files
        for k, v in tqdm(url_dict.items()):
            cmd = 'wget -O ' + args.save_path + '/' + k + '.pt ' + v
            os.system(cmd)
