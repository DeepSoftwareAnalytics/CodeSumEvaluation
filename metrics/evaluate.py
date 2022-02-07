#!/usr/bin/env python
# !-*-coding:utf-8 -*-
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import numpy as np
import sys

sys.path.append("./")
from bleu.codenn_bleu import codenn_smooth_bleu
from bleu.google_bleu import compute_bleu
from bleu.rencos_bleu import Bleu as recos_bleu
import warnings
import argparse
import logging
import prettytable as pt

warnings.filterwarnings('ignore')
logging.basicConfig(format='[%(asctime)s - %(levelname)s - %(name)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def bleus(refs, preds):
    # sentence bleu
    sentence_bleu0 = [sentence_bleu(ref, pred) for ref, pred in zip(refs, preds)]
    sentence_bleu0 = np.mean(sentence_bleu0)
    sentence_bleu0 = round(sentence_bleu0 * 100, 4)

    # corpus bleu
    c_bleu4 = corpus_bleu(refs, preds, weights=(0.25, 0.25, 0.25, 0.25))
    c_bleu4 = round(c_bleu4 * 100, 4)

    # emse_bleu
    all_score = 0.0
    count = 0
    for r, p in zip(refs, preds):
        # nltk bug: https://github.com/nltk/nltk/issues/2204
#         if len(p) == 1:
#             continue
        score = nltk.translate.bleu(r, p, smoothing_function=SmoothingFunction().method4)
        all_score += score
        count += 1
#     try:
        emse_bleu = round(all_score / count * 100, 4)
#     except:
#         emse_bleu = 0

    # codenn bleu
    r_str_list = []
    p_str_list = []
    for r, p in zip(refs, preds):
        if len(r[0]) == 0 or len(p) == 0:
            continue
        r_str_list.append([" ".join([str(token_id) for token_id in r[0]])])
        p_str_list.append(" ".join([str(token_id) for token_id in p]))
    try:
        bleu_list = codenn_smooth_bleu(r_str_list, p_str_list)
    except:
        bleu_list = [0, 0, 0, 0]
    codenn_bleu = bleu_list[0]

    codenn_bleu = round(codenn_bleu, 4)

    google_bleu4 = [compute_bleu([ref], [pred], smooth=True)[0] for ref, pred in zip(refs, preds)]
    google_bleu4 = np.mean(google_bleu4)
    google_bleu4 = round(google_bleu4 * 100, 4)

    res = {k: [" ".join(v)] for k, v in enumerate(preds)}
    gts = {k: [" ".join(v[0])] for k, v in enumerate(refs)}
    _, scores_Bleu = recos_bleu(4).compute_score(gts, res)

    rencos_bleu4 =round(np.mean(scores_Bleu[3])*100, 4)

    bleus_dict = {"BLEU-DCOM": sentence_bleu0,
                  'BLEU-FC': c_bleu4,
                  'BLEU-DC': emse_bleu,
                  'BLEU-CN': codenn_bleu,
                  "BLEU-NCS": google_bleu4,
                  "BLEU-RC": rencos_bleu4
}
    return bleus_dict


def read_to_list(filename):
    f = open(filename, 'r',encoding="utf-8")
    res = []
    for row in f:
        # (rid, text) = row.split('\t')
        res.append(row.split())
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refs_filename', type=str, default="test/test.gold", required=False, help="The path of the reference file")
    parser.add_argument('--preds_filename', type=str, default="test/test.pred", required=False, help="The path of the predicted file")
    args = parser.parse_args()
    if args.refs_filename and args.preds_filename:
        logging.info("loading data")
        refs = read_to_list(args.refs_filename)
        refs = [[t] for t in refs]
        preds = read_to_list(args.preds_filename)
    else:
        raise RuntimeError("Please specify refs_filename and -preds_filename")
    all_bleu = bleus(refs, preds)
    # logging.info(all_bleu)
    tb = pt.PrettyTable()
    tb.field_names = all_bleu.keys()
    tb.add_row(all_bleu.values())
    print(tb)


if __name__ == '__main__':
    main()
