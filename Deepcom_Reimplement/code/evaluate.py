import argparse
import nltk
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction,sentence_bleu
from util.CustomedBleu.smooth_bleu import smooth_bleu
from util.meteor.meteor import Meteor
from util.rouge.rouge import Rouge
from util.cider.cider import Cider

def bleus(refs, preds):
    # sentence bleu
    sentence_bleu0 = [sentence_bleu(ref, pred) for ref, pred in zip(refs, preds)]
    sentence_bleu0 = np.mean(sentence_bleu0)
    sentence_bleu0 = round(sentence_bleu0 * 100, 4)


    # emse_bleu
    all_score = 0.0
    count = 0
    for r, p in zip(refs, preds):
        # nltk bug: https://github.com/nltk/nltk/issues/2204
        if len(p) == 1:
            continue
        score = nltk.translate.bleu(r, p, smoothing_function=SmoothingFunction().method4)
        all_score += score
        count += 1
    emse_bleu = round(all_score / count * 100, 4)



    bleus_dict = {"BLEU-DM": sentence_bleu0,'BLEU-DC': emse_bleu}

    return bleus_dict


def metetor_rouge_cider(refs, preds):
    refs_dict = {}
    preds_dict = {}
    for i in range(len(preds)):
        preds_dict[i] = [" ".join(preds[i])]
        refs_dict[i] = [" ".join(refs[i][0])]

    score_Rouge, scores_Rouge = Rouge().compute_score(refs_dict, preds_dict)
    print("rouge: ", score_Rouge)

    score_Cider, scores_Cider = Cider().compute_score(refs_dict, preds_dict)
    print("cider: ", score_Cider)

    score_Meteor, scores_Meteor = Meteor().compute_score(refs_dict, preds_dict)
    print("meteor: ", score_Meteor)


def read_to_list(filename):
    f = open(filename, 'r')
    res = []
    for row in f:
        # (rid, text) = row.split('\t')
        res.append(row.split())
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refs_filename', type=str, required=True)
    parser.add_argument('--preds_filename', type=str, required=True)
    args = parser.parse_args()

    refs = read_to_list(args.refs_filename)
    refs = [[t] for t in refs]
    preds = read_to_list(args.preds_filename)
    bleus_dict = bleus(refs, preds)
    print(bleus_dict)
    metetor_rouge_cider(refs, preds)
    


if __name__ == "__main__":
    main()
