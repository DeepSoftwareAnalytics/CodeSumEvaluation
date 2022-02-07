import argparse
import numpy as np
import os
import logging.config
import random
import torch
import pickle
import torch.nn as nn
from model import DeepComModel
from evaluate import bleus
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import sys
sys.path.append("../")
from data.utils import read_json_file
from collections import Counter

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
            'datefmt': '%m/%d/%Y %H:%M:%S'}},
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout'}},
    'loggers': {'': {'handlers': ['default']}}
}
logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)


def padding(line, max_len, padding_id):
    line_len = len(line)
    if line_len < max_len:
        line += [padding_id] * (max_len - line_len)
    return line


def build_vocab(word_count, start_id, vocab_size=-1):
    w2i, i2w = {}, {}
    # word_count_ord[i][0] -> word, word_count_ord[i][1] -> count
    word_count_ord = sorted(word_count.items(), key=lambda item: item[1], reverse=True)

    if vocab_size > 0:
        if vocab_size < len(word_count):
            size = vocab_size
        else:
            size = len(word_count)
    else:
        size = len(word_count_ord)
        logger.info("use all tokens %d " % size)

    for i in range(size):
        w2i[word_count_ord[i][0]] = i + start_id
        i2w[i + start_id] = word_count_ord[i][0]

    return w2i, i2w


def build_vocab_with_pad_unk(word_count, start_id, args, vocab_size=-1):
    w2i, i2w = build_vocab(word_count, start_id, vocab_size)

    w2i[args.pad_token] = args.pad_id
    i2w[args.pad_id] = args.pad_token

    unk_id = len(w2i)
    w2i[args.unk_token] = unk_id
    i2w[unk_id] = args.unk_token
    return w2i, i2w


def build_vocab_with_pad_unk_sos_eos(word_count, start_id, args, vocab_size=-1):
    w2i, i2w = build_vocab(word_count, start_id, vocab_size)
    w2i[args.sos_token] = args.sos_id
    i2w[args.sos_id] = args.sos_token
    w2i[args.eos_token] = args.eos_id
    i2w[args.eos_id] = args.eos_token
    w2i[args.pad_token] = args.pad_id
    i2w[args.pad_id] = args.pad_token

    unk_id = len(w2i)
    w2i[args.unk_token] = unk_id
    i2w[unk_id] = args.unk_token
    return w2i, i2w


#  Make a new directory if it is not exist.
def make_directory(path):
    is_exist = os.path.exists(path)
    if not is_exist:
        os.makedirs(path)
    else:
        pass


def code2ids(tokens, w2i, seq_len, args):
    unk_id = w2i[args.unk_token]
    # ids = [w2i.get(token, w2i.get(token.split("_")[0], unk_id)) for token in tokens[:seq_len]]
    ids = [w2i.get(token, unk_id) for token in tokens[:seq_len]]
    ids = padding(ids, seq_len, args.pad_id)
    return ids


def summary2ids(summary_tokens, summary_w2i, seq_len, args):
    summary_unk_id = summary_w2i[args.unk_token]
    summary_ids = [summary_w2i.get(token, summary_unk_id) for token in summary_tokens[:seq_len - 1]]
    summary_ids.insert(0, args.sos_id)
    if len(summary_ids) < seq_len:
        summary_ids.append(args.eos_id)
    summary_ids = padding(summary_ids, seq_len, args.pad_id)
    return summary_ids


def tokens2ids(summary_tokens, sbt_tokens, sbt_w2i, summary_w2i, args):
    sbt_max_len, sum_max_len = args.sbt_len, args.summary_len,
    sbt_token_ids = [code2ids(item, sbt_w2i, sbt_max_len, args) for item in sbt_tokens]
    summary_ids = [summary2ids(item, summary_w2i, sum_max_len, args) for item in summary_tokens]

    return sbt_token_ids, summary_ids


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pad_id', type=int, default=0)
    parser.add_argument('--sos_id', type=int, default=1)
    parser.add_argument('--eos_id', type=int, default=2)
    parser.add_argument('--unk_id', type=int, default=-1, help='unk_id is comvocabsize/datvocabsize/smlvocabsize - 1')
    parser.add_argument('--pad_token', type=str, default='<NULL>')
    parser.add_argument('--sos_token', type=str, default='<s>')
    parser.add_argument('--eos_token', type=str, default='</s>')
    parser.add_argument('--unk_token', type=str, default='<UNK>')

    # parser.add_argument('--summary_gt_filename', required=True)
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    # parser.add_argument("--input_filename", default=None, type=str, required=True)

    parser.add_argument("--load_model_dir", type=str)
    parser.add_argument('--step_log_freq', type=int, default=10, required=False)
    parser.add_argument('--debug', action='store_true', help='debug mode', required=False)
    parser.add_argument('--n_debug_samples', type=int, default=100, required=False)
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--do_train", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument("--do_eval", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument("--do_test", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument("--eval_frequency", default=1, type=int, required=False)
    parser.add_argument("--trim_til_eos", action='store_true', default=True, required=False)
    parser.add_argument("--use_full_sum", action='store_true', default=True, required=False)
    parser.add_argument("--use_oov_sum", action='store_true', default=False, required=False)
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('-data', required=False)
    parser.add_argument('-batch_size', type=int, default=32, required=False)
    parser.add_argument('-summary_dim', type=int, default=128, required=False)
    parser.add_argument('-sbt_dim', type=int, default=128, required=False)
    parser.add_argument('-rnn_hidden_size', type=int, default=128, required=False)
    parser.add_argument('-lr', type=float, default=0.001, required=False)
    parser.add_argument('-lr_decay', type=float, default=None, required=False)
    parser.add_argument('-epoch', type=int, default=10, required=False)
    # code processing  / 5
    # parser.add_argument('-djl', action='store_true', help="Parse source code using javalang")
    # parser.add_argument('-dfp', action='store_true', help="Filter punctuation in code tokens")
    # parser.add_argument('-dsi', action='store_true', help="Split identifiers according to camelCase and snake_case")
    # parser.add_argument('-dlc', action='store_true', help="Lowercase code tokens")
    # parser.add_argument('-dr', action='store_true', help="Replace string and number in code tokens")
    # summary processing/3
    parser.add_argument('-cfp', action='store_true', help="Filter punctuation in summaries")
    parser.add_argument('-csi', action='store_true', help="Split summary token according to camelCase and snake_case")
    parser.add_argument('-clc', action='store_true', help="Lowercase summary tokens")
    # seq len /3
    # parser.add_argument('-dlen', "--code_len", type=int, default=50, required=False)
    parser.add_argument('-clen', "--summary_len", type=int, default=10, required=False)
    parser.add_argument('-slen', "--sbt_len", type=int, default=50, required=False)
    # voc size /3
    # parser.add_argument('-dvoc', "--code_vocab_size", type=int, default=100, required=False)
    parser.add_argument('-cvoc', "--summary_vocab_size", type=int, default=100, required=False)
    parser.add_argument('-svoc', "--sbt_vocab_size", type=int, default=100, required=False)
    # beam search
    parser.add_argument('-beam_search_method', default='greedy', required=False)
    parser.add_argument('-beam_width', default=1, type=int, required=False)
    # split way
    parser.add_argument('-pkg', "--package_wise", type=str, choices=["True", "False"], required=False)
    parser.add_argument('-mtd', "--method_wise", type=str, choices=["True", "False"], required=False)

    parser.add_argument('--num_subprocesses', type=int, default=4)
    args = parser.parse_args()
    return args


def get_data(args):
    # code_process = 'djl{}_dfp{}_dsi{}_dlc{}_dr{}'.format(str(args.djl + 0), str(args.dfp + 0), str(args.dsi + 0),
    #                                                      str(args.dlc + 0), str(args.dr + 0))
    # sbt_process = "sbt"
    # summary_process = 'cfp{}_csi{}_cfd0_clc{}'.format(str(args.cfp + 0), str(args.csi + 0), str(args.clc + 0))
    # param_setting = summary_process + "_" + sbt_process
    # param_setting = "cfp0_csi0_cfd0_clc1_sbt"

    # word_count
    data_dir = args.data_dir
    model_params = None
    if args.load_model_dir is not None:
        try:
            logger.info("reload vocab from {}".format(args.load_model_dir))

            # model.load_state_dict(torch.load(os.path.join(args.load_model_dir, "model.pt")))
            model_params = torch.load(os.path.join(args.load_model_dir, "model.pt"))
            vocab = model_params["sbt_vocab"]
            sbt_word_count, sbt_w2i, sbt_i2w = vocab["word_count"], vocab["w2i"], vocab["i2w"]
            vocab = model_params["summary_vocab"]
            summary_word_count, summary_w2i, summary_i2w = vocab["word_count"], vocab["w2i"], vocab["i2w"]
        except FileNotFoundError:
            logger.info('Checkpoint is not found. Train the model from scratch')
    if not model_params:
        filename = os.path.join(data_dir, "train/word_count/sbt_word_count.json")
        sbt_word_count = read_json_file(filename)
        # type_dict = {}
        # for word, wf in sbt_word_count.items():
        #     if len(word.split("_")) > 1:
        #         type_dict[word.split("_")[0]] = type_dict.get(word.split("_")[0], 0) + wf
        # X, Y = Counter(sbt_word_count), Counter(type_dict)
        # sbt_word_count = dict(X + Y)

        filename = os.path.join(data_dir, "train/word_count/summary_word_count.json")
        summary_word_count = read_json_file(filename)
        # build sbt vocabulary
        sbt_w2i, sbt_i2w = build_vocab_with_pad_unk(sbt_word_count, 1, args, vocab_size=args.sbt_vocab_size - 2)
        # build summary vocab, include <s>, </s>, <NULL> and <UNK>. Start after </s>=2.
        summary_w2i, summary_i2w = build_vocab_with_pad_unk_sos_eos(summary_word_count, 3, args,
                                                                    vocab_size=args.summary_vocab_size - 4)
        # logger.info("sbt_w2i%d, summary_w2i%d" % (len(sbt_w2i), len(summary_w2i)))

    # sbt and summary
    n_debug_samples = args.n_debug_samples  # 100
    # train
    filename = os.path.join(data_dir, "train/sbt.json")
    sbt = read_json_file(filename)
    filename = os.path.join(data_dir, "train/summary.json")
    summary = read_json_file(filename)
    if args.debug:
        sbt = sbt[:n_debug_samples]
        summary = summary[:n_debug_samples]
    strain, ctrain = tokens2ids(summary, sbt, sbt_w2i, summary_w2i, args)
    # save_pickle_data("./", "data.pkl",strain[:200])
    # logger.info("sbt_w2i%d, summary_w2i%d" % (len(sbt_w2i), len(summary_w2i)))

    # val
    filename = os.path.join(data_dir, "val/sbt.json")
    sbt = read_json_file(filename)
    filename = os.path.join(data_dir, "val/summary.json")
    val_summary = read_json_file(filename)
    if args.debug:
        sbt = sbt[:n_debug_samples]
        val_summary = val_summary[:n_debug_samples]
    sval, cval = tokens2ids(val_summary, sbt, sbt_w2i, summary_w2i, args)

    # test
    filename = os.path.join(data_dir, "test/sbt.json")
    sbt = read_json_file(filename)
    filename = os.path.join(data_dir, "test/summary.json")
    test_summary = read_json_file(filename)
    if args.debug:
        sbt = sbt[:n_debug_samples]
        test_summary = test_summary[:n_debug_samples]
    stest, ctest = tokens2ids(test_summary, sbt, sbt_w2i, summary_w2i, args)
    data = {"ctrain": ctrain, "cval": cval, "ctest": ctest,
            "strain": strain, "sval": sval, "stest": stest,
            "comstok": {"i2w": summary_i2w, "w2i": summary_w2i, "word_count": summary_word_count},
            "sbtstok": {"i2w": sbt_i2w, "w2i": sbt_w2i, "word_count": sbt_word_count},
            "cf": {"sbtvocabsize": len(sbt_i2w), "comvocabsize": len(summary_i2w)}}

    return data, val_summary, test_summary


def build_dataloader(data, args):
    all_source_ids = torch.tensor(data['strain'], dtype=torch.long)
    all_target_ids = torch.tensor(data['ctrain'], dtype=torch.long)
    train_dataset = TensorDataset(all_source_ids, all_target_ids)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    all_source_ids = torch.tensor(data['sval'], dtype=torch.long)
    all_target_ids = torch.tensor(data['cval'], dtype=torch.long)
    val_dataset = TensorDataset(all_source_ids, all_target_ids)
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.batch_size)

    all_source_ids = torch.tensor(data['stest'], dtype=torch.long)
    all_target_ids = torch.tensor(data['ctest'], dtype=torch.long)
    test_dataset = TensorDataset(all_source_ids, all_target_ids)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    logger.info("train {}, val {}, test {}".format(len(train_dataset), len(val_dataset), len(test_dataset)))
    return train_dataloader, val_dataloader, test_dataloader


def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    args = parse_args()
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device

    logger.info(args)
    set_seed(args.seed)
    # load data
    data, val_trgs, test_trgs = get_data(args)
    sbt_vocab, summary_vocab = data["sbtstok"], data["comstok"]
    train_dataloader, val_dataloader, test_dataloader = build_dataloader(data, args)

    sbt_vocab_size = len(sbt_vocab['i2w'])
    summary_vocab_size = len(summary_vocab['i2w'])
    logger.info("sbt_vocab_size%d, summary_vocab_size%d" % (sbt_vocab_size, summary_vocab_size))

    args.unk_id = len(summary_vocab['i2w']) - 1

    # load target
    if not args.use_full_sum:
        val_trgs = [t[:args.summary_len - 1] for t in val_trgs]
    val_trgs = [[t] for t in val_trgs]

    if not args.use_full_sum:
        test_trgs = [t[:args.summary_len - 1] for t in test_trgs]
    test_trgs = [[t] for t in test_trgs]

    # create model
    model = DeepComModel(sbt_vocab_size, summary_vocab_size, args.summary_len, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    if args.load_model_dir is not None:
        try:
            logger.info("reload model from {}".format(args.load_model_dir))
            # model.load_state_dict(torch.load(os.path.join(args.load_model_dir, "model.pt")))
            model_params = torch.load(os.path.join(args.load_model_dir, "model.pt"))
            model.load_state_dict(model_params["model_state_dict"])
            if args.do_train:
                args_checkpoint = torch.load(os.path.join(args.load_model_dir, "args.pt"))
                optimizer.load_state_dict(args_checkpoint['optimizer_state_dict'])
                if torch.cuda.is_available() and not args.no_cuda:
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()
        except FileNotFoundError:
            args_checkpoint = None
            logger.info('Checkpoint is not found. Train the model from scratch')

    model.to(device)
    if args.local_rank != -1:  # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    loss_fn = nn.CrossEntropyLoss(ignore_index=args.pad_id)
    table = model.module.model_parameters() if hasattr(model, 'module') else model.model_parameters()
    logger.info('Breakdown of the trainable paramters\n%s' % table)
    logger.info('The model has %s trainable parameters' % str(count_parameters(model)))

    # Start training
    if args.do_train:
        make_directory(os.path.join(args.output_dir, "checkpoint-last"))
        make_directory(os.path.join(args.output_dir, "model-best-loss"))
        make_directory(os.path.join(args.output_dir, "model-best-bleu"))
        logger.info("***** Running training *****")
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Num epoch = %d", args.epoch)
        best_loss, best_loss_epoch, best_bleu, best_bleu_epoch, checkpoint_epoch = 1e6, 0, -1.0, 0, 0
        if args.load_model_dir and args_checkpoint:
            best_loss = args_checkpoint['best_loss']
            best_bleu = args_checkpoint['best_bleu']
            checkpoint_epoch = args_checkpoint['epoch']
            best_loss_epoch = args_checkpoint['best_loss_epoch']
            best_bleu_epoch = args_checkpoint['best_bleu_epoch']
        for epoch in range(checkpoint_epoch, args.epoch):
            model.train()
            tr_loss = 0
            for idx, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                output = model(method_sbt=batch[0], beam_width=args.beam_width, is_test=False, args=args)
                sum_vocab_size = output.shape[-1]
                output = output.view(-1, sum_vocab_size)  # output: batch_size, summary_length - 1, sum_vocab_size
                trg = batch[-1][:, 1:].reshape(-1)  # exclude <s> # trg = [batch size * (summary_length - 1)]
                loss = loss_fn(output, trg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad()
                tr_loss += loss.item()
                avg_loss = round(tr_loss / (idx + 1), 4)
                if (idx + 1) % args.step_log_freq == 0:
                    logger.info("epoch {} step {} loss {}".format(epoch, idx + 1, avg_loss))

            logger.info("epoch {} loss {}".format(epoch, round(tr_loss, 4)))
            # save last checkpoint
            model_to_save = model.module if hasattr(model, 'module') else model
            # torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, "checkpoint-last", "model.pt"))
            model_params = {
                'model_state_dict': model_to_save.state_dict(),
                "sbt_vocab": sbt_vocab,
                "summary_vocab": summary_vocab
            }
            torch.save(model_params, os.path.join(args.output_dir, "checkpoint-last", "model.pt"))
            params = {
                # 'model_state_dict': model_to_save.state_dict(),
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'best_bleu': best_bleu,
                'best_loss_epoch': best_loss_epoch,
                'best_bleu_epoch': best_bleu_epoch,
            }
            try:
                torch.save(params, os.path.join(args.output_dir, "checkpoint-last", "args.pt"))
            except BaseException:
                logger.warning('WARN: Saving failed... continuing anyway.')

            if args.lr_decay:
                optimizer.param_groups[0]['lr'] = args.lr * args.lr_decay

            if args.do_eval:  # eval model with dev dataset
                logger.info("***** Running evaluation *****")
                eval_loss = 0
                model.eval()
                preds = []
                for batch in val_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    with torch.no_grad():
                        # output = model(method_sbt=batch[0], beam_width=args.beam_width, is_test=True, args=args)
                        output = model(method_sbt=batch[0], beam_width=0, is_test=True, args=args)
                        pred = torch.argmax(output, dim=2)
                        sum_vocab_size = output.shape[-1]
                        output = output.view(-1, sum_vocab_size)
                        trg = batch[-1][:, 1:].reshape(-1)  # exclude <s> # trg = [batch size * (summary_length - 1)]
                        loss = loss_fn(output, trg)
                        eval_loss += loss.item()
                        # If p or t contain <s>, </s>, <UNK>, <NULL>, they will be included as output.
                        for p in pred:
                            p = p.cpu().tolist()
                            if args.trim_til_eos:
                                if args.eos_id in p:  # truncate at first </s>
                                    p = p[:p.index(args.eos_id) + 1]
                            p = [summary_vocab['i2w'][value] for value in p if
                                 value not in [args.pad_id, args.sos_id, args.eos_id, args.unk_id]]
                            preds.append(p)
                model.train()
                # save best eval loss checkpoint
                if eval_loss < best_loss:
                    best_loss, best_loss_epoch = round(eval_loss, 4), epoch
                    model_to_save = model.module if hasattr(model, 'module') else model
                    # torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, "model-best-loss", "model.pt"))
                    model_params = {
                        'model_state_dict': model_to_save.state_dict(),
                        "sbt_vocab": sbt_vocab,
                        "summary_vocab": summary_vocab
                    }
                    torch.save(model_params, os.path.join(args.output_dir, "model-best-loss", "model.pt"))
                logger.info("  val loss: {}".format(eval_loss))
                logger.info("  best val loss: {}".format(best_loss))
                logger.info("  best loss epoch: {}".format(best_loss_epoch))

                # save best eval bleu checkpoint
                val_bleu = round(bleus(val_trgs[:len(preds)], preds)['BLEU-DM'], 4)
                if best_bleu < val_bleu:
                    best_bleu, best_bleu_epoch = val_bleu, epoch
                    model_to_save = model.module if hasattr(model, 'module') else model
                    # torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, "model-best-bleu", "model.pt"))
                    model_params = {
                        'model_state_dict': model_to_save.state_dict(),
                        "sbt_vocab": sbt_vocab,
                        "summary_vocab": summary_vocab
                    }
                    torch.save(model_params, os.path.join(args.output_dir, "model-best-bleu", "model.pt"))
                logger.info("  val bleu: {}".format(val_bleu))
                logger.info("  best val bleu: {}".format(best_bleu))
                logger.info("  best bleu epoch: {}".format(best_bleu_epoch))

    if args.do_test:
        if args.do_train:
            model_params = torch.load(os.path.join(args.output_dir, "model-best-bleu", "model.pt"))
            if hasattr(model, 'module'):
                model.module.load_state_dict(model_params["model_state_dict"])
            else:
                model.load_state_dict(model_params["model_state_dict"])
        logger.info("***** Running testing *****")
        model.eval()
        preds = []
        for batch in test_dataloader:
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                output = model(method_sbt=batch[0], beam_width=args.beam_width, is_test=True, args=args)
                pred = torch.argmax(output, dim=2)
                sum_vocab_size = output.shape[-1]
                # output = output.view(-1, sum_vocab_size)
                # trg = batch[-1][:, 1:].reshape(-1)  # exclude <s> # trg = [batch size * (summary_length - 1)]
                for p in pred:
                    p = p.cpu().tolist()
                    if args.trim_til_eos:
                        if args.eos_id in p:  # truncate at first </s>
                            p = p[:p.index(args.eos_id) + 1]
                    p = [summary_vocab['i2w'][value] for value in p if
                         value not in [args.pad_id, args.sos_id, args.eos_id, args.unk_id]]
                    preds.append(p)

        model.train()
        with open(os.path.join(args.output_dir, "test.pred"), 'w') as f, open(
                os.path.join(args.output_dir, "test.gold"), 'w') as f1:
            # for fid, pred, ref in zip(test_fids, preds, test_trgs[:len(preds)]):
            for pred, ref in zip(preds, test_trgs[:len(preds)]):
                f.write('{}\n'.format(' '.join(pred)))
                f1.write('{}\n'.format(' '.join(ref[0])))

        test_bleu = round(bleus(test_trgs[:len(preds)], preds)['BLEU-DM'], 4)
        logger.info(" test bleu: {}".format(test_bleu))


if __name__ == "__main__":
    main()
