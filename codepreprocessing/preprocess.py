import os
import sys
import time

import logging
import logging.config
import argparse
from multiprocessing import cpu_count, Pool
import sys
import random
from functools import partial
import pickle
import json

sys.path.append("../")
from utils import filter_punctuation_pl, code_tokens_split_identifier, lower_case_str_arr, save_json_data, \
    count_word_parallel, get_all_tokens, time_format, tokenize_source_code, code_tokens_replace_str_num, \
    merge_code_summary_sbt_ast, read_json_file, load_summary, \
    write_source_code_to_java_file, sbt_parser, split_by_whitespace, gen_ast, get_sbt_using_javalang

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


def process(args):
    """
    processing code/summary and count the word frequency.
    """
    is_cfp = args.cfp
    is_csi = args.csi
    is_clc = args.clc
    is_djl = args.djl
    is_dsi = args.dsi
    is_dfp = args.dfp
    is_dlc = args.dlc
    is_dr = args.dr

    code_params = 'djl{}_dr{}_dsi{}_dfp{}_dlc{}'.format(is_djl + 0, is_dr + 0, is_dsi + 0, is_dfp + 0, is_dlc + 0)
    summary_params = 'cfp{}_csi{}_cfd0_clc{}'.format(is_cfp + 0, is_csi + 0, is_clc + 0)
    sbt_params = "sbt{}".format(args.sbt_type)
    ast_params = "ast"
    params = code_params + "_" + summary_params + "_" + sbt_params + "_" + ast_params 
    outdir = os.path.join(args.output_dir, params)
    # outdir = args.output_dir
    # duplication ratio
    logging.info("output directory %s" % outdir)

    #
    ori_data = pickle.load(open(args.data_filename, "rb"))
    summary = {part: {fid: item["summary"] for fid, item in ori_data[part].items()} for part in ori_data}
    process_summary(args, outdir, is_cfp, is_csi, is_clc, summary)
    code = {part: {fid: item['code'] for fid, item in ori_data[part].items()} for part in ori_data}

    process_code(args, outdir, is_djl, is_dr, is_dsi, is_dfp, is_dlc,code)
    process_sbt(args, outdir,code)
    process_ast(args, outdir,code)

    # merge_code_summary_sbt(outdir)
    merge_code_summary_sbt_ast(outdir)


def sample_corpus(corpus, sample_num, mapping):
    corpus_size = len(corpus)
    random.seed(0)
    random_res = [random.randint(0, corpus_size - 1) for _ in range(sample_num)]
    # print( random_res)
    samples = [corpus[idx] for idx in random_res]
    samples_mapping = [mapping[idx] for idx in random_res]
    return samples, samples_mapping, random_res


def process_ast(args, outdir, all_codes):
    start_time = time.perf_counter()
    logger.info("***** Generate ast *****")
    for partition in ["train", "val", "test"]:
        if os.path.exists(os.path.join(outdir, partition, "tmp", "ast.json")):
            continue
        code_dict = all_codes[partition]
        codes = list(code_dict.values())
        ast_mapping= list(code_dict.keys())
        cores = cpu_count()
        pool = Pool(cores)
        results = pool.map(gen_ast, codes)
        pool.close()
        pool.join()
        ast_dict = {idx: results[idx] for idx in range(len(codes)) if results[idx]}
        # ast_mapping = list(ast_dict.keys())
        if ast_mapping:
            ast_mapping = [ast_mapping[idx] for idx in list(ast_dict.keys())]
        else:
            ast_mapping = list(ast_dict.keys())
        ast = [item for item in list(ast_dict.values())]

        save_json_data(os.path.join(outdir, partition, "tmp"), "ast.json", ast)
        save_json_data(os.path.join(outdir, partition, "tmp"), "ast.mapping", ast_mapping)
        # word count
        if partition == "train":
            logger.info("count ast tokens")
            code_word_count = count_word_parallel(get_all_tokens(ast))
            save_json_data(os.path.join(outdir, partition, "word_count"), "ast_word_count.json",
                           code_word_count)
        logger.info("%d/%d is save in %s ast" % (len(ast), len(codes), partition))
    logger.info("time cost %s" % time_format(time.perf_counter() - start_time))


def process_sbt(args, outdir, codes):
    if args.sbt_type in [1, 2]:
        processing_sbt_using_javalang(args, outdir, codes)
    else:
        raise RuntimeError(
            "Unspport sbt type %s 0:get_sbt_using_srcml; 1:get_sbt_ao_using_javalang,2:get_sbt_using_javalang" % args.sbt_type)


def processing_sbt_using_javalang(args, outdir, all_codes):
    logger.info("***** Generate SBT *****")
    for partition in ["train", "val", "test"]:
        if os.path.exists(os.path.join(outdir, partition, "tmp", "sbt.json")):
            continue

        start_time = time.perf_counter()
        if not os.path.exists(os.path.join(args.java_files_dir, partition)):
            os.makedirs(os.path.join(args.java_files_dir, partition))

        code_dict = all_codes[partition]
        codes = list(code_dict.values())
        sbt_mapping = list(code_dict.keys())

        cores = cpu_count()
        pool = Pool(cores)
        # results = pool.map(get_sbt, codes)
        if args.sbt_type == 1:
            # types = ["SBT_AO"] * len(codes)
            get_sbt = partial(get_sbt_using_javalang, type="SBT_AO")
        elif args.sbt_type == 2:
            get_sbt = partial(get_sbt_using_javalang, type="SBT")
            # types = ["SBT"] * len(codes)
        else:
            raise RuntimeError("Unsupporting sbt_type %d" % args.sbt_type)

        results = pool.map(get_sbt, codes)
        pool.close()
        pool.join()
        sbt_dict = {idx: results[idx] for idx in range(len(codes)) if results[idx]}

        if sbt_mapping:
            sbt_mapping = [sbt_mapping[idx] for idx in list(sbt_dict.keys())]
        else:
            sbt_mapping = list(sbt_dict.keys())
        # getting sbt token
        sbt = [item for item in list(sbt_dict.values())]

        save_json_data(os.path.join(outdir, partition, "tmp"), "sbt.json", sbt)
        save_json_data(os.path.join(outdir, partition, "tmp"), "sbt.mapping", sbt_mapping)

        # word count
        if partition == "train":
            logger.info("count sbt tokens")
            sbt_word_count = count_word_parallel(get_all_tokens(sbt))
            save_json_data(os.path.join(outdir, partition, "word_count"), "sbt_word_count.json",
                           sbt_word_count)
        logger.info("%d/%d is save in %s sbt" % (len(sbt), len(codes), partition))
        logger.info("time cost %s" % time_format(time.perf_counter() - start_time))


def process_code(args, outdir, is_djl, is_dr, is_dsi, is_dfp, is_dlc, codes):

    start_time = time.perf_counter()

    # code data processing
    logger.info("***** Process code *****")
    for partition in ["train", "val", "test"]:
        if os.path.exists(os.path.join(outdir, partition, "tmp", "code.json")):
            continue
        code = list(codes[partition].values())
        mapping = list(codes[partition].keys())
        if is_djl:
            cores = cpu_count()
            pool = Pool(cores)
            results = pool.map(tokenize_source_code, code)
            pool.close()
            pool.join()
            code_dict = {idx: results[idx] for idx in range(len(results)) if results[idx]}
            code = list(code_dict.values())
            mapping =[mapping[idx] for idx in code_dict]
        if is_dr:
            code = [code_tokens_replace_str_num(item) for item in code]
        if is_dsi:
            code = [code_tokens_split_identifier(item) for item in code]
        if is_dfp:
            code = [filter_punctuation_pl(item) for item in code]
        if is_dlc:
            code = [lower_case_str_arr(item) for item in code]

        save_json_data(os.path.join(outdir, partition, "tmp"), "code.json", code)
        if not mapping:
            mapping = list(range(0, len(code)))
        save_json_data(os.path.join(outdir, partition, "tmp"), "code.mapping", mapping)
        # word count
        if partition == "train":
            logger.info("count code tokens")
            code_word_count = count_word_parallel(get_all_tokens(code))
            save_json_data(os.path.join(outdir, partition, "word_count"), "code_word_count.json",
                           code_word_count)
        logger.info("%d/%d is save in %s code" % (len(code), len(code), partition))

    logger.info("time cost %s" % time_format(time.perf_counter() - start_time))


def process_summary(args, outdir, is_cfp, is_csi, is_clc, summaries):
    start_time = time.perf_counter()
    logger.info("***** Process summary *****")
    for partition in ["train", "val", "test"]:
        if os.path.exists(os.path.join(outdir, partition, "tmp", "summary.json")):
            continue

        summary = list(summaries[partition].values())
        mapping = list(summaries[partition].keys())
        if is_cfp:
            summary = [filter_punctuation_pl(item) for item in summary]
        if is_csi:
            summary = [code_tokens_split_identifier(item) for item in summary]
        if is_clc:
            summary = [lower_case_str_arr(item) for item in summary]

        # summary data saving
        save_json_data(os.path.join(outdir, partition, "tmp"), "summary.json", summary)
        if not mapping:
            mapping = list(range(0, len(summary)))
        save_json_data(os.path.join(outdir, partition, "tmp"), "summary.mapping", mapping)

        # word count
        if partition == "train":
            logger.info("count summary tokens")
            summary_word_count = count_word_parallel(get_all_tokens(summary))
            save_json_data(os.path.join(outdir, partition, "word_count"), "summary_word_count.json", summary_word_count)
        logger.info("%d/%d is save in %s summary" % (len(summary), len(summary), partition))
    logger.info("time cost %s" % time_format(time.perf_counter() - start_time))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_filename', type=str, default="original/data.pkl", help="The path of raw dataset")
    parser.add_argument('-java_files_dir', type=str, default="java_files", help="The directory to save extracted Java file")
    parser.add_argument('-output_dir', type=str, default="processed", help="The directory to save processed files")
    # code processing  / 5
    parser.add_argument('-djl', action='store_true', help="Parse source code using javalang")
    parser.add_argument('-dfp', action='store_true', help="Filter punctuation in code tokens")
    parser.add_argument('-dsi', action='store_true', help="Split identifiers according to camelCase and snake_case")
    parser.add_argument('-dlc', action='store_true', help="Lowercase code tokens")
    parser.add_argument('-dr', action='store_true', help="Replace string and number witt generic symbols <STRING> and <NUM> in code tokens")
    # summary processing/3
    parser.add_argument('-cfp', action='store_true', help="Filter punctuation in summaries")
    parser.add_argument('-csi', action='store_true', help="Split summary token according to camelCase and snake_case")
    parser.add_argument('-clc', action='store_true', help="Lowercase summary tokens")
    parser.add_argument('-sbt_type', default=2, choices=[1, 2], type=int,
                        help=" 1：SBT_AO; 2:SBT")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))
    process(args)


if __name__ == '__main__':
    main()
