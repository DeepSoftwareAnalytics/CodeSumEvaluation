import os
import time
import json
import logging.config

from multiprocessing import cpu_count, Pool
import sys
from functools import partial
sys.path.append("../")
from utils import filter_punctuation_pl, code_tokens_split_identifier, lower_case_str_arr, save_json_data, \
    count_word_parallel, get_all_tokens, time_format,  \
    read_json_file, parse_args, load_summary, write_source_code_to_java_file,  \
    split_by_whitespace, merge_sbt_summary, get_sbt_using_javalang


sys.path.append("../")

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


def load_code(filename):
    with open(filename, 'r') as fp:
        data = fp.readlines()
    data = [json.loads(line) for line in data]
    source_codes = [item["code"] for item in data]
    return source_codes


def process(args):
    """
    processing code/summary and count the word frequency.
    """
    is_cfp = args.cfp
    is_csi = args.csi
    is_clc = args.clc
    summary_params = 'cfp{}_csi{}_cfd0_clc{}'.format(is_cfp + 0, is_csi + 0, is_clc + 0)
    sbt_params ="sbt{}".format(args.sbt_type)
    params = summary_params + "_" + sbt_params
    outdir = os.path.join(args.output_dir, params)

    # process summary
    logger.info("***** Process summary *****")
    for partition in ["train", "valid", "test"]:
        if os.path.exists(os.path.join(outdir, partition, "tmp", "summary.json")):
            continue
        if partition == "valid" and os.path.exists(os.path.join(outdir, "val", "tmp", "summary.json")):
            continue
        start_time = time.perf_counter()
        summary = load_summary(os.path.join(args.data_dir, partition, "%s.token.nl" % partition))
        if is_cfp:
            summary = [filter_punctuation_pl(item) for item in summary]
        if is_csi:
            summary = [code_tokens_split_identifier(item) for item in summary]
        if is_clc:
            summary = [lower_case_str_arr(item) for item in summary]
        # summary data saving
        if partition == "valid":
            partition = "val"
        save_json_data(os.path.join(outdir, partition, "tmp"), "summary.json", summary)
        summary_mapping = list(range(0, len(summary)))
        save_json_data(os.path.join(outdir, partition, "tmp"), "summary.mapping", summary_mapping)
        # word count
        if partition == "train":
            logger.info("count summary tokens")
            summary_word_count = count_word_parallel(get_all_tokens(summary))
            save_json_data(os.path.join(outdir, partition, "word_count"), "summary_word_count.json", summary_word_count)
        logger.info("%d/%d is save in %s summary" % (len(summary), len(summary), partition))
        logger.info("time cost %s" % time_format(time.perf_counter() - start_time))

    # process sbt.
    processing_sbt(args, outdir)

    start_time = time.perf_counter()
    merge_sbt_summary(outdir)
    logger.info("time cost %s" % time_format(time.perf_counter() - start_time))


def processing_sbt(args, outdir):
    if args.sbt_type in [1, 2]:
        processing_sbt_using_javalang(args, outdir)
    else:
        raise RuntimeError("Unspport sbt type %s 0:get_sbt_using_srcml; 1:get_sbt_ao_using_javalang" %args.sbt_type)


def processing_sbt_using_javalang(args, outdir):
    logger.info("***** Generate SBT *****")
    for partition in ["train", "valid", "test"]:
        if os.path.exists(os.path.join(outdir, partition, "tmp", "sbt.json")):
            continue
        if partition == "valid" and os.path.exists(os.path.join(outdir, "val", "tmp", "sbt.json")):
            continue

        start_time = time.perf_counter()
        if not os.path.exists(os.path.join(args.java_files_dir, partition)):
            os.makedirs(os.path.join(args.java_files_dir, partition))

        # load code
        codes = load_code(os.path.join(args.data_dir, partition, "%s.json" % partition))
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
            raise RuntimeError("Unsupporting sbt_type %d" %args.sbt_type)

        results = pool.map(get_sbt, codes)
        pool.close()
        pool.join()
        sbt_dict = {idx: results[idx] for idx in range(len(codes)) if results[idx]}
        sbt_mapping = list(sbt_dict.keys())
        # getting sbt token
        sbt = [item for item in list(sbt_dict.values())]

        if partition == "valid":
            partition = "val"
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


def main():
    args = parse_args()
    logger.info(args)
    process(args)


if __name__ == '__main__':
    main()
