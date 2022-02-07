#!/usr/bin/env python
# !-*-coding:utf-8 -*-
import os
import time
import json
import logging.config
from collections import Counter
from multiprocessing import cpu_count, Pool
import math
import re
from spiral import safe_simple_split
import subprocess
import argparse
import javalang
import collections
import traceback

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


def write_source_code_to_java_file(path, method_id, method):
    java_path = os.path.join(path, str(method_id) + ".java")
    with open(java_path, "w") as f:
        f.write(method)
    # return java_path


def split_by_whitespace(s):
    return s.split(" ")


def save_json_data(data_dir, filename, data):
    os.makedirs(data_dir, exist_ok=True)
    file_name = os.path.join(data_dir, filename)
    with open(file_name, 'w') as output:
        if type(data) == list:
            if type(data[0]) in [str, list]:
                for item in data:
                    output.write(json.dumps(item))
                    output.write('\n')

            else:
                json.dump(data, output)
        elif type(data) == dict:
            json.dump(data, output)
        else:
            raise RuntimeError('Unsupported type: %s' % type(data))
    logger.info("saved dataset in " + file_name)


def get_all_tokens(data):
    all_tokens = []
    for seq in data:
        all_tokens.extend(seq)
    return all_tokens


def array_split(original_data, core_num):
    data = []
    total_size = len(original_data)
    per_core_size = math.ceil(total_size / core_num)
    for i in range(core_num):
        lower_bound = i * per_core_size
        upper_bound = min((i + 1) * per_core_size, total_size)
        data.append(original_data[lower_bound:upper_bound])
    return data


def count_word_parallel(word_list):
    start = time.perf_counter()
    cores = cpu_count()
    pool = Pool(cores)
    word_split = array_split(word_list, cores)
    word_counts = pool.map(Counter, word_split)
    result = Counter()
    for wc in word_counts:
        result += wc
    pool.close()
    pool.join()
    logger.info("count_word_parallel time cost %s " % (time_format(time.perf_counter() - start)))
    logger.info("token_word_count length: %d" % len(dict(result.most_common())))
    return dict(result.most_common())  # return the dict sorted by frequency reversely.


def time_format(time_cost):
    m, s = divmod(time_cost, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)


def load_summary(filename):
    with open(filename, 'r') as fp:
        data = fp.readlines()
    summary_tokens = [item[:-1].split("\t")[1].split() for item in data]
    return summary_tokens


def filter_punctuation_pl(sequence):
    tokens = []
    for s in sequence:
        # https://www.jianshu.com/p/4f476942dca8
        # https://stackoverflow.com/questions/5843518/remove-all-special-characters-punctuation-and-spaces-from-string
        s = re.sub('\W+', '', s).replace("_", '')
        if s:
            tokens.append(s)
    return tokens


def code_tokens_split_identifier(sequence):
    tokens = []
    for s in sequence:
        # sub_sequence = [tok for tok in ronin.split(s) if tok]
        sub_sequence = [tok for tok in safe_simple_split(s) if tok]
        tokens.extend(sub_sequence)
    return tokens


def lower_case_str_arr(str_arr):
    return [tok.lower() for tok in str_arr]


def read_json_file(filename):
    with open(filename, 'r') as fp:
        data = fp.readlines()
    if len(data) == 1:
        data = json.loads(data[0])
    else:
        data = [json.loads(line) for line in data]
    return data


def get_merged_data(code, code_mapping, summary, summary_mapping):
    merged_idx = list(set(code_mapping).intersection(set(summary_mapping)))
    code_dict = dict(zip(code_mapping, code))
    summary_dict = dict(zip(summary_mapping, summary))
    merged_code = [code_dict[idx] for idx in merged_idx]
    merged_sum = [summary_dict[idx] for idx in merged_idx]
    return merged_code, merged_sum, merged_idx


def merge_sbt_summary(output_dir):
    for partition in ["train", "val", "test"]:
        file_dir = os.path.join(output_dir, partition)
        sbt = read_json_file(os.path.join(file_dir, "tmp", "sbt.json"))
        sbt_mapping = read_json_file(os.path.join(file_dir, "tmp", "sbt.mapping"))
        summary = read_json_file(os.path.join(file_dir, "tmp", "summary.json"))
        summary_mapping = read_json_file(os.path.join(file_dir, "tmp", "summary.mapping"))

        merged_sbt, merged_summary, mapping = get_merged_data(sbt, sbt_mapping, summary, summary_mapping)

        save_json_data(file_dir, "sbt.json", merged_sbt)
        save_json_data(file_dir, "summary.json", merged_summary)
        save_json_data(file_dir, "mapping.json", mapping)


# https://github.com/xing-hu/EMSE-DeepCom/blob/master/data_utils/get_ast.py
def get_name(obj):
    if (type(obj).__name__ in ['list', 'tuple']):
        a = []
        for i in obj:
            a.append(get_name(i))
        return a
    elif (type(obj).__name__ in ['dict', 'OrderedDict']):
        a = {}
        for k in obj:
            a[k] = get_name(obj[k])
        return a
    elif (type(obj).__name__ not in ['int', 'float', 'str', 'bool']):
        return type(obj).__name__
    else:
        return obj


# https://github.com/xing-hu/EMSE-DeepCom/blob/master/data_utils/get_ast.py
def process_source(code):
    try:
        tokens = list(javalang.tokenizer.tokenize(code))
    except:
        # logger.info(code_string)
        # logger.info(10 * "*")
        with open("error.log", "a") as f:
            f.write(code)
            f.write("\n")
            f.write(traceback.format_exc())
            f.write("\n")
            f.write(20 * "*")
            f.write("\n")
        return None
    output_tokens = []
    for tk in tokens:
        if tk.__class__.__name__ == 'String' or tk.__class__.__name__ == 'Character':
            output_tokens.append('STR_')
        elif 'Integer' in tk.__class__.__name__ or 'FloatingPoint' in tk.__class__.__name__:
            output_tokens.append('NUM_')
        elif tk.__class__.__name__ == 'Boolean':
            output_tokens.append('BOOL_')
        else:
            output_tokens.append(tk.value)
    return output_tokens


# https://github.com/xing-hu/EMSE-DeepCom/blob/master/data_utils/get_ast.py
def get_ast(code):
    tokens = javalang.tokenizer.tokenize(code)
    token_list = list(javalang.tokenizer.tokenize(code))
    length = len(token_list)
    parser = javalang.parser.Parser(tokens)
    try:
        tree = parser.parse_member_declaration()
    except (javalang.parser.JavaSyntaxError, IndexError, StopIteration, TypeError):
        with open("error.log", "a") as f:
            f.write(code)
            f.write("\n")
            f.write(traceback.format_exc())
            f.write("\n")
            f.write(20 * "*")
            f.write("\n")
        return None
    flatten = []
    for path, node in tree:
        flatten.append({'path': path, 'node': node})

    format_ast = []
    for i, item in enumerate(flatten):
        format_node = collections.OrderedDict()
        path = item['path']
        node = item['node']
        children = []
        for child in node.children:
            if isinstance(child, javalang.ast.Node):
                child_path = path + tuple((node,))
                for j in range(i + 1, len(flatten)):
                    if child_path == flatten[j]['path'] and child == flatten[j]['node']:
                        children.append(j)
            if isinstance(child, list) and child:
                child_path = path + (node, child)
                for j in range(i + 1, len(flatten)):
                    if child_path == flatten[j]['path']:
                        children.append(j)
        format_node["id"] = i
        format_node["type"] = get_name(node)
        if children:
            format_node["children"] = children
        value = None
        if hasattr(node, 'name'):
            value = node.name
        elif hasattr(node, 'value'):
            value = node.value
        elif hasattr(node, 'position') and node.position:
            for i, token in enumerate(token_list):
                if node.position == token.position:
                    pos = i + 1
                    value = str(token.value)
                    while (pos < length and token_list[pos].value == '.'):
                        value = value + '.' + token_list[pos + 1].value
                        pos += 2
                    break
        elif type(node) is javalang.tree.This \
                or type(node) is javalang.tree.ExplicitConstructorInvocation:
            value = 'this'
        elif type(node) is javalang.tree.BreakStatement:
            value = 'break'
        elif type(node) is javalang.tree.ContinueStatement:
            value = 'continue'
        elif type(node) is javalang.tree.TypeArgument:
            value = str(node.pattern_type)
        elif type(node) is javalang.tree.SuperMethodInvocation \
                or type(node) is javalang.tree.SuperMemberReference:
            value = 'super.' + str(node.member)
        elif type(node) is javalang.tree.Statement \
                or type(node) is javalang.tree.BlockStatement \
                or type(node) is javalang.tree.ForControl \
                or type(node) is javalang.tree.ArrayInitializer \
                or type(node) is javalang.tree.SwitchStatementCase:
            value = 'None'
        elif type(node) is javalang.tree.VoidClassReference:
            value = 'void.class'
        elif type(node) is javalang.tree.SuperConstructorInvocation:
            value = 'super'

        if value is not None and type(value) is type('str'):
            format_node['value'] = value
        format_ast.append(format_node)

    return format_ast


# https://github.com/xing-hu/EMSE-DeepCom/blob/master/data_utils/ast_traversal.py
# type 1:SBT_AO 2:SBT
def ast2sbt(cur_root_id, node_list, type=1):
    current_root = node_list[cur_root_id]
    sbt = []
    sbt.append("(")
    if type == "SBT_AO":
        token = current_root['type']
    elif type == "SBT":
        token_value = current_root.get("value", None)
        token = (current_root['type'] + "_" + token_value) if token_value else current_root['type']

    sbt.append(token)

    if 'children' in current_root:
        chs = current_root['children']
        for ch in chs:
            sbt.extend(ast2sbt(ch, node_list, type=type))
    sbt.append(")")
    sbt.append(token)
    return sbt


def get_sbt_using_javalang(code, type):
    # type 1:SBT_AO 2:SBT
    # pre-process the source code: strings -> STR_, numbers-> NUM_, Booleans-> BOOL_

    code_tokens = process_source(code)
    # generate ast file for source code
    if code_tokens:
        ast = get_ast(" ".join(code_tokens))
    else:
        return None
    # generate sbt
    if ast:
        sbt = ast2sbt(0, ast, type=type)
    else:
        return None

    return sbt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str, default="original")
    parser.add_argument('-java_files_dir', type=str, default="/home/user/tl_codesum/java_files")
    parser.add_argument('-output_dir', type=str, default="processed")
    # summary processing/3
    parser.add_argument('-cfp', action='store_true', help="Filter punctuation in summaries")
    parser.add_argument('-csi', action='store_true', help="Split summary token according to camelCase and snake_case")
    parser.add_argument('-clc', action='store_true', help="Lowercase summary tokens")
    # sbt_type
    parser.add_argument('-sbt_type', default=2, choices=[0, 1, 2], type=int,
                        help="0:processing_sbt_using_srcml; 1:get_sbt_ao_using_javalang; 2:get_sbt_using_javalang")

    args = parser.parse_args()
    return args
