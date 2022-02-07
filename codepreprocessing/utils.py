
# !/usr/bin/env python
# !-*-coding:utf-8 -*-
import os
import time
import json
import logging.config
from collections import Counter
from multiprocessing import cpu_count, Pool
import itertools
import math
import re
import javalang
import traceback
import subprocess
import collections
import pickle
from spiral import ronin, safe_simple_split

logging.basicConfig(format='[%(asctime)s - %(levelname)s - %(name)s ] %(message)s',
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


def filter_punctuation_pl(sequence):
    tokens = []
    for s in sequence:
        # https://www.jianshu.com/p/4f476942dca8
        # https://stackoverflow.com/questions/5843518/remove-all-special-characters-punctuation-and-spaces-from-string
        s = re.sub('\W+', '', s).replace("_", '')
        if s:
            tokens.append(s)
    return tokens


def lower_case_str_arr(str_arr):
    return [tok.lower() for tok in str_arr]


def code_tokens_split_identifier(sequence):
    tokens = []
    for s in sequence:
        sub_sequence = [tok for tok in safe_simple_split(s) if tok]
        tokens.extend(sub_sequence)
    return tokens


def tokenize_source_code(code_string):
    """
    Generate a list of string after javalang tokenization.
    :param code_string: a string of source code
    :return:
    """
    code_string.replace("#", "//")
    try:
        tokens = list(javalang.tokenizer.tokenize(code_string))
        return [token.value for token in tokens]

    except:
        # logger.info(code_string)
        # logger.info(10 * "*")
        # with open("error.log", "a") as f:
        #     f.write(code_string)
        #     f.write("\n")
        #     f.write(traceback.format_exc())
        #     f.write("\n")
        #     f.write(20 * "*")
        #     f.write("\n")
        return None


def code_tokens_replace_str_num(sequence):
    tokens = []
    for s in sequence:
        if s[0] == '"' and s[-1] == '"':
            tokens.append("<STRING>")
        elif s.isdigit():
            tokens.append("<NUM>")
        else:
            tokens.append(s)
    return tokens


#
#
# def get_all_tokens(data):
#     # data is a dict: {idx: string_seq, ....}
#     all_tokens = []
#     for seq in data.values():
#         # tokens = seq.split(" ")
#         # tokens = list(filter(lambda x: x,   tokens))
#         all_tokens.extend(seq)
#     return all_tokens


def count_word(token_word_count, tokens):
    for token in tokens:
        token_word_count[token.lower()] = token_word_count.get(token.lower(), 0) + 1
    return token_word_count


def extract_col(data, col):
    return [d[col] for d in data.values()]


def extract(data):
    return [d for d in data.values()]


# li = [[1, 2, 3], [4, 5, 6], [7], [8, 9]] -> [1,2,3,4,5,6,7,8,9]
def list_flatten(li):
    flatten = itertools.chain.from_iterable
    return list(flatten(li))


def array_split(original_data, core_num):
    data = []
    total_size = len(original_data)
    per_core_size = math.ceil(total_size / core_num)
    for i in range(core_num):
        lower_bound = i * per_core_size
        upper_bound = min((i + 1) * per_core_size, total_size)
        data.append(original_data[lower_bound:upper_bound])
    return data


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


def read_json_file(filename):
    with open(filename, 'r') as fp:
        data = fp.readlines()
    if len(data) == 1:
        data = json.loads(data[0])
    else:
        data = [json.loads(line) for line in data]
    return data


def time_format(time_cost):
    m, s = divmod(time_cost, 60)
    h, m = divmod(m, 60)
    # print("time_cost: %d" % (time_cost))
    return "%02d:%02d:%02d" % (h, m, s)


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


def get_all_tokens(data):
    all_tokens = []
    for seq in data:
        all_tokens.extend(seq)
    return all_tokens


def load_summary(filename):
    with open(filename, 'r') as fp:
        data = fp.readlines()
    summary_tokens = [item[:-1].split("\t")[1].split() for item in data]
    return summary_tokens


def get_merged_data(code, code_mapping, summary, summary_mapping, sbt, sbt_mapping, ast, ast_mapping):
    merged_idx = list(set(code_mapping).intersection(set(summary_mapping)).intersection(set(sbt_mapping)).intersection(
        set(ast_mapping)))
    code_dict = dict(zip(code_mapping, code))
    summary_dict = dict(zip(summary_mapping, summary))
    sbt_dict = dict(zip(sbt_mapping, sbt))
    ast_dict = dict(zip(ast_mapping, ast))

    code_mapping_counter = dict(Counter(code_mapping))
    summary_mapping_counter = dict(Counter(summary_mapping))
    sbt_mapping_counter = dict(Counter(sbt_mapping))
    ast_mapping_counter = dict(Counter(ast_mapping))

    merged_code = [code_dict[idx] for idx in merged_idx for _ in range(code_mapping_counter[idx])]
    merged_sum = [summary_dict[idx] for idx in merged_idx for _ in range(summary_mapping_counter[idx])]
    merged_sbt = [sbt_dict[idx] for idx in merged_idx for _ in range(sbt_mapping_counter[idx])]
    merged_ast = [ast_dict[idx] for idx in merged_idx for _ in range(ast_mapping_counter[idx])]
    merged_mappping = [idx for idx in merged_idx for _ in range(code_mapping_counter[idx])]

    return merged_code, merged_sum, merged_sbt, merged_ast, merged_mappping


def merge_code_summary_sbt_ast(output_dir):
    start_time = time.perf_counter()
    for partition in ["train", "val", "test"]:
        file_dir = os.path.join(output_dir, partition)
        code = read_json_file(os.path.join(file_dir, "tmp", "code.json"))
        code_mapping = read_json_file(os.path.join(file_dir, "tmp", "code.mapping"))

        summary = read_json_file(os.path.join(file_dir, "tmp", "summary.json"))
        summary_mapping = read_json_file(os.path.join(file_dir, "tmp", "summary.mapping"))

        sbt = read_json_file(os.path.join(file_dir, "tmp", "sbt.json"))
        sbt_mapping = read_json_file(os.path.join(file_dir, "tmp", "sbt.mapping"))

        ast = read_json_file(os.path.join(file_dir, "tmp", "ast.json"))
        ast_mapping = read_json_file(os.path.join(file_dir, "tmp", "ast.mapping"))

        merged_code, merged_summary, merged_sbt, merged_ast, mapping = get_merged_data(code, code_mapping, summary,
                                                                                       summary_mapping, sbt,
                                                                                       sbt_mapping, ast, ast_mapping)

        save_json_data(file_dir, "code.json", merged_code)
        save_json_data(file_dir, "summary.json", merged_summary)
        save_json_data(file_dir, "sbt.json", merged_sbt)
        save_json_data(file_dir, "ast.json", merged_ast)
        save_json_data(file_dir, "mapping.json", mapping)

        logger.info("%d samples are save in %s " % (len(mapping), partition))

    logger.info("merge code time cost %s" % time_format(time.perf_counter() - start_time))


def write_source_code_to_java_file(path, method_id, method):
    java_path = os.path.join(path, str(method_id) + ".java")
    with open(java_path, "w", encoding="utf-8") as f:
        f.write(method)
    # return java_path


def split_by_whitespace(s):
    return s.split(" ")


def process_sbt_token(token, xml_tokens, index, node_type):
    continue_flag = False
    terminal_node_flag = False

    # Replacing '<type' with '(type' for non-terminal
    # Replacing '<\type' with 'type)' for non-terminal
    if node_type == "non_terminal":
        if token[:2] == '</':
            new_token = token.replace('</', ') ')
        else:
            new_token = token.replace('<', '( ')
    elif node_type == "terminal":
        new_token = ') ' + xml_tokens[index - 1].replace('<', '') + '_' + token
        terminal_node_flag = True
    else:
        literal_type = {
            'number': '<NUM>',
            'string': '<STR>',
            'null': 'null',
            'char': '<STR>',
            'boolean': token
        }
        new_token = ') ' + xml_tokens[index - 2].replace('<', '') + '_' + node_type + '_' + literal_type[
            node_type]
        terminal_node_flag = True

    return new_token, terminal_node_flag


def verify_node_type(token, xml_tokens, index, literal_list):
    """
    non_terminal:
        <type>
        </type>
    terminal:
        <type> token </type>  (this is right)
        <type> token <value> ... ( this is wrong)
    literal:
        <literal type="String" token </literal> | number | char | string | null | boolean
    """
    try:
        if token[0] == '<':
            node_type = 'non_terminal'
        elif xml_tokens[index - 1][1:] == xml_tokens[index + 1][2:] and xml_tokens[index - 1][0] == '<':
            node_type = 'terminal'
        elif xml_tokens[index - 1][:5] == 'type=':
            token_type = xml_tokens[index - 1].replace('type=', '')
            token_type = token_type.replace('\"', '')
            if token_type in literal_list:
                node_type = token_type
            else:
                node_type = None
        else:
            node_type = None
        return node_type
    except IndexError as e:
        print(e)
        return None


# Given the AST generating by http://131.123.42.38/lmcrs/beta/ ,
# it return SBT proposed by https://xin-xia.github.io/publication/icpc182.pdf
def xml2sbt(xml):
    # Replacing '<...>' with ' <...'
    xml = xml.replace('<', ' <')
    xml = xml.replace('>', ' ')

    #  splitting xml and filtering ''
    xml_tokens = xml.split(' ')
    xml_tokens = [i for i in xml_tokens if i != '']

    sbt = []
    terminal_node_flag = False
    literal_list = ['number', 'string', 'null', 'char', 'boolean']
    for i in range(len(xml_tokens)):

        # i = i+1 is unavailable in for loop, so we set terminal_node_flag to skip
        # terminal_nodes that have already been processed
        if terminal_node_flag:
            terminal_node_flag = False
            continue
        token = xml_tokens[i]
        node_type = verify_node_type(token, xml_tokens, i, literal_list)
        if node_type:
            new_token, terminal_node_flag = process_sbt_token(token, xml_tokens, i, node_type)
            sbt.append(new_token)
        else:
            continue
    return sbt


# Obtaining the sbt of the java file
# srcml doesn't check the grammar of code.
def sbt_parser(file_path):
    if os.path.isfile(file_path):
        commandline = 'srcml ' + file_path
    else:
        commandline = 'srcml -l Java -t "{}"'.format(file_path)
    # https://docs.python.org/3/library/subprocess.html
    # Window
    # xml, _ = subprocess.Popen(commandline, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate(timeout=20)
    # xml = re.findall(r"(<unit .*</unit>)", xml.decode('utf-8'), re.S)[0]
    # sbt = xml2sbt(xml)
    # sbt = ' '.join(sbt)
    # ubutu
    try:
        xml, _ = subprocess.Popen(commandline.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate(
            timeout=10)
        xml = re.findall(r"(<unit .*</unit>)", xml.decode('utf-8'), re.S)[0]
        sbt = xml2sbt(xml)
        sbt = ' '.join(sbt)
    except:
        sbt = None
    return sbt


def preorder_traverse_tree(tree, seq):
    seq.append(str(tree))
    for i, attr in enumerate(tree.attrs):
        child = tree.children[i]
        # if child == None:
        if not child:
            continue
        elif isinstance(child, str) and child:
            seq.append(child)
        elif isinstance(child, javalang.ast.Node):
            preorder_traverse_tree(child, seq)
        elif isinstance(child, list) and child:
            for subchild in child:
                if isinstance(subchild, str):
                    seq.append(subchild)
                elif isinstance(subchild, javalang.ast.Node):
                    preorder_traverse_tree(subchild, seq)
        elif attr == "modifiers" and child:
            seq.append("Modifier")
            seq.append(list(child)[0])


def gen_ast(line):
    preorder_result = []
    code = line.strip()
    try:
        tokens = javalang.tokenizer.tokenize(code)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse_member_declaration()
        preorder_traverse_tree(tree, preorder_result)
        return preorder_result
    except:
        # with open("error.log", "a") as f:
        #     f.write(code)
        #     f.write("\n")
        #     f.write(traceback.format_exc())
        #     f.write("\n")
        #     f.write(20 * "*")
        #     f.write("\n")
        return None


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


#  Make a new directory if it is not exist.
def make_directory(path):
    is_exist = os.path.exists(path)
    if not is_exist:
        os.makedirs(path)
    else:
        pass


def time_format(time_cost):
    m, s = divmod(time_cost, 60)
    h, m = divmod(m, 60)
    # print("time_cost: %d" % (time_cost))
    return "%02d:%02d:%02d" % (h, m, s)


def save_pickle_data(path_dir, filename, data):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    with open(os.path.join(path_dir, filename), 'wb') as f:
        pickle.dump(data, f)
    print("write file to " + os.path.join(path_dir, filename))


def save_pickle_data(path_dir, filename, data):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    with open(os.path.join(path_dir, filename), 'wb') as f:
        pickle.dump(data, f)
    print("write file to " + os.path.join(path_dir, filename))


def read_pickle_data(data_path):
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

