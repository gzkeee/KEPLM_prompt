import json
import random
from util import save_file, load_file, load_json, save_json
from transformers import BertTokenizer
import copy
from tqdm import tqdm
import torch
import torch.utils.data as Data
from config import Config
from util import sql_rel
from multiprocessing import Pool

data_path = 'data/dev.json'
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
rel2text = load_file('./data/rel2text')
# 处理文本数据


# 输入文本和三元组
# 文本的token数量
def get_rel_idx(token_num, triples):
    triples = copy.deepcopy(triples)
    rel_len = Config.rel_token_num
    rel2id = Config.get_rel2id()
    prefix = []
    rels = []
    # [CLS] [SEP]
    prefix.append(token_num+2)
    knowledge = ''

    for tri in triples:
        rels.append(copy.deepcopy(tri[1]))

    for tri in triples:
        tri[1] = rel2text.get(tri[1], ' ')
        prefix.append(len(tokenizer._tokenize(tri[0])))
        rel_truncate = tokenizer._tokenize(tri[1])[:rel_len]
        rel = tokenizer.convert_tokens_to_string(rel_truncate+[' [PAD] ']*(rel_len-len(rel_truncate)))
        tri[1] = rel
        prefix.append(rel_len)

        # . 分隔triple
        prefix.append(len(tokenizer._tokenize(tri[2]))+1)
        knowledge += ' '.join(tri)+' . '

    rel_id = [-1]*Config.token_max_length*2
    for i in range(len(triples)):
        start = sum(prefix[:3*i+2])
        r = rel2id[rels[i]]
        rel_id[start:start+rel_len] = list(range(r, r+rel_len))
    rel_id = rel_id[:Config.token_max_length]
    # text = text
    know = knowledge
    # text_know = text+' [SEP] ' + knowledge
    return know, rel_id
    # for idx, i in enumerate(tokenizer._tokenize(text_know)):
    #     if rel_id[idx]>=0:
    #         print(i)



# 接受json格式数据  返回text [SEP] positive triple [SEP]
def process_json_item(item):
    text = tokenizer.convert_tokens_to_string(item['token'])
    return text

# 接受json格式数据  返回negative triple
def decode_json_item(item):
    triple = {}
    subj = item['token'][item['subj_start']:item['subj_end'] + 1]
    subj = tokenizer.convert_tokens_to_string(subj)
    subj = (subj, item['subj_label'])
    obj = item['token'][item['obj_start']:item['obj_end'] + 1]
    obj = tokenizer.convert_tokens_to_string(obj)
    obj = (obj, item['obj_label'])
    relation = item['relation']

    triple['subj'] = subj
    triple['obj'] = obj
    triple['relation'] = relation
    return triple



# 获取文本和文本对应的三元组
def get_text():
    text_and_triple = load_json('data/dev.json')
    text_file = open('./data/dev_data/dev_text.txt', 'w', encoding='utf-8')
    for i in tqdm(text_and_triple):
        # print(i)
        text = process_json_item(i)
        text_file.write(text)
        text_file.write('\n')
    text_file.close()


# 获取三元组
def get_triples():
    text_and_triple = load_json('data/dev.json')
    all_triples = []
    for i in tqdm(text_and_triple):
        triple = decode_json_item(i)
        all_triples.append(triple)
    save_json(all_triples, './data/dev_data/triples.json')


# 统计三元组中关系的分布
def cal_rel_num():
    rel_num = {}
    triples = load_file('./data/triples')
    for tri in triples:
        rel_num[tri[1]] = rel_num.get(tri[1], 0) + 1

    b = json.dumps(rel_num)
    f2 = open('rel2num.json', 'w')
    f2.write(b)
    f2.close()
    # json.dump(rel_num, 'rel2num.json')


 # 将头尾实体转化为 根据ent2text转化为文本
            # pos_triple['subj'] = ent2text.get(pos_triple['subj'][1], [pos_triple['subj'][0]])
            # pos_triple['subj'] = random.sample(pos_triple['subj'], 1)[0]
            #
            # pos_triple['obj'] = ent2text.get(pos_triple['obj'][1], [pos_triple['obj'][0]])
            # pos_triple['obj'] = random.sample(pos_triple['obj'], 1)[0]

# 获取训练数据
# 50% mask掉句子中的subj和obj
# 50% 只mask掉subj或obj
def get_train_sample(neg_num, i_mp, n_mp):
    text = open('./data/text.txt', 'r', encoding='utf-8')
    all_triples = load_json('./data/dev_data/triples.json')
    # ent2text = load_json('./data/dev_data/ent2text')
    token_num2ent = load_json('./data/dev_data/token_num2ent.json')

    input_ids = []
    token_type_ids = []
    attention_mask = []
    rels_ids = []
    label = []

    text = list(text)
    text = text[i_mp::n_mp]
    all_triples = all_triples[i_mp::n_mp]

    for idx, sen in enumerate(tqdm(text, desc=f'process {i_mp}')):
        strategy = random.random()
        sen = sen.strip()
        pos_t = all_triples[idx]
        pos_rel = pos_t['relation']

        strategy /= 3
        if strategy < 0.5:
            # 需要排除相同的关系
            neg_sample = random.sample(all_triples, neg_num*10)
            neg_use = [i for i in neg_sample if i['relation'] != pos_rel][:neg_num]
            # neg_use = neg_sample
            neg_use = copy.deepcopy(neg_use)

            triples = [pos_t] + neg_use
            triple_list = []
            pos_triple = copy.deepcopy(pos_t)

            triple_list.append([pos_triple['subj'][0], pos_triple['relation'], pos_triple['obj'][0]])

            subj_num = len(tokenizer._tokenize(pos_triple['subj'][0]))
            obj_num = len(tokenizer._tokenize(pos_triple['obj'][0]))

            for tri in neg_use:
                a = token_num2ent[str(subj_num)][rel2id(tri['relation'])]
                a = [' [PAD] '] if len(a)==0 else a
                b = token_num2ent[str(obj_num)][410+rel2id(tri['relation'])]
                b = [' [PAD] '] if len(b) == 0 else b
                tri['subj'] = random.sample(a, 1)[0]
                tri['obj'] = random.sample(b, 1)[0]
                triple_list.append([tri['subj'], tri['relation'], tri['obj']])
            # print(triple_list)
            triples = triple_list
            # exit()
            random.shuffle(triples)
            y_t, y_k, rel_id = get_rel_idx(sen, triples)
            y = y_t+y_k

            x = y_t.replace(pos_t['subj'][0], ' [MASK] ' * len(tokenizer._tokenize(pos_t['subj'][0])), 1)
            x = x.replace(pos_t['obj'][0], ' [MASK] ' * len(tokenizer._tokenize(pos_t['obj'][0])), 1)
            x += y_k


        inputs = tokenizer(x, max_length=Config.token_max_length, padding='max_length', truncation=True)

        # 在mask 10% 的token
        sep_idx = inputs.input_ids.index(tokenizer.sep_token_id)
        # [CLS] 不该被包含
        mask_idx = list(range(1, sep_idx))
        random.shuffle(mask_idx)
        mask_idx = mask_idx[:int(sep_idx/10)]
        for i in mask_idx:
            inputs.input_ids[i] = tokenizer.mask_token_id
        # print(inputs.input_ids)
        lab = tokenizer(y, max_length=Config.token_max_length, padding='max_length', truncation=True)

        input_ids.append(inputs.input_ids)
        token_type_ids.append(inputs.token_type_ids)
        attention_mask.append(inputs.attention_mask)
        label.append(lab.input_ids)
        rels_ids.append(rel_id)

    save_file([torch.tensor(input_ids),
               torch.tensor(token_type_ids),
               torch.tensor(attention_mask),
               torch.tensor(label), torch.tensor(rels_ids)], f'./data/dev_data/agg/train_new_{i_mp}')





# 是否添加噪音
def mask_data(text, pos, neg, add_noise=False):
    pos = copy.deepcopy(pos)
    neg = copy.deepcopy(neg)
    # 若strategy<0.5，则mask掉subj，否则mask掉obj
    # 若mask subj 则需要替换neg的obj与pos的obj一致


    strategy = random.random()
    text_mask = text
    if add_noise:
        if strategy < 0.5:
            subj_len = len(tokenizer(pos[0]).input_ids)-2
            obj_len = len(tokenizer(pos[2]).input_ids)-2
            text_mask = text_mask.replace(pos[0], '[MASK] '*subj_len)
            text_mask = text_mask.replace(pos[2], '@ '*obj_len)

            pos[2] = '@ '*obj_len
            for n in neg:
                n[2] = pos[2]
        else:
            subj_len = len(tokenizer(pos[0]).input_ids) - 2
            obj_len = len(tokenizer(pos[2]).input_ids) - 2
            text_mask = text_mask.replace(pos[2], '[MASK] '*obj_len)
            text_mask = text_mask.replace(pos[0], '@ '*subj_len)

            pos[0] = '@ '*subj_len
            for n in neg:
                n[0] = pos[0]

    else:
        if strategy < 0.5:
            token_len = len(tokenizer(pos[0]).input_ids)-2
            text_mask = text_mask.replace(pos[0], '[MASK] '*token_len)
            for n in neg:
                n[2] = pos[2]
        else:
            token_len = len(tokenizer(pos[2]).input_ids)-2
            text_mask = text_mask.replace(pos[2], '[MASK] '*token_len)
            for n in neg:
                n[0] = pos[0]

    # print(pos)
    # print(neg)
    triples = [pos]+neg

    random.shuffle(triples)
    # print(triples)
    rels = []
    for tri in triples:
        rels.append(tri[1])
        tri[1] = '#'

    knowledge = '[SEP] '
    for k in triples:
        knowledge += ' '.join(k)
        knowledge += '. '

    # print(text_mask+" "+knowledge+ '[SEN]' +text+" "+knowledge, rels)
    # exit()

    return text_mask+" "+knowledge+'[SEN]'+text+" "+knowledge, rels


def mask_data_only_triple(pos_, neg_):
    pos = copy.deepcopy(pos_)
    neg = copy.deepcopy(neg_)
    # 若strategy<0.5，则mask掉subj，否则mask掉obj
    # 若mask subj 则需要替换neg的obj与pos的obj一致


    strategy = random.random()
    # toke
    if strategy < 0.5:
        token_len = len(tokenizer(pos[0]).input_ids)-2
        pos[0] = '[MASK] '*token_len
        for n in neg:
            n[2] = pos[2]
    else:
        token_len = len(tokenizer(pos[2]).input_ids)-2
        pos[2] = '[MASK] '*token_len
        for n in neg:
            n[0] = pos[0]

    # print(pos)
    # print(neg)
    triples = [pos]+neg
    labels = [copy.deepcopy(pos_)]+neg

    random.shuffle(triples, lambda :strategy)
    random.shuffle(triples, lambda :strategy)
    # print(triples)
    rels = []
    for tri in triples:
        rels.append(tri[1])
        tri[1] = '#'

    for tri in labels:
        # rels.append(tri[1])
        tri[1] = '#'

    knowledge = ''
    for k in triples:
        knowledge += ' '.join(k)
        knowledge += '. '

    label_text = ''
    for k in labels:
        label_text += ' '.join(k)
        label_text += '. '

    # print(text_mask+" "+knowledge+ '[SEN]' +text+" "+knowledge, rels)
    # exit()
    # print(knowledge)
    # print(knowledge+ '[SEN]' +label_text, rels)
    # exit(0)

    return knowledge+ '[SEN]' +label_text, rels

# 将文本数据转化为token id
def tokenizer_txt():
    f = open('./data/train_only_triple.txt', 'r', encoding='utf-8')
    rel = open('./data/relations.json', 'r')
    rel2id = json.load(rel)
    input_ids = []
    token_type_ids = []
    attention_mask = []
    label = []

    all_rel = []
    print(rel)
    for idx, item in tqdm(enumerate(f)):
        item = item.strip()
        # 文本
        if idx % 2 == 0:

            pair = item.split('[SEN]')
            inputs = tokenizer(pair[0], max_length=64, padding='max_length', truncation=True)
            input_ids.append(inputs.input_ids)
            token_type_ids.append(inputs.token_type_ids)
            attention_mask.append(inputs.attention_mask)
            pair2 = tokenizer(pair[1], max_length=64, padding='max_length', truncation=True)
            label.append(pair2.input_ids)


        else:
            rels = item.split()
            rels = [rel2id[r] for r in rels]
            all_rel.append(rels)
            # print(rels)
            # break
    save_file([torch.tensor(input_ids),
               torch.tensor(token_type_ids),
               torch.tensor(attention_mask),
               torch.tensor(label),
               torch.tensor(all_rel)], './data/train_data_only_triple')


# 读取实体对应文本
def create_ent2text():
    f = open('./data/wikidata5m_entity.txt', 'r', encoding='utf-8')
    ent2text = {}
    for line in tqdm(f):
        line = line.strip()
        items = line.split('\t')
        ent2text[items[0]] = items[1:5]

    save_json(ent2text, './data/dev_data/ent2text')
    # print(ent2text)

def create_ent2tri():
    f = open('./data/wikidata5m_inductive_train.txt', 'r', encoding='utf-8')
    ent2tri = {}
    for line in f:
        line = line.strip()
        items = line.split('\t')
        # print(items)
        ent2tri[items[0]] = ent2tri.get(items[0], [])+[items]
        # print(ent2tri)
        # exit()
        # ent2text[items[0]] = items[1]

    save_file(ent2tri, './data/ent2tri')
    print(ent2tri)

def create_rel2tri():
    f = list(open('./data/wikidata5m_inductive_train.txt', 'r', encoding='utf-8'))[::20]
    rel2tri = {}
    for line in tqdm(f):
        line = line.strip()
        items = line.split('\t')
        # print(items)
        rel2tri[items[1]] = rel2tri.get(items[1], [])+[items]
        # print(ent2tri)
        # exit()
        # ent2text[items[0]] = items[1]

    save_json(rel2tri, './data/rel2tri')
    print(rel2tri)

# 计算双向关系
def cal_undir_rel():
    cur = Config.get_cur()
    def cal_inter(tris):
        head = set()
        tail = set()
        for t in tris:
            head.add((t[0], t[2]))
            tail.add((t[2], t[0]))
        return len(head & tail)/(len(head | tail)+1)

    f = open('./data/wikidata5m_relation.txt', 'r', encoding='utf-8')
    for i in f:
        rel = i.split()[0]
        tris = sql_rel(cur, rel)
    # rel2tri = load_json('./data/rel2tri')
    # for k, v in rel2tri.items():
    #     print(rel)
        print(f"{rel}:{cal_inter(tris)}")

# 根据 token_num 将实体组织起来
def f():
    all_triples = load_json('./data/dev_data/triples.json')
    token_num2ent = {str(i): [[] for _ in range(410*2)] for i in range(40)}
    for t in tqdm(all_triples):
        rel_id = rel2id(t['relation'])
        head = t['subj'][0]
        head_num = len(tokenizer._tokenize(head))
        tail = t['obj'][0]
        tail_num = len(tokenizer._tokenize(tail))
        token_num2ent[str(head_num)][rel_id].append(head)
        token_num2ent[str(tail_num)][rel_id+410].append(tail)
    save_json(token_num2ent, './data/dev_data/token_num2ent.json')


def get_rel2ent():
    all_rels = list(load_json('./data/relations.json').keys())
    all_triples = load_json('./data/dev_data/triples.json')
    rel2ent = {rel: [[], []] for rel in all_rels}
    for t in tqdm(all_triples):
        rel = t['relation']
        head = t['subj'][0]
        tail = t['obj'][0]
        rel2ent[rel][0].append(head)
        rel2ent[rel][1].append(tail)
    for rel in all_rels:
        rel2ent[rel][0] = list(set(rel2ent[rel][0]))
        rel2ent[rel][1] = list(set(rel2ent[rel][1]))
    save_json(rel2ent, './data/dev_data/rel2ent.json')


import os
def agg_data(path='./data/dev_data', num=6):
    p = f'{path}/agg/'
    data = [[] for i in range(num)]
    print(len(data))
    for i in os.listdir(p):
        data_p = os.path.join(p, i)
        print(data_p)
        item = load_file(data_p)
        print(len(item))
        for j in range(len(item)):
            data[j].append(item[j])
    print(len(data))
    for j in range(len(item)):
        data[j] = torch.cat(data[j])
        print(data[j].size())
    save_file(data, f'{path}/dev')


# 测试T_REX原始数据
def test_t_res():
    data = load_json('./data/re-nlg_0-10000.json')
    relation = set()
    print(len(data))
    for d in data:
        for j in d['triples']:
            rel = j['predicate']['uri'].split('/')[-1]
            relation.add(rel)

    print(len(relation))


def load_t_res():
    data = load_json('./data/train.json')



if __name__ == '__main__':
    # cal_undir_rel()
    # test_t_res()
    load_t_res()