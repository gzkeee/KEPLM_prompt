import json
import random
from util import save_file, load_file, load_json, save_json, sql_ent
import copy
from tqdm import tqdm
import torch
from config import Config
from util import rel2id, sql_head_tail
from multiprocessing import Pool, Manager
from data_process import get_rel_idx, agg_data
from transformers import BertTokenizerFast
from util import mp_data


def get_train_sample(neg_num, i_mp, n_mp):
    tokenizer = Config.get_tokenizer()
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


            subj_num = len(tokenizer._tokenize(pos_triple['subj'][0]))
            obj_num = len(tokenizer._tokenize(pos_triple['obj'][0]))
            sen = sen.replace(pos_triple['subj'][0], ' @ '*subj_num)
            triple_list.append([' @ '*subj_num, pos_triple['relation'], pos_triple['obj'][0]])

            for tri in neg_use:
                # a = token_num2ent[str(subj_num)][rel2id(tri['relation'])]
                # a = [' [PAD] '] if len(a)==0 else a
                b = token_num2ent[str(obj_num)][410+rel2id(pos_rel)]
                b = [' [PAD] '] if len(b) == 0 else b
                # tri['subj'] = random.sample(a, 1)[0]
                tri['obj'] = random.sample(b, 1)[0]
                triple_list.append([' @ '*subj_num, tri['relation'], tri['obj']])
            # print(triple_list)
            triples = triple_list
            # exit()
            random.shuffle(triples)
            y_t, y_k, rel_id = get_rel_idx(sen, triples)
            y = y_t+y_k

            # x = y_t.replace(pos_t['subj'][0], ' [MASK] ' * len(tokenizer._tokenize(pos_t['subj'][0])), 1)
            x = y_t.replace(pos_t['obj'][0], ' [MASK] ' * len(tokenizer._tokenize(pos_t['obj'][0])), 1)
            x += y_k


        inputs = tokenizer(x, max_length=Config.token_max_length, padding='max_length', truncation=True)

        # # 在mask 10% 的token
        # sep_idx = inputs.input_ids.index(tokenizer.sep_token_id)
        # # [CLS] 不该被包含
        # mask_idx = list(range(1, sep_idx))
        # random.shuffle(mask_idx)
        # mask_idx = mask_idx[:2]
        # for i in mask_idx:
        #     inputs.input_ids[i] = tokenizer.mask_token_id
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


all_rels = list(load_json('./data/relations.json').keys())
# all_triples = load_json('./data/dev_data/triples.json')
rel2ent = load_json('./data/dev_data/rel2ent.json')
def sample_negative_triple(pos_triple):
    # 完全随机采样
    def sample0(pos_t):
        neg = random.choice(all_triples)
        neg = [neg['subj'][0], neg['relation'], neg['obj'][0]]
        return neg

    # 头尾实体相同 关系不同
    def sample1(pos_t):
        neg = copy.deepcopy(pos_t)
        neg[1] = random.choice(all_rels)
        return neg

    # 关系相同 头实体或尾实体不同
    def sample2(pos_t):
        neg = copy.deepcopy(pos_t)
        r = random.random()
        if r < 0.5:
            neg[0] = random.choice(rel2ent[neg[1]][0])
        else:
            neg[2] = random.choice(rel2ent[neg[1]][1])
        return neg

    r = random.random()
    if r < 0.9:
        neg = sample0(pos_triple)
    elif r < 0.98:
        neg = sample1(pos_triple)
    else:
        neg = sample2(pos_triple)

    return neg

# 一次生成全部数据  所有负样本中头实体或者尾实体相同
def sample_full_negative_triple(pos_triple, num):
    # 头尾实体相同 关系不同
    def sample1(pos_t):
        neg = copy.deepcopy(pos_t)
        neg[1] = random.choice(set(all_rels)-set(pos_t[1]))
        return neg

    nsp = 0
    r1 = random.random()
    r2 = random.random()
    h, r, t = pos_triple
    neg_triples = []
    # 固定头实体
    if r1 < 0.5:
        for i in range(num):
            rel_ng = random.choice(all_rels)
            t_neg = random.choice(rel2ent[rel_ng][1])
            neg_triples.append([h, rel_ng, t_neg])
    # 固定尾实体
    else:
        for i in range(num):
            rel_ng = random.choice(all_rels)
            h_neg = random.choice(rel2ent[rel_ng][0])
            neg_triples.append([h_neg, rel_ng, t])

    # 引入正triple
    if r2 < 0.5:
        neg_triples.append(pos_triple)
        nsp = 0
    else:
        neg_triples.append(sample1(pos_triple))
        nsp = 1


    return neg_triples, nsp


def mask_tokens(inputs, tokenizer, ent_pos=torch.zeros(Config.token_max_length)):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    masked_indices = torch.bernoulli(torch.full(labels.shape, 0.15)).bool()
    loc = torch.argwhere(inputs==tokenizer.sep_token_id)[0, 1]
    # print(loc)
    masked_indices[0, loc:] = False
    masked_indices[0, 0] = False
    masked_indices[0, ent_pos == 1] = False
    # print(masked_indices)
    # exit()

    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    """
    对于mask_indices剩下的20% 在进行提取,取其中一半进行random 赋值,剩下一般保留原来值. 
    """
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    """
    最后返回 mask之后的input 和 label.
    inputs 为原文+Mask+radom 单词
    labels 为 1 和 -1. 其中1是Mask的位置, -1是没有mask的位置
    """
    return inputs, labels

# 通过NSP任务来学习知识
def get_train_data(neg_num, i_mp, n_mp):
    tokenizer = Config.get_tokenizer()
    text = open('./data/dev_data/dev_text.txt', 'r', encoding='utf-8')
    all_triples = load_json('./data/dev_data/triples.json')
    nsp_data = open(f'./data/data_nsp/train_{i_mp}.txt', 'w', encoding='utf-8')

    input_ids = []
    token_type_ids = []
    attention_mask = []
    rels_ids = []
    labels = []
    next_sentence_label = []

    text = list(text)

    text = text[i_mp::n_mp]
    all_triples = all_triples[i_mp::n_mp]

    for i in tqdm(range(len(text))):
        # if i==1000:
        #     break
        t = text[i]
        pos = all_triples[i]
        pos = [pos['subj'][0], pos['relation'], pos['obj'][0]]
        neg_triples = []
        # for j in range(neg_num):
        #     neg_triples.append(sample_negative_triple(pos))
        neg_triples, nsp = sample_full_negative_triple(pos, neg_num)
        # r = random.random()
        # if r <= 0.5:
        #     neg_triples[0] = pos
        #     random.shuffle(neg_triples)
        #     nsp = 0

        t, k, rel_id = get_rel_idx(t, neg_triples)
        # print(t)
        # print(k)
        # print(rel_ids)
        inputs = tokenizer(t, k, max_length=Config.token_max_length, padding='max_length', truncation=True, return_tensors='pt')
        x, y = mask_tokens(inputs.input_ids, tokenizer)

        input_ids.append(x)
        token_type_ids.append(inputs.token_type_ids)
        attention_mask.append(inputs.attention_mask)
        labels.append(y)
        rels_ids.append(rel_id)
        next_sentence_label.append(nsp)

        # print(x)
        # print(inputs.token_type_ids)
        # print(inputs.input_ids)
        # print(y)
        # print(nsp)
        # print(rel_id)
        # exit()

        nsp_data.write(t)
        nsp_data.write(k+'\n')
        nsp_data.write(str(nsp)+'\n')
        # exit()
    save_file([torch.cat(input_ids),
               torch.cat(token_type_ids),
               torch.cat(attention_mask),
               torch.cat(labels), torch.tensor(rels_ids),
               torch.tensor(next_sentence_label)], f'./data/data_nsp/agg/train_new_{i_mp}')
    nsp_data.close()




    # get_train_data()
    # get_train_sample(6, 0, 10)
    # n = int(6)
    # p = Pool(n)
    # for i in range(n):
    #     p.apply_async(get_train_sample, args=(12, i, n))
    #
    # p.close()
    # p.join()
    # agg_data()



# 获取数据0的关系标签
def f():
    all_triples = load_json('./data/dev_data/triples.json')
    all_triples = all_triples[0::6]
    print(all_triples[:10])
    relations = []
    for i in all_triples:
        relations.append(i['relation'])

    save_json(relations, './data/Entity_Shuffling/relation_label.json')



# 调换文本中实体的位置 观察预训练能否识别
def get_train_data_3(neg_num, i_mp, n_mp):
    tokenizer = Config.get_tokenizer()
    dev_json = load_json('./data/dev.json')
    # all_triples = load_json('./data/dev_data/triples.json')
    nsp_data = open(f'./data/Entity_Shuffling/train_{i_mp}.txt', 'w', encoding='utf-8')

    def get_text(item):
        subj = item['token'][item['subj_start']:item['subj_end'] + 1]
        obj = item['token'][item['obj_start']:item['obj_end'] + 1]
        r = random.random()
        label = 0
        # 调换实体位置
        if r < 0.5:
            if item['subj_start'] < item['obj_start']:
                text = item['token'][:item['subj_start']]+obj+\
                       item['token'][item['subj_end'] + 1:item['obj_start']]+subj+item['token'][item['obj_end'] + 1:]
            else:
                text = item['token'][:item['obj_start']] + subj + \
                       item['token'][item['obj_end'] + 1:item['subj_start']] + obj + item['token'][
                                                                                      item['subj_end'] + 1:]
            label = 1
        else:
            text = item['token']
            label = 0
        return tokenizer.convert_tokens_to_string(text), label

    input_ids = []
    token_type_ids = []
    attention_mask = []
    labels = []
    next_sentence_label = []

    dev_json = list(dev_json)

    dev_json = dev_json[i_mp::n_mp]
    # all_triples = all_triples[i_mp::n_mp]

    for i in tqdm(range(len(dev_json))):
        # if i==1000:
        #     break

        # nsp = 1
        t = dev_json[i]
        t, nsp = get_text(t)
        # print(t)
        # print(nsp)
        # exit()

        inputs = tokenizer(t, max_length=Config.token_max_length, padding='max_length', truncation=True, return_tensors='pt')
        x, y = mask_tokens(inputs.input_ids, tokenizer)
        input_ids.append(x)
        token_type_ids.append(inputs.token_type_ids)
        attention_mask.append(inputs.attention_mask)
        labels.append(y)
        next_sentence_label.append(nsp)

        nsp_data.write(t)
        nsp_data.write(str(nsp)+'\n')
        # exit()
    save_file([torch.cat(input_ids),
               torch.cat(token_type_ids),
               torch.cat(attention_mask),
               torch.cat(labels),
               torch.tensor(next_sentence_label)], f'./data/Entity_Shuffling/agg/train_new_{i_mp}')
    nsp_data.close()




    # get_train_data()
    # get_train_sample(6, 0, 10)
    # n = int(6)
    # p = Pool(n)
    # for i in range(n):
    #     p.apply_async(get_train_sample, args=(12, i, n))
    #
    # p.close()
    # p.join()
    # agg_data()



# 调换文本中实体的位置 并为模型判断提供三元组依据
def get_train_data_4(dev_json, i_mp, k_num=12):
    tokenizer = Config.get_tokenizer()
    cur = Config.get_cur(i_mp%8)
    nsp_data = open(f'./data/T-REX/see/{i_mp}.txt', 'w', encoding='utf-8')
    # ent2tri = ent2tri
    ent2text = Config.get_ent2text()
    rels = Config.get_rel2id()

    undir_rel = open('./data/rel_undir', 'r')
    un_rels = []
    for i in undir_rel:
        p = i.strip().split(':')
        rel = p[0]
        ratio = float(p[1])
        if ratio>0.1:
            un_rels.append(p[0])
            print(rel)

    def get_text(item):
        subj = item['token'][item['subj_start']:item['subj_end'] + 1]
        obj = item['token'][item['obj_start']:item['obj_end'] + 1]
        r = random.random()
        ent_pos = torch.zeros((Config.token_max_length))

        # 标注实体位置
        ent_a = None
        ent_b = None

        # 调换实体位置
        if r < 0.5 and item['relation'] not in un_rels:
            if item['subj_start'] < item['obj_start']:
                text = item['token'][:item['subj_start']] + obj + \
                       item['token'][item['subj_end'] + 1:item['obj_start']] + subj + item['token'][
                                                                                      item['obj_end'] + 1:]
                ent_a = (item['subj_start'], item['subj_start'] + len(obj))
                ent_b = (item['obj_start']+len(obj)-len(subj), item['obj_start']+len(obj))
            else:
                text = item['token'][:item['obj_start']] + subj + \
                       item['token'][item['obj_end'] + 1:item['subj_start']] + obj + item['token'][
                                                                                     item['subj_end'] + 1:]
                ent_a = (item['obj_start'], item['obj_start'] + len(subj))
                ent_b = (item['subj_start'] + len(subj) - len(obj), item['subj_start'] + len(subj))
            label = 1

        else:
            ent_a = (item['subj_start'], item['subj_end'] + 1)
            ent_b = (item['obj_start'], item['obj_end'] + 1)
            text = item['token']
            label = 0

            if item['relation'] in un_rels:
                rr = item['relation']
                # print(f'{subj} {rr} {obj}')
                label = 0

        # 要跳过[CLS]token
        ent_pos[ent_a[0]+1:ent_a[1]+1] = 1
        ent_pos[ent_b[0]+1:ent_b[1]+1] = 1

        # print(text)
        # print(ent_pos)
        # exit()


        return text, label, ent_pos

    # 输入实体集合 返回所有实体相关知识
    def get_knowledge(ents, text_name):
        # replace
        save = []
        for idx, e in enumerate(ents):
            if e not in ent2text.keys():
                save.append(None)
                continue
            save.append(ent2text[e])
            ent2text[e] = text_name[idx]

        triples = []
        for e in ents:
            triples += sql_ent(cur, e)
        triples = [n for n in triples if n[1] in rels.keys()]
        triples = [[ent2text[n[0]], n[1], ent2text[n[2]]] for n in triples
                   if n[0] in ent2text.keys() and n[2] in ent2text.keys()]

        # back
        # mutex.acquire()
        for idx, e in enumerate(ents):
            if e not in ent2text.keys():
                continue
            ent2text[e] = save[idx]
            # print(ent2text[e])
        # mutex.release()
        random.shuffle(triples)
        return triples

    input_ids = []
    token_type_ids = []
    attention_mask = []
    labels = []
    rels_ids = []
    next_sentence_label = []

    dev_json = list(dev_json)


    for i in tqdm(range(len(dev_json))):
        t = dev_json[i]
        ents = [t['subj_label'], t['obj_label']]
        tokens = t['token']
        text_name = [tokenizer.convert_tokens_to_string(tokens[t['subj_start']:t['subj_end']+1]),
                     tokenizer.convert_tokens_to_string(tokens[t['obj_start']:t['obj_end']+1])]
        t, nsp, ent_pos = get_text(t)

        knowledge = get_knowledge(ents, text_name)[:k_num]

        k, rel_id = get_rel_idx(len(t), knowledge)
        t = tokenizer.convert_tokens_to_string(t)
        inputs = tokenizer(t, k, max_length=Config.token_max_length, padding='max_length', truncation=True,
                           return_tensors='pt')
        x, y = mask_tokens(inputs.input_ids, tokenizer, ent_pos)

        # print(tokenizer.convert_ids_to_tokens(x[0, ent_pos == 1]))
        # if i%100 == 0:
        #     print(y[0, ent_pos == 1])
        # exit()
        input_ids.append(x)
        token_type_ids.append(inputs.token_type_ids)
        attention_mask.append(inputs.attention_mask)
        labels.append(y)
        rels_ids.append(rel_id)
        next_sentence_label.append(nsp)

        nsp_data.write(t+'\n')
        nsp_data.write(k + '\n')
        nsp_data.write(str(nsp) + '\n')
        # exit()
    save_file([torch.cat(input_ids),
               torch.cat(token_type_ids),
               torch.cat(attention_mask),
               torch.cat(labels), torch.tensor(rels_ids),
               torch.tensor(next_sentence_label)], f'./data/T-REX/token_id/id_{i_mp}')
    nsp_data.close()



# 调换文本中实体的位置 不使用prompt 添加多个entity的知识
def get_train_data_5(file, glo_idx):
    cur = Config.get_cur(glo_idx%8)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    ent2text = Config.get_ent2text()
    rel2text = load_file('./data/rel2text')

    def swap_entities(text, p1, p2):
        start1 = min(p1[0], p2[0])
        start2 = max(p1[0], p2[0])
        end1 = min(p1[1], p2[1])
        end2 = max(p1[1], p2[1])
        # assume start1 < end1 < start2 < end2
        entity1 = text[start1:end1]
        entity2 = text[start2:end2]
        swapped_text = text[:start1] + entity2 + text[end1:start2] + entity1 + text[end2:]

        return swapped_text



    def swap_and_update_loc(ent_loc, swap_index):
        swap_index = sorted(swap_index)
        ent_a = ent_loc[swap_index[0]]
        len_a = ent_a[1] - ent_a[0]
        ent_b = ent_loc[swap_index[1]]
        len_b = ent_b[1] - ent_b[0]
        diff = len_b - len_a

        temp = copy.copy(ent_b)
        ent_b = [ent_a[0], ent_a[0] + len_b]
        ent_a = [temp[0], temp[0] + len_a]
        ent_loc[swap_index[0]] = ent_b
        ent_loc[swap_index[1]] = ent_a

        for i in range(swap_index[0] + 1, swap_index[1] + 1):
            ent_loc[i][0] += diff
            ent_loc[i][1] += diff
        return ent_loc

    def get_ent_position(text, knowledge, entity_spans):
        # character-level positions of "John" and "New York"
        encoding = tokenizer(text, knowledge, return_offsets_mapping=True, max_length=Config.token_max_length, padding='max_length', truncation=True,
                           return_tensors='pt')  # encode the text
        tokens = encoding.input_ids  # get the tokens
        # print(encoding)
        offsets = encoding.offset_mapping  # get the offsets
        position_ids = [0] * len(tokens[0])  # initialize position_ids with zeros
        # print(position_ids)
        # print(offsets[0][:torch.sum(encoding.token_type_ids==0)])
        for start, end in entity_spans:  # loop over entities
            for i, (token_start, token_end) in enumerate(offsets[0][:torch.sum(encoding.token_type_ids==0)]):  # loop over offsets
                if start <= token_start and end >= token_end:  # check if token is within entity span
                    # print(i)
                    position_ids[i] = 1  # set position_id to 1
        encoding['ent_pos'] = torch.tensor(position_ids)
        # print(tokens)
        # print(position_ids)
        #
        # print(position_ids)
        return encoding
        # Output: [0, 0, 0, 1, 0, 0, 0, 1, 1, 0]

    def find_first_max_part(seq):
        if len(seq) == 0:
            return seq
        # 遍历序列
        for i in range(len(seq) - 1):
            # 比较当前元素和后面的元素
            if seq[i] > seq[i + 1]:
                # 返回当前元素的索引
                return seq[:i+1]
        # 没有找到极大值，返回-1
        return seq


    def get_knowledge(knowledge):
        save = []
        for pid, name in ents:
            if pid not in ent2text.keys():
                save.append(None)
                continue
            save.append(ent2text[pid])
            ent2text[pid] = name

        knowledge = [[ent2text[k[0]], rel2text[k[1]], ent2text[k[2]]] for k in knowledge
                     if k[0] in ent2text.keys() and k[2] in ent2text.keys() and
                     k[1] in rel2text.keys()]
        knowledge = [' '.join(k) for k in knowledge]
        random.shuffle(knowledge)
        knowledge = knowledge[:max_num]
        knowledge = ' . '.join(knowledge) + ' . '
        # print(knowledge)

        for idx, (pid, _) in enumerate(ents):
            if pid not in ent2text.keys():
                continue
            ent2text[pid] = save[idx]
        return knowledge

    input_ids = []
    token_type_ids = []
    attention_mask = []
    labels = []
    rels_ids = []
    next_sentence_label = []
    # data = load_json('./data/T-REX_ent/clean_data/0.json')
    data = load_json(file)
    nsp_data = open(f'./data/T-REX_ent/see/{glo_idx}.txt', 'w', encoding='utf-8')
    undir_rel = []
    for i in open('./data/rel_undir', 'r'):
        p = i.strip().split(':')
        rel = p[0]
        ratio = float(p[1])
        if ratio > 0.1:
            undir_rel.append(p[0])
            # print(rel)

    # 调换实体位置
    for item in tqdm(list(data), desc=file):
        swap_idx = []
        sentence = item['sentence']
        ent_loc = item['ent_loc']
        if len(item['triples']) == 0:
            continue
        triple = random.sample(item['triples'], 1)[0]
        subj_boundaries = triple['subject']['boundaries']
        obj_boundaries = triple['object']['boundaries']
        locs = [loc['boundaries'] for loc in ent_loc]
        locs = find_first_max_part(locs)
        # locs.sort(key=lambda k:k[0])
        ents = list(set([(loc['id'], loc['name']) for loc in ent_loc]))




        # print(sentence[subj_boundaries[0]:subj_boundaries[1]])
        # print(sentence[obj_boundaries[0]:obj_boundaries[1]])
        r = random.random()
        nsp_label = 0
        # print(triple)
        if r<0.5 and triple['relation'] not in undir_rel:
            if subj_boundaries not in locs:
                # print('e')
                continue
            if obj_boundaries not in locs:
                # print('e')
                continue
            # 获取需要交换位置的实体的索引
            swap_idx.append(locs.index(subj_boundaries))
            swap_idx.append(locs.index(obj_boundaries))
            # 交换实体位置
            swapped_text = swap_entities(sentence, subj_boundaries, obj_boundaries)
            # 更新文本中所有实体的位置
            locs = swap_and_update_loc(locs, swap_idx)
            nsp_label = 1
        else:
            swapped_text = sentence

        # 根据文本中的实体添加知识
        knowledge = []

        # print(ents)
        max_triple_per_ent = 4
        max_num = 12
        for pid, _ in ents:
            k = sql_ent(cur, pid)
            k = random.sample(k, 4) if len(k) > max_triple_per_ent else k
            # k = random.sample(k, 4)
            # print(len(k))
            knowledge += k

        # print(knowledge)

        knowledge = get_knowledge(knowledge)


        # print(swapped_text)
        # print(locs)
        # for i in locs:
        #     print(swapped_text[i[0]:i[1]])

        nsp_data.write(sentence+'\n')
        nsp_data.write(swapped_text+'\n')
        nsp_data.write(knowledge+'\n')
        nsp_data.write(json.dumps(triple)+'\n')
        nsp_data.write(str(nsp_label)+'\n')

        # 根据实体的字符级别位置 确定实体的token级别位置
        encoding = get_ent_position(swapped_text, knowledge, locs)
        # print(tokenizer.convert_ids_to_tokens(torch.tensor(encoding['input_ids'][0])[torch.tensor(encoding['ent_pos'])==1]))

        # print(encoding)

        x, y = mask_tokens(encoding.input_ids, tokenizer, encoding['ent_pos'])

        # print(tokenizer.convert_ids_to_tokens(x[0, ent_pos == 1]))
        # if i%100 == 0:
        #     print(y[0, ent_pos == 1])
        # exit()
        input_ids.append(x)
        token_type_ids.append(encoding.token_type_ids)
        attention_mask.append(encoding.attention_mask)
        labels.append(y)
        next_sentence_label.append(nsp_label)

    data_save = [torch.cat(input_ids),
               torch.cat(token_type_ids),
               torch.cat(attention_mask),
               torch.cat(labels),
               torch.tensor(next_sentence_label)]
    print(len(input_ids))
    save_file(data_save, f'./data/T-REX_ent/token_id/id_{glo_idx}')
    nsp_data.close()

    return data_save
    #     exit()



# 开始使用维基百科数据集
# 调换文本中实体的位置 并为模型判断提供三元组依据
def get_train_data_wiki(n, index, file_list, k_num=12):
    tokenizer = Config.get_tokenizer()
    cur = get_cur(index)
    # 50%概率交换文本中实体位置
    def get_text(item):
        if len(item['ent']) < 2:
            return item['token'], 0, torch.zeros((Config.token_max_length))

        sample = random.sample(list(zip(item['ent'], item['ent_loc'])), 2)
        ent, ent_loc = zip(*sample)
        # ent_loc = [g[1] for g in sample]
        item['subj_start'] = ent_loc[0][0]
        item['subj_end'] = ent_loc[0][1] - 1
        item['obj_start'] = ent_loc[1][0]
        item['obj_end'] = ent_loc[1][1] - 1

        subj = item['token'][item['subj_start']:item['subj_end'] + 1]
        obj = item['token'][item['obj_start']:item['obj_end'] + 1]
        r = random.random()
        ent_pos = torch.zeros((Config.token_max_length))

        # 标注实体位置
        ent_a = None
        ent_b = None

        is_undir = False
        for relation in sql_head_tail(cur, ent[0], ent[1]) + sql_head_tail(cur, ent[0], ent[1]):
            if relation in un_rels:
                is_undir = True
                break

        # 调换实体位置
        if r < 0.5 and not is_undir:
            if item['subj_start'] < item['obj_start']:
                text = item['token'][:item['subj_start']] + obj + \
                       item['token'][item['subj_end'] + 1:item['obj_start']] + subj + item['token'][
                                                                                      item['obj_end'] + 1:]
                ent_a = (item['subj_start'], item['subj_start'] + len(obj))
                ent_b = (item['obj_start'] + len(obj) - len(subj), item['obj_start'] + len(obj))
            else:
                text = item['token'][:item['obj_start']] + subj + \
                       item['token'][item['obj_end'] + 1:item['subj_start']] + obj + item['token'][
                                                                                     item['subj_end'] + 1:]
                ent_a = (item['obj_start'], item['obj_start'] + len(subj))
                ent_b = (item['subj_start'] + len(subj) - len(obj), item['subj_start'] + len(subj))
            label = 1

        else:
            ent_a = (item['subj_start'], item['subj_end'] + 1)
            ent_b = (item['obj_start'], item['obj_end'] + 1)
            text = item['token']
            label = 0

            if is_undir:
                # rr = item['relation']
                print(f'{subj} {relation} {obj}')
                label = 0

        # 要跳过[CLS]token
        ent_pos[ent_a[0] + 1:ent_a[1] + 1] = 1
        ent_pos[ent_b[0] + 1:ent_b[1] + 1] = 1

        # print(text)
        # print(ent_pos)
        # exit()

        # return tokenizer.convert_tokens_to_string(text), label, ent_pos
        return text, label, ent_pos

    # 输入实体集合 返回所有实体相关知识
    def get_knowledge(ents, text_name):
        # replace
        save = []
        for idx, e in enumerate(ents):
            if e not in ent2text.keys():
                save.append(None)
                continue
            save.append(ent2text[e])
            ent2text[e] = text_name[idx]

        triples = []
        for e in ents:
            triples += sql_ent(cur, e)
        triples = [n for n in triples if n[1] in rel2id.keys()]
        triples = [[ent2text[n[0]], n[1], ent2text[n[2]]] for n in triples
                   if n[0] in ent2text.keys() and n[2] in ent2text.keys()]

        # back
        for idx, e in enumerate(ents):
            if e not in ent2text.keys():
                continue
            ent2text[e] = save[idx]

        random.shuffle(triples)
        return triples




    ent2text = load_file('./data/ent2text')
    undir_rel = open('./data/rel_undir', 'r')
    un_rels = []
    rel2id = {}
    for idx, i in enumerate(undir_rel):
        p = i.strip().split(':')
        rel = p[0]
        rel2id[rel] = idx
        ratio = float(p[1])
        if ratio > 0.1:
            un_rels.append(p[0])
    # save_json(rel2id, './data/rel2id_wiki.json')
    for idx, file in enumerate(tqdm(file_list)):
        if idx % n == index:
            nsp_data = open(f'./data/pretrain_data/see/{idx}.txt', 'w', encoding='utf-8')
            dev_json = load_json(file)
            input_ids = []
            token_type_ids = []
            attention_mask = []
            labels = []
            rels_ids = []
            next_sentence_label = []

            dev_json = list(dev_json)


            for i in range(len(dev_json)):
                # if i==1000:
                #     break

                t = dev_json[i]
                ents = t['ent']
                text_name = [tokenizer.convert_tokens_to_string(t['token'][loc[0]:loc[1]]) for loc in t['ent_loc']]
                # print(text_name)
                # exit()
                t, nsp, ent_pos = get_text(t)

                # text_name = [for loc in t['ent_loc']]
                # knowledge = get_knowledge(ents, text_name)[:k_num]
                knowledge = get_knowledge(ents, text_name)[:k_num]

                t, k, rel_id = get_rel_idx(t, knowledge)

                inputs = tokenizer(t, k, max_length=Config.token_max_length, padding='max_length', truncation=True,
                                   return_tensors='pt')
                x, y = mask_tokens(inputs.input_ids, tokenizer, ent_pos)

                # print(tokenizer.convert_ids_to_tokens(x[0, ent_pos == 1]))
                # if i%100 == 0:
                #     print(y[0, ent_pos == 1])
                # exit()
                input_ids.append(x)
                token_type_ids.append(inputs.token_type_ids)
                attention_mask.append(inputs.attention_mask)
                labels.append(y)
                rels_ids.append(rel_id)
                next_sentence_label.append(nsp)

                nsp_data.write(t)
                nsp_data.write(k + '\n')
                nsp_data.write(str(nsp) + '\n')
            save_file([torch.cat(input_ids),
                       torch.cat(token_type_ids),
                       torch.cat(attention_mask),
                       torch.cat(labels), torch.tensor(rels_ids),
                       torch.tensor(next_sentence_label)], f'./data/pretrain_data/agg/train_{idx}')
            nsp_data.close()



# 将dev数据集中的token转化为Bert形式的token
def transfer_dev_token():
    tokenizer = Config.get_tokenizer()
    dev_json = load_json('./data/dev.json')
    data = []
    for item in tqdm((dev_json)):
        subj_start = len(tokenizer._tokenize(tokenizer.convert_tokens_to_string(item['token'][:item['subj_start']])))
        subj_end = subj_start+len(tokenizer._tokenize(tokenizer.convert_tokens_to_string(item['token'][item['subj_start']:item['subj_end']+1])))

        obj_start = len(tokenizer._tokenize(tokenizer.convert_tokens_to_string(item['token'][:item['obj_start']])))
        obj_end = obj_start+len(tokenizer._tokenize(tokenizer.convert_tokens_to_string(item['token'][item['obj_start']:item['obj_end']+1])))

        item['token'] = tokenizer._tokenize(tokenizer.convert_tokens_to_string(item['token']))
        item['subj_start'] = subj_start
        item['subj_end'] = subj_end-1
        item['obj_start'] = obj_start
        item['obj_end'] = obj_end-1
        # print(item['token'])
        # str = tokenizer.convert_tokens_to_string(item['token'])
        # print(str)
        # print(item['token'][item['subj_start']:item['subj_end']+1])
        # print(item['token'][item['obj_start']:item['obj_end']+1])
        data.append(item)

    save_json(data, './data/dev_token.json')


# 将train数据集中的token转化为Bert形式的token
def transfer_train_token(train_json, index, n):
    tokenizer = Config.get_tokenizer()
    data = []
    for idx, item in enumerate(tqdm(train_json)):
        # if idx % n == index:
        subj_start = len(tokenizer._tokenize(tokenizer.convert_tokens_to_string(item['token'][:item['subj_start']])))
        subj_end = subj_start+len(tokenizer._tokenize(tokenizer.convert_tokens_to_string(item['token'][item['subj_start']:item['subj_end']+1])))

        obj_start = len(tokenizer._tokenize(tokenizer.convert_tokens_to_string(item['token'][:item['obj_start']])))
        obj_end = obj_start+len(tokenizer._tokenize(tokenizer.convert_tokens_to_string(item['token'][item['obj_start']:item['obj_end']+1])))

        item['token'] = tokenizer._tokenize(tokenizer.convert_tokens_to_string(item['token']))
        item['subj_start'] = subj_start
        item['subj_end'] = subj_end-1
        item['obj_start'] = obj_start
        item['obj_end'] = obj_end-1
        # print(item['token'])
        # str = tokenizer.convert_tokens_to_string(item['token'])
        # print(str)
        # print(item['token'][item['subj_start']:item['subj_end']+1])
        # print(item['token'][item['obj_start']:item['obj_end']+1])
        data.append(item)

    save_json(data, f'./data/T-REX/{index}.json')


if __name__ == '__main__':
    import os
    path = './data/T-REX_ent/clean_data'
    file_list = os.listdir(path)
    pool = Pool(8)

    for idx, file in enumerate(file_list):
        pool.apply_async(get_train_data_5, (os.path.join(path, file), idx))

    pool.close()
    pool.join()

    # n = int(8)
    #
    # # cur =
    # divide = 8
    # for e in range(divide):
    #     li = list(range(64))[e::divide]
    #     print(li)
    #     p = Pool(n)
    #     for i in li:
    #         p.apply_async(get_train_data_4, args=(load_json(f'./data/T-REX/{i}.json'), i, 12))
    #     p.close()
    #     p.join()
