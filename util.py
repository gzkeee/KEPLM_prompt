import pickle
import json
from config import Config
from multiprocessing import Pool

def save_file(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_file(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_json(path):
    f = open(path, 'r', encoding='utf-8')
    return json.load(f)

def save_json(obj, path):
    b = json.dumps(obj)
    f2 = open(path, 'w')
    f2.write(b)
    f2.close()


# 查询实体相关三元组
def sql_ent(cur, ent):
    sql_query_entity = '''
        select * from triples where Head == '{}'
        '''
    # print(sql_query_entity)
    cur.execute(sql_query_entity.format(ent))
    res = cur.fetchall()
    return res

# 根据头尾实体调查关系
def sql_head_tail(cur, head, tail):
    sql_query_entity = '''
        select * from triples where Head == '{}' and Tail == '{}'
        '''
    cur.execute(sql_query_entity.format(head, tail))
    # print(sql_query_entity.format(head, tail))
    res = cur.fetchall()
    rel = [r[1] for r in res]
    return rel

# 查询关系相关三元组
def sql_rel(cur, rel):
    sql_query_entity = '''
        select * from triples where Rel == '{}'
        '''
    cur.execute(sql_query_entity.format(rel))
    res = cur.fetchall()

    return res


def rel2id(rel):
    rel_to_id = Config.get_rel2id()
    return rel_to_id[rel]

def get_ent_info(ent, text_name, triple_num, extra_info=None):
    ent2text = Config.get_ent2text()
    rel_to_id = Config.get_rel2id()
    cur = Config.get_cur()
    have_ans = False
    ht_pair = 0

    infos = sql_ent(cur, ent)

    triples = []
    for i in infos:
        head, rel, tail = i
        if rel in rel_to_id.keys() and tail in ent2text.keys():
            # 头尾实体都出现在文本中  但是关系不一定和文本中的一致
            if extra_info != None and extra_info[1] == tail:
                triples.append([text_name, rel, extra_info[0]])
                # triples.append([text_name, rel, ent2text[tail]])
                if extra_info[2] == rel:
                    # 包含正确的三元组
                    have_ans = True
                #     文本中的头尾实体存在于三元组中
                ht_pair += 1

            else:
                triples.append([text_name, rel, ent2text[tail]])



    triples = triples[:triple_num] if len(triples) > triple_num else triples
    return triples, ht_pair, have_ans
    # print(ent)
    # return [], ht_pair, False



def mp_data(f, args, n):
    p = Pool(n)
    for i in range(n):
        p.apply_async(f, args=args)
    p.close()
    p.join()

if __name__ == '__main__':
    # print(get_ent_info('Q676576'))
    # print(sql_head_tail(cur, 'Q5142631', 'Q3141'))
    pass