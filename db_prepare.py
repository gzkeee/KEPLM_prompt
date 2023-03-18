import os
import sqlite3
from tqdm import tqdm

"""
将知识图谱存储到数据库中
"""

def text2db(copy):
        conn = sqlite3.connect(f'wikidata_5m_{copy}.db')
        file_list = ['./data/wikidata5m_inductive_train.txt',
                     './data/wikidata5m_inductive_valid.txt',
                     './data/wikidata5m_inductive_test.txt']
        cur = conn.cursor()

        sql_text_1 = '''CREATE TABLE triples(Head TEXT,
                    Rel TEXT,
                    Tail TEXT);'''

        cur.execute(sql_text_1)

        for file in tqdm(file_list):
                data = []
                with open(file, 'r') as f:
                        for line in tqdm(list(f)):
                                data.append(line.split())


                cur.executemany('INSERT INTO triples VALUES (?,?,?)', data)


        # 创建头实体和关系的索引
        sql_text_2 = '''CREATE INDEX head
                on triples (Head);
                '''

        sql_text_3 = '''CREATE INDEX relation
                        on triples (Rel);
                        '''

        cur.execute(sql_text_2)
        cur.execute(sql_text_3)

        conn.commit()


def get_cur():
    db_path = './wikidata_5m.db'
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    return cur

def sql_ent(cur, ent):
    sql_query_entity = '''
        select * from triples where Head == '{}'
        '''
    cur.execute(sql_query_entity.format(ent))
    res = cur.fetchall()
    return res



if __name__ == '__main__':
        for i in range(8):
            text2db(i)
