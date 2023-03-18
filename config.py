# from util import load_json
import json
from transformers import BertTokenizer
import sqlite3
import pickle

def load_json(path):
    f = open(path, 'r', encoding='utf-8')
    return json.load(f)

def load_file(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

class Config:
    use_rel_prompt = True
    rel_token_num = 3
    rel_size = 825
    token_max_length = 256

    bsz = 256
    accumulation_steps = bsz / 16

    rel2id = None
    ent2text = None
    tokenizer = None
    cur = None

    @staticmethod
    def get_rel2id():
        if Config.rel2id is None:
            Config.rel2id = load_json('./data/relations.json')
        # print(Config.rel2id)
        return Config.rel2id

    @staticmethod
    def get_ent2text():
        if Config.ent2text is None:
            Config.ent2text = load_file('./data/ent2text')
        return Config.ent2text

    @staticmethod
    def get_tokenizer():
        if Config.tokenizer is None:
            Config.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        return Config.tokenizer

    @staticmethod
    def get_cur(i=0):
        if Config.cur is None:
            db_path = f'./wikidata_5m_{i}.db'
            conn = sqlite3.connect(db_path)
            Config.cur = conn.cursor()
        return Config.cur
