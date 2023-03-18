from transformers import BertForMaskedLM, BertTokenizer
from model import BertForMaskedLM_prompt, BertForPreTraining_prompt
from data_process import get_rel_idx
from util import load_file
from config import Config

import torch
#
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertForMaskedLM_prompt.from_pretrained("bert-base-uncased")
# param = load_file('./param/m1/mode_full_38500')
# model.load_state_dict(param)
# # text = 'The capital of France is [MASK] .'
# text = "The regiment distinguished itself in the Asia Minor Campaign and the [MASK] [MASK] [MASK] [MASK]  , where it participated in the battles of Klisura and Pogradec ."
# # print(text[131:138])
# # print(text[69:88])
# print(tokenizer._tokenize(text))
# # exit()
# # 作弊1：根据答案长度去三元组中找答案
# # 可能2：只识别关系  没有识别到主语
# triple = [['Klisura', 'P361', 'ok - France fuck War'], ['Asia Minor Campaign', 'P36', 'fuck ooo war']]
#
# text_know, rel_id = get_rel_idx(text, triple)
# rel_id = torch.tensor(rel_id)
# print(tokenizer._tokenize(text_know))
# print(text_know)
# # print(rel_id)
#
# # exit()
#
# inputs = tokenizer(text_know, return_tensors="pt", max_length=Config.token_max_length, padding='max_length', truncation=True)
#
# with torch.no_grad():
#     logits = model(**inputs, rel=rel_id).logits
#
# # retrieve index of [MASK]
# mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
#
# print(12)
# print(mask_token_index)
#
# predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
# print(tokenizer.decode(predicted_token_id))
# exit()
#
# labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
# # mask labels of non-[MASK] tokens
# labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
#
# outputs = model(**inputs, labels=labels)
# round(outputs.loss.item(), 2)