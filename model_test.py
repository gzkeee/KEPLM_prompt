from transformers import AutoTokenizer, BertForMaskedLM, BertTokenizer
import torch
import random
import string
import pickle
from tqdm import tqdm
import torch.utils.data as Data
from model import BertForMaskedLM_prompt
from config import Config
from test import get_rel_idx

def save_file(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_file(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# param = load_file('./data/generate/test_model_fina')
# model = BertForMaskedLM_prompt.from_pretrained("bert-base-uncased")
# # model.load_state_dict(param)
#
# inputs = tokenizer("The capital of B is [MASK] [MASK] [MASK] . [SEP] B # how are you . A # not not ok", return_tensors="pt")
#
# with torch.no_grad():
#     logits = model(**inputs, rel=torch.tensor([[0]])).logits
#
# # retrieve index of [MASK]
# mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
#
# predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
# print(tokenizer.decode(predicted_token_id))
#
# labels = tokenizer("The capital of B is how are you . [SEP] B # how are you . A # not not ok", return_tensors="pt")["input_ids"]
# # mask labels of non-[MASK] tokens
# labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
# # inputs.input_ids[inputs.input_ids == 101] = -1
#
# outputs = model(**inputs, labels=labels, rel=torch.tensor([[0]]))
# print(round(outputs.loss.item(), 2))
#
#
# random_string = ''.join(random.choice(string.ascii_lowercase) for _ in range(5))
# print(random_string)

f = open('./data/generate/random_data.txt', 'w', encoding='utf-8')
input_ids = []
token_type_ids = []
attention_mask = []
rels_ids = []
label = []

for i in tqdm(range(100000)):
  ent1_num = random.randint(1, 2)
  ent2_num = random.randint(1, 2)
  ent3_num = random.randint(1, 2)
  ent4_num = random.randint(1, 2)

  vocab = list(tokenizer.vocab.keys())
  vocab = [i for i in vocab if '#' not in i]
  # print(vocab)
  random_country = ''.join(random.choice(vocab) for _ in range(ent1_num))
  random_city = ''.join(random.choice(vocab) for _ in range(ent2_num))
  second_country = ''.join(random.choice(vocab) for _ in range(ent3_num))
  second_city = ''.join(random.choice(vocab) for _ in range(ent4_num))


  stragety = random.random()
  # relation = '# #'
  if stragety < 0.5:
      num = len(tokenizer._tokenize(random_city))
      mask = ' [MASK] '*num
      # x = f'The capital of {random_country} is {mask} . [SEP] {random_country} {relation} {random_city} . {second_country} {relation} {second_city}'
      # y = f'The capital of {random_country} is {random_city}. [SEP] {random_country} {relation} {random_city} . {second_country} {relation} {second_city}'

      text = f'The capital of {random_country} is {random_city} .'
      triples = [[random_country, 'P40', random_city], [second_country, 'P40', second_city]]
      y, rel_id = get_rel_idx(text, triples)
      x = y.replace(random_city, mask, 1)

  else:
      num = len(tokenizer._tokenize(random_city))
      mask = ' [MASK] ' * num
      # x = f'The capital of {random_country} is {mask} . [SEP] {second_country} {relation} {second_city} . {random_country} {relation} {random_city}  '
      # y = f'The capital of {random_country} is {random_city}. [SEP]  {second_country} {relation} {second_city} . {random_country} {relation} {random_city} '

      text = f'The capital of {random_country} is {random_city} .'
      triples = [[second_country, 'P40', second_city], [random_country, 'P40', random_city], ]
      y, rel_id = get_rel_idx(text, triples)

      x = y.replace(random_city, mask, 1)



  inputs = tokenizer(x, max_length=Config.token_max_length, padding='max_length', truncation=True)
  # print(tokenizer.decode(inputs.input_ids))
  # exit()
  input_ids.append(inputs.input_ids)
  token_type_ids.append(inputs.token_type_ids)
  attention_mask.append(inputs.attention_mask)
  pair2 = tokenizer(y, max_length=Config.token_max_length, padding='max_length', truncation=True)
  label.append(pair2.input_ids)

  if len(rel_id) != Config.token_max_length:
      print(rel_id)

  rels_ids.append(rel_id)
  # print(pair2.input_ids)

  f.write(x+'[SEN]'+y+'\n')
f.close()
save_file([torch.tensor(input_ids),
               torch.tensor(token_type_ids),
               torch.tensor(attention_mask),
               torch.tensor(label), torch.tensor(rels_ids)], './data/generate/random_train')


device = 'cuda'


# exit()
#
if __name__ == '__main__':
    dataset = Data.TensorDataset(*load_file('./data/generate/random_train'))
    bsz = 128
    load = Data.DataLoader(dataset, batch_size=bsz, shuffle=True)
    model = BertForMaskedLM_prompt.from_pretrained("bert-base-uncased")
    model.load_state_dict(load_file('./data/generate/test_model_fina'))
    frozen = []
    for idx, (name, para) in enumerate(model.named_parameters()):
        if idx > 180:
            pass
        else:
            para.requires_grad = False

    for idx, (name, para) in enumerate(model.named_parameters()):
        print(idx, name, para.requires_grad)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # loss_sum = 0

    for idx, item in enumerate(tqdm(load)):
        item[-2] = torch.where(item[0] == tokenizer.mask_token_id, item[-2], -100)
        item = [i.to(device) for i in item]
        input_ids, token_type_ids, attention_mask, label, rels_ids = item
        # print(rels_ids)
        rels_ids[rels_ids == -1] = 410 * Config.rel_num

        # print(input_ids[0])
        # print(input_ids[1])
        # print(tokenizer.mask_token_id)
        # exit()

        res = model(input_ids, attention_mask, token_type_ids, labels=label, rel=rels_ids.long())
        loss = res.loss
        logits = res.logits
        # loss.requires_grad=True
        # print(loss)
        # print(loss.requires_grad)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # loss_sum += loss
        if idx % 32 == 0:
            # retrieve index of [MASK]
            mask_token_index = (input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
            predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
            true_token_id = label[0, mask_token_index]
            print(true_token_id)
            print(predicted_token_id)
            print(loss)
        if idx % 100 == 0:
            save_file(model.state_dict(), f'./data/generate/test_model_{idx}')
    save_file(model.state_dict(), f'./data/generate/test_model_fina')


