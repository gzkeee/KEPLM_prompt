import os.path
import random

import torch.utils.data as Data
import torch
from util import save_file, load_file, save_json
from model import BertForMaskedLM_prompt, BertForPreTraining_prompt
from transformers import BertTokenizer, BertForPreTraining
import numpy as np
from tqdm import tqdm
from config import Config


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def train_mask_lm():
    device = 'cuda'
    dataset = Data.TensorDataset(*load_file('./data/dev_data/dev'))
    # bsz = Config.bsz
    accumulation_steps = Config.accumulation_steps
    print(accumulation_steps)
    load = Data.DataLoader(dataset, batch_size=16, shuffle=True)
    model = BertForMaskedLM_prompt.from_pretrained("bert-base-uncased")
    # param = load_file('./param/mode_full_14000')
    # model.load_state_dict(param)
    for idx, (name, para) in enumerate(model.named_parameters()):
        if idx >= 197:
            pass
        else:
            para.requires_grad = False

    for idx, (name, para) in enumerate(model.named_parameters()):
        print(idx, name, para.requires_grad)

    # exit()
    #
    # for name, para in model.named_parameters():
    #     para.requires_grad = False
    #
    # model.bert.encoder.param.requires_grad = True

    # model.bert.encoder.param = torch.nn.Parameter(load_file('./param/mode_para_only_triple_fina'))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # loss_sum = 0

    loss_print = 0
    for epc in range(5):
        for idx, item in enumerate(tqdm(load)):
            # for name, para in model.named_parameters():
            #     print(name, para.device)
            item[-2] = torch.where(item[0] == tokenizer.mask_token_id, item[-2], -100)
            item[-1][item[-1] == -1] = 410 * Config.rel_num
            item = [i.to(device) for i in item]
            input_ids, token_type_ids, attention_mask, label, rels_ids = item

            res = model(input_ids, attention_mask, token_type_ids, labels=label, rel=rels_ids.long())
            loss = res.loss / accumulation_steps
            loss_print += loss
            logits = res.logits

            loss.backward()
            if (idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            # loss_sum += loss
            if idx % 32 == 0:
                print(loss_print / 4)
                loss_print = 0
            if idx % 500 == 0:
                # mask_token_index = (input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
                # predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
                # true_token_id = label[0, mask_token_index]
                # print(tokenizer.decode(true_token_id))
                # print(tokenizer.decode(predicted_token_id))
                # print(tokenizer.decode(input_ids[0][rels_ids[0] != 410 * Config.rel_num]))
                print(tokenizer.decode(input_ids[0]))
                save_file(model.state_dict(), f'./param/{epc}_mode_full_{idx}')
        save_file(model.state_dict(), f'./param/{epc}_mode_full_fina')
        # break
    # print(dataset[0])


# 判断给定三元组中是否有三元组与文本的含义相符
def train_nsp_lm():
    # 训练数据太大 需要分批导入
    def agg_data(file_list, num=6):
        data = [[] for i in range(num)]
        for data_p in file_list:
            print(data_p)
            item = load_file(data_p)
            for j in range(len(item)):
                data[j].append(item[j])
        for j in range(len(item)):
            data[j] = torch.cat(data[j])
            print(data[j].size())
        return data

    accumulation_steps = Config.accumulation_steps
    print(accumulation_steps)




    loss_print = 0
    mlm_loss = 0
    nsp_loss = 0
    correct = 0
    # file_dir = './data/pretrain_data/agg'
    # file_dir = './data/T-REX/token_id'
    file_dir = './data/T-REX_ent/token_id'
    all_file = [os.path.join(file_dir, i) for i in os.listdir(file_dir)]
    random.shuffle(all_file)

    device = 'cuda'
    model = BertForPreTraining_prompt.from_pretrained("bert-base-uncased")
    # model = BertForPreTraining.from_pretrained("bert-base-uncased")
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

    for idx, (name, para) in enumerate(model.named_parameters()):
        if idx >= 5:
            pass
        else:
            para.requires_grad = False

    for idx, (name, para) in enumerate(model.named_parameters()):
        print(idx, name, para.requires_grad)

    load = False
    epc_start = -1
    if load:
        epc_start = 12
        check_point = load_file(f'./param/{epc_start}_state_fina')
        # model.load_state_dict(check_point['param'])
        optimizer.load_state_dict(check_point['opt'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        param = check_point['param']
        param_set = {}
        for k, v in param.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            param_set[new_k] = v
        model.load_state_dict(param_set)
        all_file = check_point['file_list']

    model = model.to(device)
    # model = torch.nn.DataParallel(model)

    print(all_file)
    file_divide = 3
    for epc in range(epc_start+1, file_divide):
        print(all_file[epc::file_divide])
        dataset = Data.TensorDataset(*agg_data(all_file[epc::file_divide], 5))
        # dataset = Data.TensorDataset(*load_file('./data/Entity_Shuffling/dev'))
        bsz = 16
        load = Data.DataLoader(dataset, batch_size=bsz, shuffle=True)

        for idx, item in enumerate(tqdm(load)):
            # break
            item = [i.to(device) for i in item]
            # input_ids, token_type_ids, attention_mask, label, rels_ids, nsp_label = item
            input_ids, token_type_ids, attention_mask, label, nsp_label = item

            # with torch.no_grad():
            # res = model(input_ids, attention_mask, token_type_ids, labels=label, rel=rels_ids.long(),
            #             next_sentence_label=nsp_label)
            res = model(input_ids, attention_mask, token_type_ids, labels=label,
                        next_sentence_label=nsp_label)
            loss = res['loss'] / accumulation_steps
            loss = loss.mean()
            loss_print += loss
            pred = res['seq_relationship_logits']
            correct += (pred.argmax(1) == nsp_label).type(torch.float).sum().item()


            loss.backward()
            if (idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()


            mlm_loss += res['masked_lm_loss']
            nsp_loss += res['next_sentence_loss']
            if idx % 256 == 0:
                print(f'masked_lm_loss:{mlm_loss/4}\nnext_sentence_loss:{nsp_loss/4}')  # print(loss_print / 4)
                print(f'next_sentence_acc:{correct/(256*bsz)}')  # print(loss_print / 4)
                mlm_loss = 0
                nsp_loss = 0
                correct = 0
            if idx % 5000 == 0:
                # print(tokenizer.decode(input_ids[0][rels_ids[0] != 0]))
                print(tokenizer.decode(input_ids[0]))
                save_file(model.state_dict(), f'./param/{epc}_mode_{idx}')
        model_to_save = model.module if hasattr(model, 'module') else model
        save_file({'param': model_to_save.state_dict(), 'file_list': all_file, 'opt': optimizer.state_dict()}, f'./param/{epc}_state_fina')



def test_model():
    loss_print = 0
    mlm_loss = 0
    nsp_loss = 0
    correct = 0
    # file_dir = './data/pretrain_data/agg'
    # file_dir = './data/T-REX/token_id'
    file_dir = './data/T-REX/token_id'
    all_file = [os.path.join(file_dir, i) for i in os.listdir(file_dir)]
    random.shuffle(all_file)

    device = 'cuda'
    model = BertForPreTraining_prompt.from_pretrained("bert-base-uncased")
    # model = BertForPreTraining.from_pretrained("bert-base-uncased")

    load = True
    epc_start = -1
    if load:
        epc_start = 2
        # check_point = load_file(f'./param/{epc_start}_state_fina')
        # param = check_point['param']

        # model.load_state_dict(param)
        param = load_file(f'./param/{epc_start}_state_fina')['param']
        param_set = {}
        for k, v in param.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            param_set[new_k] = v
        model.load_state_dict(param_set)
        # model.load_state_dict(param)
        # model = model.to('cpu')
        # save_file(model.state_dict(), './param/wiki_2_mode_15000')
        # exit()

    model = model.to(device)


    dataset = Data.TensorDataset(*load_file('./data/Entity_Shuffling/dev'))
    dataset = Data.TensorDataset(*load_file('./data/T-REX_ent/token_id/id_0'))
    # dataset = Data.TensorDataset(*load_file('./data/T-REX_ent/token_id/id_0'))
    bsz = 16
    load = Data.DataLoader(dataset, batch_size=bsz, shuffle=True)

    for idx, item in enumerate(tqdm(load)):
        # break
        item = [i.to(device) for i in item]
        # input_ids, token_type_ids, attention_mask, label, rels_ids, nsp_label = item
        input_ids, token_type_ids, attention_mask, label, nsp_label = item

        # 去除知识
        # input_ids[token_type_ids == 1] = 0
        # rels_ids = torch.ones_like(rels_ids)*-1
        #
        # # 无prompt
        # rels_ids = torch.ones_like(rels_ids)*-1
        #
        # rel->0
        # input_ids[rels_ids == -1] = 0




        with torch.no_grad():
            # res = model(input_ids, attention_mask, token_type_ids, labels=label, rel=rels_ids.long(),
            #             next_sentence_label=nsp_label)
            res = model(input_ids, attention_mask, token_type_ids, labels=label,
                        next_sentence_label=nsp_label)


        pred = res['seq_relationship_logits']
        correct += (pred.argmax(1) == nsp_label).type(torch.float).sum().item()

        if idx % 256 == 0:
            print(f'next_sentence_acc:{correct/(256*bsz)}')  # print(loss_print / 4)
            correct = 0



# 测试无额外知识时，模型是否具备能力分辨被调换顺序的实体
def train_shuffle_ent_raw():
    ans_corr = []
    device = 'cuda'
    # print(len(load_file('./data/data_nsp/dev')))
    dataset = Data.TensorDataset(*load_file('./data/Entity_Shuffling/agg/train_new_0'))

    accumulation_steps = Config.accumulation_steps
    print(accumulation_steps)
    load = Data.DataLoader(dataset, batch_size=16, shuffle=False)
    model = BertForPreTraining.from_pretrained("bert-base-uncased")
    model.load_state_dict(load_file('./param/0_mode_full_fina'))


    for idx, (name, para) in enumerate(model.named_parameters()):
        if idx >= 5:
            pass
        else:
            para.requires_grad = False

    for idx, (name, para) in enumerate(model.named_parameters()):
        print(idx, name, para.requires_grad)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # loss_sum = 0

    loss_print = 0
    mlm_loss = 0
    nsp_loss = 0
    correct = 0
    for epc in range(5):
        for idx, item in enumerate(tqdm(load)):
            item = [i.to(device) for i in item]
            input_ids, token_type_ids, attention_mask, label, nsp_label = item
            # input_ids[token_type_ids]


            with torch.no_grad():
                res = model(input_ids, attention_mask, token_type_ids, labels=label, next_sentence_label=nsp_label)
            loss = res.loss / accumulation_steps
            loss_print += loss
            pred = res.seq_relationship_logits
            correct += (pred.argmax(1) == nsp_label).type(torch.float).sum().item()
            # print(pred.argmax(1) == nsp_label)
            ans_corr += (pred.argmax(1) == nsp_label).cpu().numpy().tolist()
            # exit()
            # logits = res.logits

            if (idx+1) % 1000 == 0:
                save_json(ans_corr, './data/Entity_Shuffling/result.json')



            # loss.backward()
            if (idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            # loss_sum += loss

            # mlm_loss += res.masked_lm_loss
            # nsp_loss += res.next_sentence_loss
            if idx % 256 == 0:
                # print(f'masked_lm_loss:{mlm_loss/4}\nnext_sentence_loss:{nsp_loss/4}')  # print(loss_print / 4)
                print(f'next_sentence_acc:{correct/(256*16)}')  # print(loss_print / 4)
                # mlm_loss = 0
                # nsp_loss = 0
                correct = 0
                # loss_print = 0
            if idx % 1000 == 0:
                print(tokenizer.decode(input_ids[0]))
                # save_file(model.state_dict(), f'./param/{epc}_mode_full_{idx}')
        # save_file(model.state_dict(), f'./param/{epc}_mode_full_fina')


if __name__ == '__main__':
    # train_shuffle_ent_raw()
    # train_nsp_lm()
    test_model()