import torch.utils.data as Data
import torch
from util import save_file, load_file
from model import Model, BertForMaskedLM_prompt, BertForPreTraining_prompt
from transformers import BertTokenizer
import numpy as np
from tqdm import tqdm
from config import Config
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import random
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def agg_data(file_list, num=6):
    data = [[] for i in range(num)]
    # print(len(data))
    for data_p in file_list:
        # print(data_p)
        item = load_file(data_p)
        # print(len(item))
        for j in range(len(item)):
            data[j].append(item[j])
    # print(len(data))
    for j in range(len(item)):
        data[j] = torch.cat(data[j])
        print(data[j].size())
    return data


def main(rank, world_size, all_file):
    # rank += 1
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # device = 'cuda'

    accumulation_steps = Config.accumulation_steps//world_size*2
    print(accumulation_steps)
    model = BertForPreTraining_prompt.from_pretrained("bert-base-uncased")
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # loss_sum = 0
    log = open('./log', 'w')



    # for idx, (name, para) in enumerate(model.named_parameters()):
    #     if idx >= 5:
    #         pass
    #     else:
    #         para.requires_grad = False

    for idx, (name, para) in enumerate(model.named_parameters()):
        print(idx, name, para.requires_grad, para.device)


    print(all_file)
    file_divide = 8
    loss_print = 0
    mlm_loss = 0
    nsp_loss = 0
    correct = 0
    for epc in range(file_divide):
        print(all_file[epc::file_divide])
        dataset = Data.TensorDataset(*agg_data(all_file[epc::file_divide]))
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        load = Data.DataLoader(dataset, batch_size=8, sampler=train_sampler)
        train_sampler.set_epoch(epc)
        for idx, item in enumerate(tqdm(load)):
            item = [i.to(rank) for i in item]
            input_ids, token_type_ids, attention_mask, label, rels_ids, nsp_label = item




            res = model(input_ids, attention_mask, token_type_ids, labels=label, rel=rels_ids.long(), next_sentence_label=nsp_label)
            loss = res['loss'] / accumulation_steps
            loss_print += loss
            pred = res['seq_relationship_logits']
            correct += (pred.argmax(1) == nsp_label).type(torch.float).sum().item()
            loss.backward()

            # for idx, (name, para) in enumerate(model.named_parameters()):
            #     # print(para.zer)
            #     if idx<5:
            #         para.grad.data.zero_()
                    # print('============')
                    # print()
                    # print('============')
                    # para.grad = 0
                # print(idx, name, para.requires_grad, para.device)



            if (idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            # loss_sum += loss

            mlm_loss += res['masked_lm_loss']
            nsp_loss += res['next_sentence_loss']
            if idx % 256 == 0:
                print(f'masked_lm_loss:{mlm_loss/4}\nnext_sentence_loss:{nsp_loss/4}')  # print(loss_print / 4)
                print(f'next_sentence_acc:{correct/(256*16)}')  # print(loss_print / 4)
                mlm_loss = 0
                nsp_loss = 0
                correct = 0
                # loss_print = 0
            if idx % 5000 == 0 and rank == 0:
                # print(rels_ids[0])
                print(tokenizer.decode(input_ids[0][rels_ids[0] != 825 * Config.rel_num]))
                print(tokenizer.decode(input_ids[0]))
                save_file(model.module.state_dict(), f'./param/{epc}_mode_{idx}')
        if rank == 0:
            save_file(model.module.state_dict(), f'./param/mode_full_fina')


    # print(dataset[0])


if __name__ == '__main__':
    file_dir = './data/T-REX/token_id'
    # file_dir = './data/T-REX/token_id'
    all_file = [os.path.join(file_dir, i) for i in os.listdir(file_dir)]
    random.shuffle(all_file)
    world_size = 4
    mp.spawn(main,
             args=(world_size, all_file),
             nprocs=world_size,
             join=True)
