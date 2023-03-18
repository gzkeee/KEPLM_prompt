from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random

import tornado.log
from tqdm import tqdm, trange
import simplejson as json

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from model import BertForSequenceClassification_prompt, BertForPreTraining_prompt
from transformers import BertTokenizer, BertForSequenceClassification, BertForPreTraining
from util import get_ent_info, load_file, save_json
from data_process import get_rel_idx
from config import Config


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, labels, rel_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.labels = labels
        self.rel_id = rel_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, "r") as f:
            return json.load(f)


class TypingProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.json")))
        examples = self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")
        d = {}
        for e in examples:
            for l in e.label:
                if l in d:
                    d[l] += 1
                else:
                    d[l] = 1
        for k, v in d.items():
            d[k] = (len(examples) - v) * 1. /v 
        return examples, list(d.keys()), d

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")


    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = i
            text_a = (line['sent'], [["SPAN", line["start"], line["end"]]])
            text_b = line['ents']
            label = line['labels']
            #if guid != 51:
            #    continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, tokenizer, threshold):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {label: i for i, label in enumerate(label_list)}
    # print(label_map)



    ana = {}

    # 只有目标实体包含知识
    d0 = []

    # 包含句中实体的知识
    d1 = []

    # 完全没有知识
    nk = []

    features = []
    for (ex_index, example) in enumerate(examples):
        ex_text_a = example.text_a[0]
        h = example.text_a[1][0]
        ent_target = ex_text_a[h[1]:h[2]]
        ex_text_a = ex_text_a[:h[1]] + "。 " + ex_text_a[h[1]:h[2]] + " 。" + ex_text_a[h[2]:]
        begin, end = h[1:3]
        h[1] += 2
        h[2] += 2


        # print(ex_text_a)
        # print(example.text_b)
        # print(example.label)
        ent_pos = [x for x in example.text_b if x[-1] > threshold]
        for x in ent_pos:
            if x[1] > end:
                x[1] += 4
                x[2] += 4
            elif x[1] >= begin:
                x[1] += 2
                x[2] += 2
        # print(ent_pos)
        triples = []
        hav_ans = False
        gold_k = []
        for x in ent_pos:
            e_name = ex_text_a[x[1]:x[2]]
            e_dix = x[0]
            t, _, _ = get_ent_info(e_dix, e_name, 3)
            print(e_name)

            if e_name == ent_target:
                hav_ans = True
                d0.append(1)
                d1.append(1)
                nk.append(0)
                gold_k = t
            else:
                triples += t

            # print(e_name)
        triples = gold_k+triples


        if not hav_ans and len(triples) != 0:
            d0.append(0)
            d1.append(1)
            nk.append(0)
        if len(triples) == 0:
            d0.append(0)
            d1.append(0)
            nk.append(1)
        # print(h)
        # exit()
        know, rel_id = get_rel_idx(len(tokenizer._tokenize(ex_text_a)), triples)
        # text_know = text + know

        input = tokenizer(ex_text_a, know, max_length=Config.token_max_length, padding='max_length', truncation=True)
        # input = tokenizer(text, max_length=Config.token_max_length, padding='max_length', truncation=True)

        labels = [0]*len(label_map)
        for l in example.label:
            l = label_map[l]
            labels[l] = 1
        # if ex_index < 10:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("Entity: %s" % example.text_a[1])
        #     logger.info("tokens: %s" % " ".join(
        #             [str(x) for x in zip(tokens, ents)]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("label: %s %s" % (example.label, labels))
        #     logger.info(real_ents)
        #
        features.append(
                InputFeatures(input_ids=input.input_ids,
                          input_mask=input.attention_mask,
                          segment_ids=input.token_type_ids,
                          labels=labels,
                          rel_id=rel_id))

    ana['d0'] = d0
    ana['d1'] = d1
    ana['nk'] = nk
    save_json(ana, './data/OpenEntity/ana.json')
    return features


def _truncate_seq_pair(tokens_a, tokens_b, ents_a, ents_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
            ents_a.pop()
        else:
            tokens_b.pop()
            ents_b.pop()

def accuracy(out, l):
    cnt = 0
    y1 = []
    y2 = []
    for x1, x2 in zip(out, l):
        yy1 = []
        yy2 = []
        top = max(x1)
        for i in range(len(x1)):
            #if x1[i] > 0 or x1[i] == top:
            if x1[i] > 0:
                yy1.append(i)
            if x2[i] > 0:
                yy2.append(i)
        y1.append(yy1)
        y2.append(yy2)
        cnt += set(yy1) == set(yy2)
    return cnt, y1, y2

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--ernie_model", default=None, type=str, required=True,
                        help="Ernie pre-trained model")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    parser.add_argument('--threshold', type=float, default=.3)

    args = parser.parse_args()




    # args.train_batch_size = 16



    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    processor = TypingProcessor()

    train_examples = None
    num_train_steps = None
    train_examples, label_list, d = processor.get_train_examples(args.data_dir)
    label_list = sorted(label_list)
    #class_weight = [min(d[x], 100) for x in label_list]
    #logger.info(class_weight)
    S = []
    for l in label_list:
        s = []
        for ll in label_list:
            if ll in l:
                s.append(1.)
            else:
                s.append(0.)
        S.append(s)

    # Prepare model
    num_labels = 9
    model = BertForSequenceClassification_prompt.from_pretrained(
    "bert-base-uncased", num_labels=num_labels, problem_type="multi_label_classification")
    model_ = BertForPreTraining_prompt.from_pretrained("bert-base-uncased")
    param = load_file('./param/12_state_fina')['param']

    param_set = {}
    for k, v in param.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        param_set[new_k] = v

    model_.load_state_dict(param_set)
    model.bert.load_state_dict(model_.bert.state_dict())

    # model = BertForSequenceClassification.from_pretrained(
    #     "bert-base-uncased", num_labels=num_labels, problem_type="multi_label_classification")
    # model_ = BertForPreTraining.from_pretrained("bert-base-uncased")
    # param = load_file('./param/0_mode_full_fina')
    # model_.load_state_dict(param)
    # model.bert.load_state_dict(model_.bert.state_dict())

    device = 'cuda'

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    global_step = 0

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, tokenizer, args.threshold)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_labels = torch.tensor([f.labels for f in train_features], dtype=torch.float)
        all_rel_ids = torch.stack([torch.tensor(f.rel_id).long() for f in train_features], dim=0)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_labels, all_rel_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)
        print(args.train_batch_size)

        output_loss_file = os.path.join(args.output_dir, "loss")
        loss_fout = open(output_loss_file, 'w')
        model.train()
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, labels, rel_ids = batch

                loss = model(input_ids, input_mask, segment_ids, labels=labels, rel=rel_ids).loss
                # loss = model(input_ids, input_mask, segment_ids, labels=labels).loss
                with torch.autograd.set_detect_anomaly(True):
                    loss.backward()

                if (step + 1) % 32 == 0:
                    print(loss.item())
                    loss_fout.write("{}\n".format(loss.item()))
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                # if global_step % 150 == 0 and global_step > 0:
                #     model_to_save = model.module if hasattr(model, 'module') else model
                #     output_model_file = os.path.join(args.output_dir, "pytorch_model.bin_{}".format(global_step))
                #     torch.save(model_to_save.state_dict(), output_model_file)
            if epoch >= 2:
                model_to_save = model.module if hasattr(model, 'module') else model
                output_model_file = os.path.join(args.output_dir, "pytorch_model.bin_{}".format(epoch))
                torch.save(model_to_save.state_dict(), output_model_file)

if __name__ == "__main__":
    main()
