
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, BertForSequenceClassification, BertTokenizer
from model import BertforFewRel, BertForSequenceClassification_prompt, BertForMaskedLM_prompt, BertForPreTraining_prompt
from util import get_ent_info, load_file, save_json
from data_process import get_rel_idx
from config import Config
import json


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

ka_path = './data/fewrel/ka_dev.txt'
k_correct = './data/fewrel/k_correct.txt'


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id, rel_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
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
        with open(input_file, "r", encoding='utf-8') as f:
            return json.loads(f.read())


class FewrelProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        examples = self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")
        labels = set([x.label for x in examples])
        return examples, list(labels)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_labels(self):
        """Useless"""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            for x in line['ents']:
                if x[1] == 1:
                    x[1] = 0
            text_a = (line['text'], line['ents'])
            label = line['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, threshold):
    """Loads a data file into a list of `InputBatch`s."""
    label_list = sorted(label_list)

    label_map = {label: i for i, label in enumerate(label_list)}
    # idx2label = {i: label for i, label in enumerate(label_list)}
    # save_json(idx2label, 'data/fewrel/idx2label')
    # exit()


    ka = open(ka_path, 'w')
    kc = open(k_correct, 'w')
    # print(len(label_map))

    features = []
    for (ex_index, example) in enumerate(examples):
        ex_text_a = example.text_a[0]
        h, t = example.text_a[1]
        h_name = ex_text_a[h[1]:h[2]]
        h_idx = h[0]
        t_name = ex_text_a[t[1]:t[2]]
        t_idx = t[0]
        # Add [HD] and [TL], which are "#" and "$" respectively.
        if h[1] < t[1]:
            ex_text_a = ex_text_a[:h[1]] + "# " + h_name + " #" + ex_text_a[
                                                                  h[2]:t[1]] + "$ " + t_name + " $" + ex_text_a[t[2]:]
        else:
            ex_text_a = ex_text_a[:t[1]] + "$ " + t_name + " $" + ex_text_a[
                                                                  t[2]:h[1]] + "# " + h_name + " #" + ex_text_a[h[2]:]


        head_triples, h_pair, h_have_ans = get_ent_info(h[0], h_name, 6, (t_name, t_idx, example.label))
        tail_triples, t_pair, t_have_ans = get_ent_info(t[0], t_name, 6, (h_name, h_idx, example.label))


        # 记录额外知识的情况
        #   0.无三元组添加
        #   1.头实体或者尾实体的有知识添加
        #   2.头实体和尾实体都有知识添加
        ka_num = int(len(head_triples)!=0) + int(len(tail_triples)!=0)

        #   10 没有三元组同时包含头尾实体
        #   10+n 有n个三元组同时包含头尾实体
        if h_pair != 0 or t_pair != 0:
            ka_num = 10+h_pair+t_pair
        ka.write(f'{ka_num}\n')

        # 记录知识中是否包含正确答案
        if h_have_ans or t_have_ans:
            kc.write(f'{1}\n')
        else:
            kc.write(f'{0}\n')

        text, know, rel_id = get_rel_idx(ex_text_a, head_triples+tail_triples)
        # text_know = text+know

        input = tokenizer(text, know, max_length=Config.token_max_length, padding='max_length', truncation=True)
        # input = tokenizer(text, max_length=Config.token_max_length, padding='max_length', truncation=True)
        # print(text)
        # print(ex_text_a)
        # exit()
        label_id = label_map[example.label]
        features.append(
            InputFeatures(input_ids=input.input_ids,
                          input_mask=input.attention_mask,
                          segment_ids=input.token_type_ids,
                          label_id=label_id,
                          rel_id=rel_id))
    # exit()
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


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
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
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--threshold', type=float, default=.3)

    args = parser.parse_args()

    processors = FewrelProcessor

    num_labels_task = 80

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    processor = processors()
    num_labels = num_labels_task
    label_list = None

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    train_examples = None
    num_train_steps = None
    train_examples, label_list = processor.get_train_examples(args.data_dir)
    num_train_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    device = 'cuda:0'
    model = BertForSequenceClassification_prompt.from_pretrained("bert-base-uncased", problem_type="single_label_classification", num_labels=80)
    model_ = BertForPreTraining_prompt.from_pretrained("bert-base-uncased")
    param = load_file('./param/m9/0_mode_fina')
    model_.load_state_dict(param)
    model.bert.load_state_dict(model_.bert.state_dict(), strict=False)

    # model = BertForSequenceClassification.from_pretrained("bert-base-uncased", problem_type="single_label_classification", num_labels=80)

    # for idx, (name, para) in enumerate(model.named_parameters()):
    #     if idx < 197 or idx>208:
    #         pass
    #     else:
    #         para.requires_grad = False

    for idx, (name, para) in enumerate(model.named_parameters()):
        print(idx, name, para.requires_grad)
    # model.param = load_file('./param/mode_para_19000')

    model.to(device)

    t_total = num_train_steps

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    global_step = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, args.threshold)



        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        # print([f.rel_id for f in train_features])
        all_rel_ids = torch.stack([torch.tensor(f.rel_id).long() for f in train_features], dim=0)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_label_ids, all_rel_ids)

        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)
        accumulation_steps = Config.accumulation_steps
        print(accumulation_steps)
        output_loss_file = os.path.join(args.output_dir, "loss")
        loss_fout = open(output_loss_file, 'w')
        model.train()
        loss_print = 0
        for epoch_idx in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                # batch[-1][batch[-1] == -1] = 410 * Config.rel_num
                batch = tuple(t.to(device) for i, t in enumerate(batch))
                input_ids, input_mask, segment_ids, label_ids, rel_ids = batch

                loss = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,  labels=label_ids, rel=rel_ids).loss
                # loss = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,  labels=label_ids).loss
                # print(loss)
                # loss = loss.item()
                loss /= accumulation_steps
                loss_print += loss
                # if n_gpu > 1:
                #     loss = loss.mean()  # mean() to average on multi-gpu.
                # if args.gradient_accumulation_steps > 1:
                #     loss = loss / args.gradient_accumulation_steps


                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % accumulation_steps == 0:
                    print(loss_print.item())
                    loss_fout.write("{}\n".format(loss_print.item()))
                    optimizer.step()
                    optimizer.zero_grad()
                    loss_print = 0
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(args.output_dir, "pytorch_model.bin_{}".format(epoch_idx))
            torch.save(model_to_save.state_dict(), output_model_file)

        # Save a trained model
        # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        # output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        # torch.save(model_to_save.state_dict(), output_model_file)


if __name__ == "__main__":
    main()