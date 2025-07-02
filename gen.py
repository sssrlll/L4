import torch
from torch.utils.data import DataLoader
from torch import nn
from pytorch_transformers import AdamW, WEIGHTS_NAME, WarmupLinearSchedule
import csv
import random
import numpy as np
import os
import logging
from fp16 import FP16_Module, FP16_Optimizer
from parallel import DataParallelModel, DataParallelCriterion
from collections import OrderedDict
from utils import *
from settings import args, TASK_DICT, init_logging, MODEL_CONFIG, MODEL_CLASS, SPECIAL_TOKENS, CONFIG_CLASS
from settings import TOKENIZER, SPECIAL_TOKEN_IDS, FILL_VAL, SAVE_NAME, FINAL_SAVE_NAME, TOKENS_WEIGHT, CONFIG_NAME
from scheduler import AnnealingLR
from regularizers import REG_TYPES, REG_TYPE_KEYS, Weight_Regularized_AdamW, Weight_Regularized_SGD
from torch.nn import CrossEntropyLoss
logger = logging.getLogger(__name__)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def swap_name(org_name, seq_distil, ref1):
    # swap_name(TASK_DICT[t]["train"], args.seq_distil, args.ref1)
    if not seq_distil and not ref1:
        return org_name
    if seq_distil:
        return org_name.replace("train", "distil")
    if ref1:
        return org_name.replace("train", "ref1")


def train(task_ids, model_path):
    # tasks = [args.tasks[task_id] for task_id in task_ids]

    # logger.info("start to train { task: %s, seq train type: %s }" % (tasks, args.seq_train_type))
    # model_dir = get_model_dir(tasks)
    # make_dir(model_dir)

    #train_dataset = [(TASK_DICT[t]["train"] if not args.seq_distil else TASK_DICT[t]["train"].replace("train", "distil")) for t in tasks]
    # train_dataset = [swap_name(TASK_DICT[t]["train"], args.seq_distil, args.ref1) for t in tasks]
    train_extra_data = []

    model = MODEL_CLASS.from_pretrained(model_path).cuda()
    
    for task in ['woz.en', 'cnn_dailymail', 'wikisql']:
        gen_token = get_gen_token(task)
        TOKENIZER.add_tokens([gen_token])
        # TOKENIZER.save_pretrained(model_dir)
        SPECIAL_TOKENS[task] = gen_token
        SPECIAL_TOKEN_IDS[task] = TOKENIZER.convert_tokens_to_ids(gen_token)
        logger.info('gen token = {} , gen token id = {}'.format(gen_token, SPECIAL_TOKEN_IDS[task]))
        # MODEL_CONFIG.vocab_size = len(TOKENIZER)
        # MODEL_CONFIG.to_json_file(os.path.join(model_dir,CONFIG_NAME))
        # global TOKENS_WEIGHT
        # if len(TOKENIZER) != TOKENS_WEIGHT.shape[0]:
            # TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, torch.ones([1]).cuda()))

    if "lll" in args.seq_train_type and task_ids[0] > 0 and not args.skip_tasks:
        prev_task = args.tasks[task_ids[0]-1]
        with torch.no_grad():
            create_extra_data_gen(None, prev_task, model, train_extra_data)
    elif "gem" in args.seq_train_type and task_ids[0] > 0: 
        get_real_data(tasks[0], train_extra_data, accum=False, encode=True)
        args.memory_data.append(train_extra_data)
        train_extra_data = []
    logger.info('extra training data size: {}'.format(len(train_extra_data)))

    


if __name__ == '__main__':

    if not args.debug:
        logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)
        logging.getLogger("pytorch_transformers.tokenization_utils").setLevel(logging.CRITICAL)

    make_dir(args.model_dir_root)

    init_logging(os.path.join(args.model_dir_root, 'log_train.txt'))
    logger.info('args = {}'.format(str(args)))

    model = None
    if args.seq_train_type in ["multitask", "multilm"]:
        model = train(list(range(len(args.tasks))), model)
    else:
        if args.unbound:
            TASK_DICT = lll_unbound_setting(split_size=args.unbound)

        model = train([len(args.tasks)], '/home/jyli/L2KD-L4/models/gpt2/lll/woz.en_cnn_dailymail_wikisql_0.2_531_2/wikisql')
