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
from settings import args, TASK_DICT, init_logging, MODEL_CONFIG, MODEL_CLASS, SPECIAL_TOKENS, CONFIG_CLASS, TOKENIZER_CLASS
from settings import TOKENIZER, SPECIAL_TOKEN_IDS, FILL_VAL, SAVE_NAME, FINAL_SAVE_NAME, TOKENS_WEIGHT, CONFIG_NAME
from scheduler import AnnealingLR
from regularizers import REG_TYPES, REG_TYPE_KEYS, Weight_Regularized_AdamW, Weight_Regularized_SGD
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import gc
# from transformers import BloomModel, GPTNeoModel, GPTJModel

logger = logging.getLogger(__name__)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
print(args)
# torch.cuda.set_device(args.device_ids[-1])
# os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(item) for item in args.device_ids])

def swap_name(org_name, seq_distil, ref1):
    # swap_name(TASK_DICT[t]["train"], args.seq_distil, args.ref1)
    if not seq_distil and not ref1:
        return org_name
    if seq_distil:
        return org_name.replace("train", "distil")
    if ref1:
        return org_name.replace("train", "ref1")
    
def align_wte_and_lmhead(lm_module, wte_module):
    new_lm = nn.Linear(in_features=lm_module.in_features, out_features=wte_module.num_embeddings, bias=False)
    new_lm.to(wte_module.weight.device)
    new_lm.weight.data[:, :] = wte_module.weight.data[:, :]
    return new_lm


def train(task_ids, model, teacher_model):
    tasks = [args.tasks[task_id] for task_id in task_ids]

    logger.info("start to train { task: %s, seq train type: %s }" % (tasks, args.seq_train_type))
    model_dir = get_model_dir(tasks)
    make_dir(model_dir)
    print('construct train dataset')
    #train_dataset = [(TASK_DICT[t]["train"] if not args.seq_distil else TASK_DICT[t]["train"].replace("train", "distil")) for t in tasks]
    train_dataset = [swap_name(TASK_DICT[t]["train"], args.seq_distil, args.ref1) for t in tasks]
    train_extra_data = []
    
    # when task_ids only include one task.
    if task_ids[0] != 0 :
        # print('when begin task not 0')
        prev_model_dir = get_model_dir([args.tasks[task_ids[0] - 1]])
        print('prev_model_dir', prev_model_dir)
        prev_model_path = os.path.join(prev_model_dir, FINAL_SAVE_NAME)
        print('prev_model_path', prev_model_path)
        prev_config_path = os.path.join(prev_model_dir, CONFIG_NAME)
        prev_model_config = CONFIG_CLASS.from_json_file(prev_config_path)
        model = MODEL_CLASS(prev_model_config).cuda()
        prev_state_dict = torch.load(prev_model_path, map_location="cpu")
        model.load_state_dict(prev_state_dict)
            
        global TOKENIZER
        TOKENIZER = TOKENIZER_CLASS.from_pretrained(prev_model_dir)
            
        print('tie_weights', hasattr(model, "tie_weights"))
    else:
        # print('when begin task is 0')
        model = MODEL_CLASS.from_pretrained(args.model_name).cuda()
        print('student', model)
        model.resize_token_embeddings(len(TOKENIZER))
        # which_model_to_load = model_dir if os.path.isfile(os.path.join(model_dir, FINAL_SAVE_NAME)) else args.model_name
    # print('len tokenizer', len(TOKENIZER))
    # print('finish 1')
    # print("lll" in args.seq_train_type)
    # print(task_ids[0])
    # print(args.skip_tasks)
    if "lll" in args.seq_train_type and task_ids[0] > 0 and not args.skip_tasks:
        # print('start to create')
        prev_task = args.tasks[task_ids[0]-1]
        with torch.no_grad():
            create_extra_data(task_ids, tasks[0], prev_task, model, train_extra_data)
    elif "gem" in args.seq_train_type and task_ids[0] > 0: 
        get_real_data(tasks[0], train_extra_data, accum=False, encode=True)
        args.memory_data.append(train_extra_data)
        train_extra_data = []
    logger.info('extra training data size: {}'.format(len(train_extra_data)))
    
    gen_token = get_gen_token(tasks[0])
    TOKENIZER.add_tokens([gen_token])
    TOKENIZER.save_pretrained(model_dir)
    SPECIAL_TOKENS[tasks[0]] = gen_token
    SPECIAL_TOKEN_IDS[tasks[0]] = TOKENIZER.convert_tokens_to_ids(gen_token)
    logger.info('gen token = {} , gen token id = {}'.format(gen_token, SPECIAL_TOKEN_IDS[tasks[0]]))
    MODEL_CONFIG.vocab_size = len(TOKENIZER)
    MODEL_CONFIG.to_json_file(os.path.join(model_dir,CONFIG_NAME))
    # print('finish 2')

    if args.skip_tasks and len(tasks) == 1:
        logger.info("*********** skip task: {} ***********".format(tasks[0]))
        if tasks[0] in args.skip_tasks:
            if len(args.skip_tasks) == 1:
                model_dir = get_model_dir(tasks)
                model_path = os.path.join(model_dir, FINAL_SAVE_NAME)
                config_path = os.path.join(model_dir,CONFIG_NAME)
                model_config = CONFIG_CLASS.from_json_file(config_path)
                model = MODEL_CLASS(model_config).cuda()
                state_dict = torch.load(model_path, map_location="cpu")
                model.load_state_dict(state_dict)
                if not args.fp32:
                    model = FP16_Module(model)
                if args.seq_train_type in REG_TYPE_KEYS:
                    logger.info("calulating reg_params ...")
                    train_qadata = QADataset(train_dataset, "train", SPECIAL_TOKEN_IDS[tasks[0]], train_extra_data)
                    max_train_batch_size = max(len(train_qadata) // args.min_n_steps, args.min_batch_size)
                    train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)
                    parallel_model = DataParallelModel(WrapModel(model), args.device_ids)
                    # parallel_model = DataParallelModel(WrapModel(model))
                    regularizer = REG_TYPES[args.seq_train_type](model, parallel_model, [train_dataloader], tasks[0])
                    regularizer.task_start_do()
                    regularizer.task_end_do()
                    torch.save(model.state_dict(), os.path.join(model_dir, FINAL_SAVE_NAME))
                    logger.info("done reg_params!")
            args.skip_tasks.remove(tasks[0])
            return model
    
    print('stu model', model)
    print('len tokenizer', len(TOKENIZER))
    
    model.resize_token_embeddings(len(TOKENIZER) if not args.multitask_specific else len(TOKENIZER)+4)
    # model.lm_head.weight = model.transformer.wte.weight
    print("resize stu", model)
    
    model.lm_head = align_wte_and_lmhead(model.lm_head, model.transformer.wte)
    print('wte', model.transformer.wte.weight.shape)
    print('lm_head', model.lm_head.weight.shape)
    if not args.fp32:
        model = FP16_Module(model)
    global TOKENS_WEIGHT
    if len(TOKENIZER) != TOKENS_WEIGHT.shape[0]:
        for i in range((len(TOKENIZER) - TOKENS_WEIGHT.shape[0])):
            TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, torch.ones([1]).cuda()))
        
    print("resize stu", model)
    if args.multitask_specific:
        for i in range(4):
            TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, torch.ones([1]).cuda()))
    if args.distil:
        teacher_model = MODEL_CLASS.from_pretrained(args.model_name).cuda()
        teacher_vocab_size = 0
        if args.pretrained_teacher:
            teacher_vocab_size = json.load(open("models/gpt2/lll/{task}_0.2_531/{task}/config.json".format(task=tasks[0])))['vocab_size']
            teacher_model.resize_token_embeddings(teacher_vocab_size)
            teacher_model.lm_head = align_wte_and_lmhead(teacher_model.lm_head, teacher_model.transformer.wte)
            print("load teacher model from {}".format("models/gpt2/lll/{task}_0.2_531/{task}/model-finish".format(task=tasks[0])))
            teacher_model.load_state_dict(torch.load("models/gpt2/lll/{task}_0.2_531/{task}/model-finish".format(task=tasks[0]), map_location="cpu"))

        elif task_ids[0] != 0:
            # load checkpoint
            # print('prev task id', task_ids[0]-1)
            # print("tasks", args.tasks)
            prev_task = args.tasks[task_ids[0]-1]

            teacher_vocab_size = json.load(open(os.path.join(args.model_dir_root+"/{task}/config.json".format(task=prev_task))))['vocab_size']
            teacher_model.resize_token_embeddings(teacher_vocab_size)
            teacher_model.lm_head = align_wte_and_lmhead(teacher_model.lm_head, teacher_model.transformer.wte)
            print("load teacher model from {}".format(os.path.join(args.model_dir_root+"/{task}/teacher/".format(task=prev_task), SAVE_NAME+str(args.n_train_epochs[prev_task]))))
            teacher_model.load_state_dict(torch.load(os.path.join(args.model_dir_root+"/{task}/teacher/".format(task=prev_task), SAVE_NAME+str(args.n_train_epochs[prev_task])), map_location="cpu"))
            # resize model
            teacher_vocab_size = len(TOKENIZER)
            teacher_model.resize_token_embeddings(teacher_vocab_size)
            teacher_model.lm_head = align_wte_and_lmhead(teacher_model.lm_head, teacher_model.transformer.wte)

        else:
            # teacher shares tokenizer and config with student
            teacher_vocab_size = len(TOKENIZER)
            teacher_model.resize_token_embeddings(teacher_vocab_size)
            teacher_model.lm_head = align_wte_and_lmhead(teacher_model.lm_head, teacher_model.transformer.wte)
 
        if not args.fp32:
            teacher_model = FP16_Module(teacher_model)
        teacher_model.eval()
        parallel_teacher_model = DataParallelModel(WrapModel(teacher_model), args.device_ids)
        print('parallel_teacher_model.device_ids', parallel_teacher_model.device_ids)
        # parallel_teacher_model = DataParallelModel(WrapModel(teacher_model))

    if not args.fp32:  # again because resize_token_embeddings makes embedding layer fp32
        model = FP16_Module(model)

    parallel_model = DataParallelModel(WrapModel(model), args.device_ids)
    # parallel_model = DataParallelModel(WrapModel(model))
    print('parallel_model.device_ids', parallel_model.device_ids)

    train_qadata = QADataset(train_dataset, "train", SPECIAL_TOKEN_IDS[tasks[0]], train_extra_data)
    max_train_batch_size = max(len(train_qadata) // args.min_n_steps, args.min_batch_size)
    train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)
    if not args.unbound and args.seq_train_type not in ["multitask", "multilm"]:
        #n_train_epochs = TASK_DICT[tasks[0]]["n_train_epochs"]
        n_train_epochs = args.n_train_epochs[tasks[0]]
    else:
        n_train_epochs = args.n_train_epochs['_'.join(tasks)]
    n_train_optimization_steps = len(train_qadata) * n_train_epochs
    logger.info('len of train dataset: {} , max train batch size {} , num of opt steps: {}'.format(
        len(train_qadata), max_train_batch_size, n_train_optimization_steps))

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.mutual_distil:
        # teacher optimizer
        teacher_param_optimizer = list(teacher_model.named_parameters())
        teacher_optimizer_grouped_parameters = [
            {'params': [p for n, p in teacher_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in teacher_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]


    if "gem" in args.seq_train_type:
        model.task_id = task_ids[0]
        if not hasattr(model, "grad_dims"):
            model.grad_dims = []
            for param in model.parameters():
                model.grad_dims.append(param.data.numel())
        if not hasattr(model, "grads"):
            model.grads = torch.zeros(sum(model.grad_dims),len(args.tasks))
            model.grads = model.grads.cuda()

    if "gem" in args.seq_train_type and args.mutual_distil:
        teacher_model.task_id = task_ids[0]
        if not hasattr(teacher_model, "grad_dims"):
            teacher_model.grad_dims = []
            for param in teacher_model.parameters():
                teacher_model.grad_dims.append(param.data.numel())
        if not hasattr(teacher_model, "grads"):
            teacher_model.grads = torch.zeros(sum(teacher_model.grad_dims),len(args.tasks))
            teacher_model.grads = teacher_model.grads.cuda()
    

    if args.seq_train_type in REG_TYPE_KEYS:
        optimizer = Weight_Regularized_AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if not args.fp32:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=None, dynamic_loss_scale=True,
                                   dynamic_loss_args={'scale_window': 100, 'min_scale': 1, 'delayed_shift': 2})

    if args.mutual_distil:
        if args.seq_train_type in REG_TYPE_KEYS:
            teacher_optimizer = Weight_Regularized_AdamW(teacher_optimizer_grouped_parameters, lr=args.teacher_learning_rate, eps=args.adam_epsilon)
        else:
            teacher_optimizer = AdamW(teacher_optimizer_grouped_parameters, lr=args.teacher_learning_rate, eps=args.adam_epsilon)
        if not args.fp32:
            teacher_optimizer = FP16_Optimizer(teacher_optimizer, static_loss_scale=None, dynamic_loss_scale=True,
                                    dynamic_loss_args={'scale_window': 100, 'min_scale': 1, 'delayed_shift': 2})
        teacher_scheduler = AnnealingLR(teacher_optimizer, start_lr=args.teacher_learning_rate, warmup_iter=int(args.n_warmup_ratio*len(train_qadata)),
            num_iters=int(n_train_optimization_steps), decay_style=args.decay_style)

    scheduler = AnnealingLR(optimizer, start_lr=args.learning_rate, warmup_iter=int(args.n_warmup_ratio*len(train_qadata)),
            num_iters=int(n_train_optimization_steps), decay_style=args.decay_style)
    print('token_weight.shape', TOKENS_WEIGHT.shape)
    train_loss_fct = DataParallelCriterion(CrossEntropyLoss(ignore_index=FILL_VAL, weight=TOKENS_WEIGHT), args.device_ids)
    # train_loss_fct = DataParallelCriterion(CrossEntropyLoss(ignore_index=FILL_VAL, weight=TOKENS_WEIGHT))
    if args.distil:
        kd_loss_fct = DataParallelCriterion(nn.KLDivLoss(reduction="mean"), args.device_ids)
        # kd_loss_fct = DataParallelCriterion(nn.KLDivLoss(reduction="batchmean"))

    if args.seq_train_type in REG_TYPE_KEYS:
        copy_train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)
        prev_task = args.tasks[task_ids[0]-1]
        regularizer = REG_TYPES[args.seq_train_type](model, parallel_model, [copy_train_dataloader], tasks[0], prev_task)
        if args.mutual_distil:
            teacher_regularizer = REG_TYPES[args.seq_train_type](teacher_model, parallel_teacher_model, [copy_train_dataloader], tasks[0], prev_task)
            teacher_regularizer.task_start_do()

    tot_n_steps = 0
    if not os.path.exists(os.path.join(model_dir, "student")):
        os.mkdir(os.path.join(model_dir, "student"))
    if not os.path.exists(os.path.join(model_dir, "teacher")):
        os.mkdir(os.path.join(model_dir, "teacher"))
    train_once = TrainStep(model, optimizer, scheduler)
    train_teacher_once = TrainStep(teacher_model, teacher_optimizer, teacher_scheduler)
    if "gem" in args.seq_train_type and task_ids[0] != 0:
        gem_step = GEMStep(model, parallel_model, train_loss_fct, optimizer)
        # teacher_gem_step = GEMStep(teacher_model, parallel_teacher_model, train_loss_fct, teacher_optimizer)
    # model.train()
    for ep in tqdm(range(n_train_epochs)):
        model.train()
        if args.distil:
            teacher_model.eval()
        cum_loss, cum_qa_loss, cum_lm_loss, cur_n_inputs = 0, 0, 0, 0
        for n_steps, (_, _, cqa, _, Y, gen_X, gen_Y, is_extra) in tqdm(enumerate(train_dataloader)):

            n_inputs = sum(_cqa.shape[0] for _cqa in cqa)
            if args.multitask_specific:
                for i in range(len(is_extra)):
                    gen_X[i][:, 0] += is_extra[i]
                    is_extra[i] = is_extra[i] * 0
            
            # print('len(cqa)', len(cqa))
            for i in range(len(cqa)):

                cqa[i] = (cqa[i].to(args.device_ids[i]),)
                Y[i] = Y[i].to(args.device_ids[i])
                gen_X[i] = (gen_X[i].to(args.device_ids[i]),)
                gen_Y[i] = gen_Y[i].to(args.device_ids[i])
                is_extra[i] = is_extra[i].to(args.device_ids[i])
               
            # 每个epoch添加对学生蒸馏loss
            if args.distil:
                losses = get_distil_losses(parallel_teacher_model, parallel_model, cqa, Y, gen_X, gen_Y, is_extra, kd_loss_fct, train_loss_fct, args.temperature_kd, pad_idx=FILL_VAL)
            else:
                losses = get_losses(parallel_model, cqa, Y, gen_X, gen_Y, train_loss_fct)
            loss = sum(losses)
            if "gem" in args.seq_train_type and task_ids[0] != 0:
                gem_step(task_ids[0])
            train_once(loss, n_inputs)

            qa_loss = losses[0].item() * n_inputs
            lm_loss = losses[1]. item() * n_inputs
            cum_loss += (qa_loss + lm_loss)
            cum_qa_loss += qa_loss
            cum_lm_loss += lm_loss
            cur_n_inputs += n_inputs

            if (n_steps + 1 ) % args.logging_steps == 0:
                logger.info('progress {:.3f} , lr {:.1E} , loss {:.3f} , qa loss {:.3f} , lm loss {:.3f} , avg batch size {:.1f}'.format(
                    ep + cur_n_inputs/len(train_qadata), scheduler.get_lr(), cum_loss/cur_n_inputs, cum_qa_loss/cur_n_inputs, cum_lm_loss/cur_n_inputs,
                    cur_n_inputs/(n_steps + 1)
                ))
            
            del cqa
            del Y
            del gen_X
            del gen_Y
            del is_extra
            
        
        torch.save(model.state_dict(), os.path.join(model_dir+"/student/", SAVE_NAME+str(ep+1)))
        tot_n_steps += (n_steps + 1)
        logger.info('epoch {}/{} done , tot steps {} , lr {:.1E} , loss {:.2f} , qa loss {:.2f} , lm loss {:.2f} , avg batch size {:.1f}'.format(
            ep+1, n_train_epochs, tot_n_steps, scheduler.get_lr(), cum_loss/cur_n_inputs, cum_qa_loss/cur_n_inputs, cum_lm_loss/cur_n_inputs, cur_n_inputs/(n_steps+1)
        ))

        if args.distil and args.mutual_distil:
            model.eval()
            teacher_model.train()
            cum_teacher_loss, cum_teacher_qa_loss, cum_teacher_lm_loss, cur_teacher_n_inputs = 0, 0, 0, 0
            for n_steps, (_, _, cqa, _, Y, gen_X, gen_Y, is_extra) in tqdm(enumerate(train_dataloader)):
    
                n_inputs = sum(_cqa.shape[0] for _cqa in cqa)
                if args.multitask_specific:
                    for i in range(len(is_extra)):
                        gen_X[i][:, 0] += is_extra[i]
                        is_extra[i] = is_extra[i] * 0

                for i in range(len(cqa)):
                    cqa[i] = (cqa[i].to(args.device_ids[i]),)
                    Y[i] = Y[i].to(args.device_ids[i])
                    gen_X[i] = (gen_X[i].to(args.device_ids[i]),)
                    gen_Y[i] = gen_Y[i].to(args.device_ids[i])
                    is_extra[i] = is_extra[i].to(args.device_ids[i])

                # 每个epoch添加对教师蒸馏loss
                losses = get_distil_losses(parallel_model, parallel_teacher_model, cqa, Y, gen_X, gen_Y, is_extra, kd_loss_fct, train_loss_fct, args.teacher_temperature_kd, pad_idx=FILL_VAL, is_teacher=True)
                loss = sum(losses)
                if "gem" in args.seq_train_type and task_ids[0] != 0:
                    gem_step(task_ids[0])
                train_teacher_once(loss, n_inputs)

                qa_loss = losses[0].item() * n_inputs
                lm_loss = losses[1].item() * n_inputs
                # cum_loss += (qa_loss + lm_loss)
                cum_teacher_loss += (qa_loss + lm_loss)
                # cum_qa_loss += qa_loss
                cum_teacher_qa_loss += qa_loss
                # cum_lm_loss += lm_loss
                cum_teacher_lm_loss += lm_loss
                # cur_n_inputs += n_inputs
                cur_teacher_n_inputs += n_inputs

                if (n_steps + 1 ) % args.logging_steps == 0:
                    logger.info('progress {:.3f} , lr {:.1E} , loss {:.3f} , qa loss {:.3f} , lm loss {:.3f} , avg batch size {:.1f}'.format(
                        ep + cur_teacher_n_inputs/len(train_qadata), teacher_scheduler.get_lr(), cum_teacher_loss/cur_teacher_n_inputs, cum_teacher_qa_loss/cur_teacher_n_inputs, cum_teacher_lm_loss/cur_teacher_n_inputs,
                        cur_teacher_n_inputs/(n_steps + 1)
                    ))
                
                del cqa
                del Y
                del gen_X
                del gen_Y
                del is_extra

            torch.save(teacher_model.state_dict(), os.path.join(model_dir+"/teacher/", SAVE_NAME+str(ep+1)))
            tot_n_steps += (n_steps + 1)
            logger.info('teacher_model epoch {}/{} done , tot steps {} , lr {:.1E} , loss {:.2f} , qa loss {:.2f} , lm loss {:.2f} , avg batch size {:.1f}'.format(
            ep+1, n_train_epochs, tot_n_steps, teacher_scheduler.get_lr(), cum_teacher_loss/cur_teacher_n_inputs, cum_teacher_qa_loss/cur_teacher_n_inputs, cum_teacher_lm_loss/cur_teacher_n_inputs, cur_teacher_n_inputs/(n_steps+1)
            ))
        torch.cuda.empty_cache()
        gc.collect()
           
    # task end do for reg
    if args.seq_train_type in REG_TYPE_KEYS:
        regularizer.task_end_do()
    torch.save(model.state_dict(), os.path.join(model_dir, FINAL_SAVE_NAME))

    if args.distil and args.mutual_distil:  
        del parallel_teacher_model
        del teacher_model
        del parallel_model
        del model

    del optimizer_grouped_parameters
    del teacher_optimizer_grouped_parameters
    del optimizer
    del teacher_optimizer
    del param_optimizer
    del teacher_param_optimizer
    torch.cuda.empty_cache()
    gc.collect()
    

    if args.distil and args.mutual_distil:
        return None, None
    return model, None



if __name__ == '__main__':

    if not args.debug:
        logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)
        logging.getLogger("pytorch_transformers.tokenization_utils").setLevel(logging.CRITICAL)

    make_dir(args.model_dir_root)

    init_logging(os.path.join(args.model_dir_root, 'log_train.txt'))
    logger.info('args = {}'.format(str(args)))

    model = None
    teacher_model = None
    if args.seq_train_type in ["multitask", "multilm"]:
        model = train(list(range(len(args.tasks))), model)
    else:
        if args.unbound:
            TASK_DICT = lll_unbound_setting(split_size=args.unbound)
        for task_id in range(args.begin_task, len(args.tasks)):
            model, teacher_model = train([task_id], model, teacher_model)
