import torch
import csv
import os
import json
import logging
from fp16 import FP16_Module
import GPUtil
from collections import OrderedDict
from settings import args, MODEL_CLASS, TOKENIZER, SPECIAL_TOKEN_IDS, init_logging
from settings import MEMORY_FACTOR, LEN_FACTOR, TASK_DICT, MODEL_CONFIG, DATA_ATTRS, SPECIAL_TOKENS, CONFIG_CLASS, CONFIG_NAME, TOKENIZER_CLASS
from utils import QADataset, top_k_top_p_filtering, create_dataloader, logits_to_tokens, get_model_dir
from utils import sample_sequence, remove_id, get_gen_token, lll_unbound_setting
from metrics import compute_metrics
from tqdm import tqdm
logger = logging.getLogger(__name__)

import pdb
# os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(item) for item in args.device_ids])

def test_one_to_one(task_load, task_eval, model, score_dict):

    logger.info("start to test { task: %s (load) %s (eval), seq train type: %s }" % (task_load, task_eval, args.seq_train_type))
    if hasattr(args, 'extra_e2e'):
        if args.extra_e2e and task_eval=='e2enlg':
            print("USE extra_e2e!", flush=True)
            TASK_DICT[task_eval]["test"] = TASK_DICT[task_eval]["test"].replace('test', 'extra')
    test_qadata = QADataset(TASK_DICT[task_eval]["test"] if not args.test_training_set else TASK_DICT[task_eval]["train"] , "test", SPECIAL_TOKEN_IDS[task_load]).sort()
    max_a_len = test_qadata.max_a_len
    test_dataloader = create_dataloader(test_qadata, "test")
    n_examples = len(test_qadata)
    logger.info("len of test dataset: {}".format(n_examples))

    need_process = OrderedDict()
    qa_results = [0 for _ in range(n_examples)]
    all_pasts = [[0 for _ in range(n_examples)] for __ in range(MODEL_CONFIG.n_layer)]
    max_tot_lens = [0 for _ in range(n_examples)]

    cnt = 0
    forbidden_token_ids = [v for k, v in SPECIAL_TOKEN_IDS.items() if k in args.tasks]
    for n_steps, (cqs, len_cqs, _, _, _, _, _, _) in tqdm(enumerate(test_dataloader)):
        # assume n_gpus == 1
        cqs = cqs[0]
        len_cqs = len_cqs[0]
        n_inputs = cqs.shape[0]
        cqs_cuda = cqs.cuda()
        all_outputs = model(input_ids=cqs_cuda)
        outputs = all_outputs[0]
        if args.model_name == "gpt2":
            pasts = all_outputs[1]
        next_logits = outputs[range(n_inputs), len_cqs-1, :] / args.temperature_qa
        next_tokens = logits_to_tokens(next_logits, forbidden_token_ids).cpu()

        for i in range(n_inputs):
            max_tot_lens[cnt] = max_a_len + test_qadata[cnt][1]
            qa_results[cnt] = cqs[i][:len_cqs[i]]
            if next_tokens[i] != SPECIAL_TOKEN_IDS["eos_token"]:
                qa_results[cnt] = torch.cat((cqs[i][:len_cqs[i]], next_tokens[i]))
                if len(qa_results[cnt]) not in [max_tot_lens[cnt], args.max_len]:
                    need_process.update([[cnt, None]])
                    if args.model_name == "gpt2":
                        for layer_id in range(MODEL_CONFIG.n_layer):
                            all_pasts[layer_id][cnt] = pasts[layer_id][:, i, ..., :len_cqs[i], :].type(torch.float32 if args.fp32 else torch.half)
            cnt += 1

        if len(need_process) > int(10 * args.memory_sizes[0] / cqs.shape[1]):  # dynamic threshold to avoid out of memory
            sample_sequence(model, need_process, qa_results, all_pasts, max_tot_lens)

        del cqs_cuda
        del all_outputs
        torch.cuda.empty_cache()

    sample_sequence(model, need_process, qa_results, all_pasts, max_tot_lens)

    if task_eval in ['wikisql','woz.en','multinli.in.out']:
        ids = test_qadata.get_indices()
        test_qadata.sort_by_index()
        qa_results = [x[1] for x in sorted([(i, g) for i, g in zip(ids, qa_results)])]
    # tokenizer = TOKENIZER_CLASS.from_pretrained("/home/jyli/L2KD/models/gpt2/lll/woz.en_cnn_dailymail_wikisql_0.2_531_2/wikisql")
    tokenizer_path = args.model_dir_root + '/wikisql'
    tokenizer = TOKENIZER_CLASS.from_pretrained(tokenizer_path)
    for i in range(len(test_qadata)):
        _, len_cq, _, _, Y, _, _, hashcode, _ = test_qadata[i]
        if task_eval in ['wikisql','woz.en']:
            Y = test_qadata.answers[i] if not args.test_training_set else test_qadata.answers[0]
        else:
            Y = list(filter(lambda x: x != -1, Y))[:-1]  # remove eos
            Y = ' '.join([str(y) for y in Y]).split(str(SPECIAL_TOKEN_IDS["pad_token"]))
            Y = [tokenizer.decode(list(map(int, y.split()))) for y in Y]
        if not args.test_training_set:
            if qa_results[i].tolist()[len_cq:]:
                # print(qa_results[i].tolist()[len_cq:])
                qa_results[i] = [tokenizer.decode(qa_results[i].tolist()[len_cq:]), Y]
            else:
                qa_results[i] = ["", Y]

        else:
            qa_results[i] = [tokenizer.decode(qa_results[i].tolist()[len_cq:]), Y, hashcode]
    get_test_score(task_eval, qa_results, score_dict)

    model_dir = model.model_dir
    ep = model.ep
    results_path = os.path.join(model_dir,"qa_{}_{}.csv".format(task_eval,ep+1) if not args.test_training_set else "qa_trainset_{}_{}.csv".format(task_eval,ep+1))
    if not args.debug:
        with open(results_path, "w",encoding="utf-8") as f:
            qa_writer = csv.writer(f,delimiter=',')
            if not args.test_training_set:
                qa_writer.writerow(["y","pred"])
                for pred, y in qa_results:
                    if task_eval == 'wikisql': 
                        y = y["answer"]
                    elif task_eval == 'woz.en': 
                        y = y[1]
                    qa_writer.writerow([y,pred])
            else:
                qa_writer.writerow(["y","pred", "hashcode"])
                for pred, y, x in qa_results:
                    if task_eval == 'wikisql': 
                        y = y["answer"]
                    elif task_eval == 'woz.en': 
                        y = y[1]
                    qa_writer.writerow([y,pred,x])

    return model, score_dict

def get_test_score(task_eval,qa_results,score_dict):

    score = compute_metrics(
            qa_results,
            bleu='iwslt.en.de' in task_eval or 'multinli.in.out' in task_eval, #or 'cnn_dailymail' in task_eval #or 'e2enlg' in task_eval or 'rnnlg.tv' in task_eval or 'rnnlg.rest' in task_eval or 'rnnlg.hotel' in task_eval or 'rnnlg.laptop' in task_eval,
            dialogue='woz.en' in task_eval,
            #rouge='cnn_dailymail' in task_eval,
            rouge='cnn_dailymail' in task_eval or 'e2enlg' in task_eval or 'rnnlg.tv' in task_eval or 'rnnlg.rest' in task_eval or 'rnnlg.hotel' in task_eval or 'rnnlg.laptop' in task_eval,
            logical_form='wikisql' in task_eval,
            corpus_f1='zre' in task_eval
    )
    score_dict[task_eval] = score


def test_one_to_many(task_load):
    score_dicts = []
    for ep in range(args.n_train_epochs[task_load]-1, args.n_train_epochs[task_load]) if not args.test_all else range(args.n_train_epochs[task_load]):
        model_dir = get_model_dir([task_load])
        model_path = os.path.join(model_dir, 'student/model-{}'.format(ep+1))
        # model_path = os.path.join(model_dir, 'model-finish')
        config_path = os.path.join(model_dir,CONFIG_NAME)

        gen_token = get_gen_token(task_load)
        TOKENIZER.add_tokens([gen_token])
        SPECIAL_TOKENS[task_load] = gen_token
        SPECIAL_TOKEN_IDS[task_load] = TOKENIZER.convert_tokens_to_ids(gen_token)
        model_config = CONFIG_CLASS.from_json_file(config_path) 
        model = MODEL_CLASS(model_config).cuda().eval()
        if args.multitask_specific:
            model.resize_token_embeddings(len(TOKENIZER)+4)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)

        if not args.fp32:
            model = FP16_Module(model)

        model.ep = ep
        model.model_dir = model_dir
        logger.info("task: {}, epoch: {}".format(task_load, ep+1))
        score_dict = {k:None for k in args.tasks}
        with torch.no_grad():
            for task_eval in args.tasks:
                test_one_to_one(task_load, task_eval, model, score_dict)
        logger.info("score: {}".format(score_dict))
        score_dicts.append(score_dict)

        del model 
        del state_dict
        torch.cuda.empty_cache()

    with open(os.path.join(model_dir, "metrics.json"),"w") as f:
        json.dump(score_dicts, f)


if __name__ == '__main__':
    if args.n_gpus > 1:
        raise NotImplementedError("test can be run with only one gpu currently!")
    
    if args.model_name == "gpt2":
        args.fp32 = False  # always use fp16 in testing

    if not args.debug:
        logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)
        logging.getLogger("pytorch_transformers.tokenization_utils").setLevel(logging.CRITICAL)
    init_logging(os.path.join(args.model_dir_root, 'log_test.txt'))
    logger.info('args = {}'.format(args))

    if args.seq_train_type in ["multitask", "multilm"]:
        test_one_to_many('_'.join(args.tasks))
    else:
        if args.unbound:
            TASK_DICT = lll_unbound_setting(split_size=args.unbound, data_type="test",test_target="origin")
            for idx, task_load in enumerate(args.splitted_tasks):
                test_one_to_many(task_load)
        else:
            for idx, task_load in enumerate(args.tasks):
                if idx >= args.begin_task:
                    test_one_to_many(task_load)
