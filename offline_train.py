import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, r2_score
import logging
import math

from LLMServe.global_scheduler.load_predictor import Conversation
from LLMServe.global_scheduler.load_predictor import ResponsePredictor


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    train_logger = logging.getLogger(name)
    train_logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    train_logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    train_logger.addHandler(sh)

    return train_logger


def setup_seed(seed):
    if seed == -1:
        seed = random.randint(0, 1000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def print_args(train_logger, args):
    train_logger.info('--------args----------')
    for k in list(vars(args).keys()):
        train_logger.info('{}: {}'.format(k, vars(args)[k]))
    train_logger.info('--------args----------\n')


def evaluate(dataset: Conversation, model, args, train_logger, device):
    y_pred = []
    y_true_cate = []
    y_true_len = []
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    with torch.no_grad():
        for prompt, response_cate, response_len in tqdm(dataloader):
            
            predict_label = model(prompt, device)
            if args.classification:
                predict_label = torch.argmax(predict_label, dim=-1)
            else:
                predict_label = predict_label.reshape(predict_label.size(0))
            y_pred.append(predict_label.cpu().numpy())
            y_true_cate.append(response_cate.cpu().numpy())
            y_true_len.append(response_len.cpu().numpy())
        
        y_pred = np.concatenate(y_pred)
        y_true_cate = np.concatenate(y_true_cate)
        y_true_len = np.concatenate(y_true_len)
        
        if args.classification:
            accuracy = accuracy_score(y_true_cate, y_pred)
            f1 = f1_score(y_true_cate, y_pred, average='macro')
            precision = precision_score(y_true_cate, y_pred, average='macro')
            recall = recall_score(y_true_cate, y_pred, average='macro')
            train_logger.info("accuracy = {:.6f}, f1 = {:.6f}, precision = {:.6f}, recall = {:.6f}".format(accuracy, f1, precision, recall))
            
            response_bins = dataset.response_bins
            response_bins.append(args.max_len)
            
            y_pred_len_right = np.floor(np.array([response_bins[y_pred_i] for y_pred_i in y_pred]))
            response_bins = [0] + response_bins
            y_pred_len_left = np.floor(np.array([response_bins[y_pred_i] for y_pred_i in y_pred]))
            mid_y_pred_len = np.floor((y_pred_len_left + y_pred_len_right) / 2)

            # # left value
            # diff = np.abs(y_pred_len_left-y_true_len)
            # acc_25 = np.sum(diff <= 25) / len(diff)
            # acc_50 = np.sum(diff <= 50) / len(diff)
            # acc_100 = np.sum(diff <= 100) / len(diff)
            # mae = np.mean(diff)
            # train_logger.info("Right Mean Abs Error: {}, Acc-25: {}, Acc-50: {}, Acc-100:{}".format(mae, acc_25, acc_50, acc_100))

            # # right value
            # diff = np.abs(y_pred_len_right-y_true_len)
            # acc_25 = np.sum(diff <= 25) / len(diff)
            # acc_50 = np.sum(diff <= 50) / len(diff)
            # acc_100 = np.sum(diff <= 100) / len(diff)
            # mae = np.mean(diff)
            # train_logger.info("Right Mean Abs Error: {}, Acc-25: {}, Acc-50: {}, Acc-100:{}".format(mae, acc_25, acc_50, acc_100))
            
            # mid value
            diff = np.abs(mid_y_pred_len-y_true_len)
            acc_25 = np.sum(diff <= 25) / len(diff)
            acc_50 = np.sum(diff <= 50) / len(diff)
            acc_100 = np.sum(diff <= 100) / len(diff)
            r2 = r2_score(y_true_len, mid_y_pred_len)
            mae = np.mean(diff)
            train_logger.info("Mid Mean Abs Error: {}, R2 Score: {}, Acc-25: {}, Acc-50: {}, Acc-100:{}".format(mae, r2, acc_25, acc_50, acc_100))
            return accuracy
        else:

            diff = np.abs(y_pred-y_true_len)
            acc_25 = np.sum(diff <= 25) / len(diff)
            acc_50 = np.sum(diff <= 50) / len(diff)
            acc_100 = np.sum(diff <= 100) / len(diff)
            r2 = r2_score(y_true_len, y_pred)
            mae = np.mean(diff)
            train_logger.info("Mean Abs Error: {}, R2 Score: {}, Acc-25: {}, Acc-50: {}, Acc-100:{}".format(mae, r2, acc_25, acc_50, acc_100))
            return -mae


def train(args):

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    args_str = f"{args.response_type}_claases-{args.use_prompt}_prompt-{args.resample}_resample"
    filename = f"{args.log_dir}/{args_str}.log"
    train_logger = get_logger(filename)
    print_args(train_logger, args)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = ResponsePredictor(args.model_name, args.response_type, args.hidden_dim, args.use_prompt, args.prompt_learning_length)
    model = model.to(device)

    train_dataset = Conversation(args.filename, response_type=args.response_type, min_len=args.min_len, max_len = args.max_len, mode='train', resample=args.resample)
    test_dataset = Conversation(args.filename, response_type=args.response_type, min_len=args.min_len, max_len = args.max_len, mode='test')
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(list(model.bert.transformer.layer[-1].parameters()) + list(model.cls.parameters()) + list([model.leanrable_prompts]), lr=args.lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_res =-1000
    
    if args.classification:
        loss_fun = torch.nn.CrossEntropyLoss()
    else:
        loss_fun = torch.nn.MSELoss()
    
    for i in range(1, args.epochs+1):
        loss_list = []
        for prompt, response_cate, response_len in tqdm(train_dataloader):
            
            response_cate = response_cate.to(device)
            response_len = response_len.to(device)
            
            pred = model(prompt, device)
            if not args.classification:
                pred = pred.reshape(pred.size(0))
                response_len = response_len.float()
                loss = loss_fun(pred, response_len)
            else:
                loss = loss_fun(pred, response_cate)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_list.append(loss.item())
        
        train_logger.info("Epoch: {}/{}, loss = {:.6f}".format(i, args.epochs, np.mean(loss_list)))

        train_logger.info("Evaluate test dataset...")
        res = evaluate(test_dataset, model, args, train_logger, device)

        model_path = f"{args.save_dir}/{args_str}.pth"
        if res > best_res:
            best_res = res
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            train_logger.info(f"Save model to {model_path}")
            torch.save({"last_layer": model.bert.transformer.layer[-1], "cls": model.cls, "prompt": model.leanrable_prompts}, model_path)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train Predictor')
    
    parser.add_argument('--filename', type=str, default='./data/datasets/ShareGPT/cleaned.csv', help='training data path')
    
    parser.add_argument('--log_dir', type=str, default="./train_log", help='model')

    parser.add_argument('--save_dir', type=str, default="./saved_model", help='model saving directory')
    
    parser.add_argument('--model_name', type=str, default="distilbert/distilbert-base-uncased", help='model')
    
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    
    parser.add_argument('--lr', type=float, default=0.0001, help='learning tate')
    
    parser.add_argument('--epochs', type=int, default=100, help='training epoch')
    
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dim')
    
    parser.add_argument('--seed', type=int, default=128, help='seed')
    
    parser.add_argument('--use_prompt', type=int, default=1)
    
    parser.add_argument('--prompt_learning_length', type=int, default=12)
    
    parser.add_argument('--response_type', type=int, default=1)
    
    parser.add_argument('--min_len', type=int, default=16)
    parser.add_argument('--max_len', type=int, default=4096)
    
    parser.add_argument('--resample', type=int, default=1)
    
    args = parser.parse_args()
    args.classification = args.response_type > 1
    
    args.seed = setup_seed(args.seed)
    train(args)
