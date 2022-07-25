import os
import logging
import argparse
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoModelForTokenClassification

from utils import init_logger, load_tokenizer, get_labels

import streamlit as st
import re

logger = logging.getLogger(__name__)

"""
    기존 predict.py 코드는 pred_config ,args로 실행 설정을 위한 입력 인자를 받는 형태로 실행됩니다.
    따라서 명령 프롬프트 등 해당 코드를 실행하기 위해서는 python predict.py --input_file ~ --output_file~~
    형태로 인자를 전달해주어야 했으며, 이부분 수정을 위해 함수의 매개변수 형태로 설정값을 전달하도록 predict.py 코드의 일부 수정하였습니다.

    매개변수의 기본값은 predict.py와 동일하게 설정되도록 초기화하였습니다.
"""


def get_device(device_name="cuda"):  # default device는 cuda로 설정하였습니다.
    return "cuda" if torch.cuda.is_available() and device_name == "cuda" else "cpu"


def get_args(model_dir="./model"):  # default model path 입니다.
    return torch.load(os.path.join(model_dir, "training_args.bin"))


"""
    load_model 함수 실제 사용 시 매개변수 args, device는 위의 get_device, get_args 함수를 통해 전달받습니다.
"""


def load_model(model_dir, args, device):
    # Check whether model exists
    if not os.path.exists(model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_dir
        )  # Config will be automatically loaded from model_dir
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")

    return model


# 파일을 기반으로 입력값을 받아오는 것이 아닌 코드 내에 존재하는 문자열을 입력 가능하도록 일부 수정하였습니다.
def read_input_str(input_text):
    words_list = []
    lines = input_text.split("\n")

    for line in lines:
        line = line.strip()
        words = line.split()
        words_list.append(words)

    print(words_list)

    return words_list


# 기본적으로 파일로 읽어오든 코드 내 문자열을 사용하든 lines에는 단어 list가 담겨있습니다. 따로 수정하지 않았습니다.
def convert_input_file_to_tensor_dataset(
    lines,
    args,
    tokenizer,
    pad_token_label_id,
    cls_token_segment_id=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_slot_label_mask = []

    for words in lines:
        tokens = []
        slot_label_mask = []
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_label_mask.extend([0] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > args.max_seq_len - special_tokens_count:
            tokens = tokens[: (args.max_seq_len - special_tokens_count)]
            slot_label_mask = slot_label_mask[
                : (args.max_seq_len - special_tokens_count)
            ]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        slot_label_mask += [pad_token_label_id]

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        slot_label_mask = [pad_token_label_id] + slot_label_mask

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + (
            [0 if mask_padding_with_zero else 1] * padding_length
        )
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_slot_label_mask.append(slot_label_mask)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask
    )

    return dataset


# 문자열 내 숫자 포함되는 지 체크하기 위한 함수
def hasNumber(stringVal):
    return any(elem.isdigit() for elem in stringVal)


# 개체명 입력 시 앞 내용만 입력했을 때 해당 내용이
def checkNer(str, str_arr):
    isExistNer = False

    print(str)
    print(str_arr)

    if len(str_arr) > 0:
        for i in str_arr:
            if i.find("-") != -1:
                continue
            else:
                if str.find(i) != -1:
                    isExistNer = True

    print(isExistNer)

    return isExistNer


# 현재 부분에서는 간단하게 보여지기 위해서 pred_config 대신 설정을 위한 args들을 직접 함수 매개변수로 전달해주는 형태로 진행했습니다.
# 기존 predict.py 코드처럼 사용할 수 있게 수정하고 싶으시다면 각 매개변수 값에 대해 __main__ 코드 부분에서 문자열 list 형대로 사용하시면
# 더 편하실 수 있습니다.
def predict(input_text, masking_ner=[]):
    # load model and args
    args = get_args("./model")
    device = get_device("cuda")
    model = load_model("./model", args, device)
    label_lst = get_labels(args)
    logger.info(args)

    # Convert input file to TensorDataset
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    tokenizer = load_tokenizer(args)
    lines = read_input_str(input_text)
    # 기존 predict.py 파일에서도 convert_input_file_to_tensor_dataset 함수 내에서 pred_config 변수가 활용되는 코드가 없어
    # predict_modify.py에서는 해당 매개변수를 제거하였습니다.
    dataset = convert_input_file_to_tensor_dataset(
        lines, args, tokenizer, pad_token_label_id
    )

    # Predict
    sampler = SequentialSampler(dataset)
    # 기존 아래줄의 코드에서 batch_size는 args로 입력받았으며, default값인 32로 초기화했습니다.
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=32)

    all_slot_label_mask = None
    preds = None

    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None}
            if args.model_type != "distilkobert":
                inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                all_slot_label_mask = batch[3].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                all_slot_label_mask = np.append(
                    all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0
                )

    preds = np.argmax(preds, axis=2)
    slot_label_map = {i: label for i, label in enumerate(label_lst)}
    preds_list = [[] for _ in range(preds.shape[0])]

    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            if all_slot_label_mask[i, j] != pad_token_label_id:
                preds_list[i].append(slot_label_map[preds[i][j]])

    # Write to output file
    # 최종적으로 predict 함수는 output text 파일 형태로 결과를 저장합니다.
    # 코드 내에서 사용하기 때문에 문자열 형태로 반환하도록 일부 수정하였습니다.

    output = ""

    for words, preds in zip(lines, preds_list):
        line = ""
        for word, pred in zip(words, preds):
            if pred == "O":
                line = line + word + " "
            elif pred != "O":
                if not pred in masking_ner:
                    if checkNer(pred, masking_ner):
                        if hasNumber(word):
                            masked_word = re.sub("\d", "#", word)
                        else:
                            masked_word = "#" * len(word)
                        line = line + "[{}:{}] ".format(masked_word, pred)
                    else:
                        line = line + "[{}:{}] ".format(word, pred)
                else:
                    if hasNumber(word):
                        masked_word = re.sub("\d", "#", word)
                    else:
                        masked_word = "#" * len(word)
                    line = line + "[{}:{}] ".format(masked_word, pred)

        output += "{}\n".format(line.strip())

    logger.info("Prediction Done!")

    return output


# if __name__ == "__main__":
#     init_logger()

#     output = predict(
#         """유행기 6승 29패 .
#     강변관광지 '보르도'에 취하다
#     특별한인연, R&D 신문로 증수
#     원정에서는 구단이 만든 상대자 원료를 본다 ."""
#     )

#     masking_ner = ["NUM-B"]

#     output_masked = predict(
#         """유행기 6승 29패 .
#     강변관광지 '보르도'에 취하다
#     특별한인연, R&D 신문로 증수
#     원정에서는 구단이 만든 상대자 원료를 본다 .""",
#         masking_ner,
#     )

#     print(output)

#     print(output_masked)
