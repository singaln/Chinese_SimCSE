# coding=utf-8
# @Time:2021/6/2313:25
# @author: SinGaln
import copy
import json
import torch
import logging
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

def load_data(args, mode):
    data_lst = []
    for line in open(args.data_path +"/"+ mode + ".txt", "r", encoding="utf-8").readlines():
        text_id, text1, text2, label = line.strip().split("||")
        data_lst.append((text1, text2, label))
    return data_lst

class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def concat_seq_pair(token_a, token_b, max_seq_len):
    while True:
        total_length = len(token_a) + len(token_b)
        if total_length < max_seq_len:
            break
        if len(token_a) > len(token_b):
            token_a.pop()
        else:
            token_b.pop()

def covert_to_feature(data, tokenizer, max_length,
                      cls_token_segment_id=0,
                      sequence_a_segment_id=0,
                      sequence_b_segment_id=1,
                      sep_token_segment_id=1):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for i, texts in enumerate(data):
        text_a = texts[0]
        text_b = texts[1]
        labels = texts[2]

        token_a = []
        token_b = []
        for word in text_a:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]
            token_a.extend(word_tokens)

        for word in text_b:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]
            token_b.extend(word_tokens)

        concat_seq_pair(token_a, token_b, max_length - 3)

        tokens = []
        token_type_ids = []
        tokens.append(cls_token)
        token_type_ids.append(cls_token_segment_id)
        for token in token_a:
            tokens.append(token)
            token_type_ids.append(sequence_a_segment_id)
        tokens.append(sep_token)
        token_type_ids.append(cls_token_segment_id)

        for token in token_b:
            tokens.append(token)
            token_type_ids.append(sequence_b_segment_id)
        tokens.append(sep_token)
        token_type_ids.append(sep_token_segment_id)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        # padding
        while len(input_ids) < max_length:
            input_ids.append(pad_token_id)
            attention_mask.append(pad_token_id)
            token_type_ids.append(pad_token_id)

        # assert
        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with attention mask length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with token type length {} vs {}".format(len(token_type_ids), max_length)

        label_id = int(labels)
        if i < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % i)
            logger.info("tokens: % s" % " ".join([str(x) for x in tokens]))
            logger.info("inputs_ids: % s" % " ".join([str(x) for x in input_ids]))
            logger.info("token_type_ids: % s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("attention_mask: % s" % " ".join([str(x) for x in attention_mask]))
            logger.info("labels: % s (id = %d)" % (label_id, label_id))

        features.append(InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                      labels=label_id))
    return features

def load_and_cache_examples(args, tokenizer, mode):
    examples = load_data(args,mode)
    features = covert_to_feature(examples, tokenizer, args.max_length)

    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    label_ids = torch.tensor([f.labels for f in features], dtype=torch.long)

    dataset = TensorDataset(input_ids, attention_mask, token_type_ids, label_ids)
    return dataset