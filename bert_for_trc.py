import os

import numpy as np

import torch
from torch.utils.data import TensorDataset

import pytorch_pretrained_bert
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel

class BertForTRC(BertPreTrainedModel):
    
    def __init__(self, config):
        super(BertForTRC, self).__init__(config)
        self.bert = BertModel(config)
        self.apply(self.init_bert_weights)
        self.classifier = torch.nn.Linear(2*768, 14)

        self.loss = torch.nn.CrossEntropyLoss()

    def set_loss_weights(self, weights_list):
        """
        Sets custom loss weights.
        ---
        Ideally should be used to increase the weight of losses on
        rare labels and decrease weight of losses on frequent labels.
        """
        self.train_class_loss_weights=np.array(weights_list)
        self.loss = torch.nn.CrossEntropyLoss(
            weight=torch.from_numpy(
              self.train_class_loss_weights)
                  .float()
        )

    def get_positions(self, sequence_output, positions):
        output_tensors = []
        for sample_idx, pos in enumerate(positions):
            position_tensor = sequence_output[sample_idx][pos]
            output_tensors.append(position_tensor)
        return torch.stack(output_tensors, dim=0)

    # def gen_mask(positions):
    #     mask = torch.zeros(len(positions), len(positions), dtype=torch.uint8)
    #     for sample_idx, pos in enumerate(positions):
    #         mask[sample_idx, pos] = 1
    #     return mask

    def eval_sequence_output(self, input_ids, token_type_ids=None, attention_mask=None, 
                             tre_labels=None, e1_pos=None, e2_pos=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                       output_all_encoded_layers=False)
        return sequence_output


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, tre_labels=None, e1_pos=None, e2_pos=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                       output_all_encoded_layers=False)

        e1_hidden = self.get_positions(sequence_output, e1_pos)
        e2_hidden = self.get_positions(sequence_output, e2_pos)
        
        
        cls_tensor = torch.cat((e1_hidden,e2_hidden),1)

        out = self.classifier(cls_tensor)
        loss = self.loss(out, tre_labels)
        return out, loss

class BertForTBD(BertForTRC):
    
    def __init__(self, config):
        super(BertForTBD, self).__init__(config)
        self.classifier = torch.nn.Linear(2*768, 6)

    def set_loss_weights(self, weights_list):
        """
        Sets custom loss weights.
        ---
        Ideally should be used to increase the weight of losses on
        rare labels and decrease weight of losses on frequent labels.
        """
        self.train_class_loss_weights=np.array(weights_list)
        self.loss = torch.nn.CrossEntropyLoss(
            weight=torch.from_numpy(
              self.train_class_loss_weights)
                  .float()
        )

class BertForMatres(BertForTRC):
    
    def __init__(self, config):
        super(BertForTBD, self).__init__(config)
        self.classifier = torch.nn.Linear(2*768, 4)

    def set_loss_weights(self, weights_list):
        """
        Sets custom loss weights.
        ---
        Ideally should be used to increase the weight of losses on
        rare labels and decrease weight of losses on frequent labels.
        """
        self.train_class_loss_weights=np.array(weights_list)
        self.loss = torch.nn.CrossEntropyLoss(
            weight=torch.from_numpy(
              self.train_class_loss_weights)
                  .float()
        )


class InputFeatures(object):
    """Object containing the features for one example/data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 tokens,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label,
                 e1_position=None,
                 e2_position=None
                 ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        
        # Indicates sentence A or sentence B.
        self.segment_ids = segment_ids
        self.label = label
        self.e1_position = e1_position
        self.e2_position = e2_position

        
def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, is_training):
    """Loads a data file into a list of InputFeatures."""

    unique_id = 1000000000

    features = []
    
                
    # Generates features from examples.
    for (example_index, example) in enumerate(examples):
        input_tokens = tokenizer.tokenize(example.text)
        
        # Maximum number of tokens that an example may have. This is equal to 
        # the maximum token length less 3 tokens for [CLS], [SEP], [SEP].
        max_tokens_for_doc = max_seq_length - 3

        # Skips this example if it is too long.
        if len(input_tokens) > max_tokens_for_doc:
            unique_id += 1
            continue
 
        # Creates mappings from words in original text to tokens.
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, word) in enumerate(example.text.split()):
            orig_to_tok_index.append(len(all_doc_tokens)) 
            tokens = tokenizer.tokenize(word)
            for token in tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(token)

        # + 1 accounts for CLS token
        tok_e1_pos = orig_to_tok_index[example.e1_pos] + 1
        tok_e2_pos = orig_to_tok_index[example.e2_pos] + 1
       

        # The -3 accounts for [CLS], [SEP] and [SEP]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in input_tokens:
            tokens.append(token)
            segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        # Zero-pads up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert tok_e1_pos < max_seq_length
        assert tok_e2_pos < max_seq_length
 
        features.append(
            InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label=example.int_label,
                e1_position=tok_e1_pos,
                e2_position=tok_e2_pos
            )
        )
        unique_id += 1

    return features

def make_tensor_dataset(train_features):
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

    all_label_ids = torch.tensor([e.label for e in train_features], dtype=torch.long)
    all_e1_pos = torch.tensor([e.e1_position for e in train_features], dtype=torch.long)
    all_e2_pos = torch.tensor([e.e2_position for e in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_e1_pos, all_e2_pos)
    
    return train_data


def to_json_file(config, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(config.to_json_string())


def save_vocabulary(tokenizer, vocab_path):
    """Save the tokenizer vocabulary to a directory or file."""
    index = 0
    if os.path.isdir(vocab_path):
        vocab_file = os.path.join(vocab_path, 'vocab.txt')
    with open(vocab_file, "w", encoding="utf-8") as writer:
        for token, token_index in sorted(tokenizer.vocab.items(), key=lambda kv: kv[1]):
            if index != token_index:
                logger.warning("Saving vocabulary to {}: vocabulary indices are not consecutive."
                               " Please check that the vocabulary is not corrupted!".format(vocab_file))
                index = token_index
            writer.write(token + u'\n')
            index += 1
    return vocab_file
