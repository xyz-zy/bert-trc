import torch
import numpy as np

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

    def gen_mask(positions):
        mask = torch.zeros(len(positions), len(positions), dtype=torch.uint8)
        for sample_idx, pos in enumerate(positions):
            mask[sample_idx, pos] = 1
        return mask

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, tre_labels=None, e1_pos=None, e2_pos=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                       output_all_encoded_layers=False)

        e1_hidden = self.get_positions(sequence_output, e1_pos)
        e2_hidden = self.get_positions(sequence_output, e2_pos)
        
        
        cls_tensor = torch.cat((e1_hidden,e2_hidden),1)

        out = self.classifier(cls_tensor)
        loss = self.loss(out, tre_labels)
        return out, loss
