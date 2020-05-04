import config
import transformers
import torch.nn as nn


class BertBaseModel(nn.Module):
    def __init__(self):
        super(BertBaseModel, self).__init__()
        self.bert = transformers.DistilBertModel.from_pretrained(
                        config.BERT_MODEL)

        self.bert_dropout = nn.Dropout(0.2)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask=None, token_type_ids=None):
        _, o2 = self.bert(
            ids
            # attention_mask=mask,
            # token_type_ids=token_type_ids
        )

        bo = self.bert_dropout(o2)
        output = self.out(bo)
        return output