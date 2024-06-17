import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, encoder, config, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.args = args
        self.classifier = nn.Linear(config.hidden_size, 3)


    def get_code_vec(self, source_ids):
        if self.config.model_type == "plbart" or self.config.model_type == "t5": 
            attention_mask = source_ids.ne(self.config.pad_token_id)
            outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask, labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs['decoder_hidden_states'][-1]
            eos_mask = source_ids.eq(self.config.eos_token_id)

            vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                                hidden_states.size(-1))[:, -1, :]
        else:
            mask = source_ids.ne(self.config.pad_token_id)
            out = self.encoder(source_ids, attention_mask=mask.unsqueeze(1) * mask.unsqueeze(2), output_hidden_states=True)
            vec = (out[0] * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)  # averege
        return vec
    

    def forward(self, input_ids, labels=None):
        vec = self.get_code_vec(input_ids)        
        logits = self.classifier(vec)
        probs = nn.functional.softmax(logits, dim=-1)
        loss = F.cross_entropy(logits, labels)

        return loss, probs, labels
    