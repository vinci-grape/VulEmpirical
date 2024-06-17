import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, encoder, config, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.args = args
        self.classifier = nn.Linear(config.hidden_size, 2)
    

    def get_code_vec(self, source_ids, nl_indices):
        if self.config.model_type == "plbart" or self.config.model_type == "t5": 
            attention_mask = source_ids.ne(self.config.pad_token_id)
            outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask, labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs['decoder_hidden_states'][-1]
            nl_final_states = hidden_states[torch.arange(hidden_states.size(0)), nl_indices[1]]
        else:
            attention_mask = source_ids.ne(self.config.pad_token_id)
            outputs = self.encoder(source_ids, attention_mask=attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2), output_hidden_states=True)
            hidden_states = outputs[0]
            nl_final_states = hidden_states[torch.arange(hidden_states.size(0)), nl_indices[1]]
        return nl_final_states
    
    
    def forward(self, input_ids, labels=None):      
        nl_indices = torch.where(input_ids == self.args.nl_ids)
        nl_final_states = self.get_code_vec(input_ids, nl_indices)
        logits = self.classifier(nl_final_states)
        probs = nn.functional.softmax(logits, dim=-1)
        
        if labels.shape != torch.Size([1, 0]):
            tensor = torch.zeros(max(len(nl_final_states), labels[0][-1]+1), dtype=torch.int64).to(self.encoder.device)
            tensor[labels] = True
        else:
            tensor = torch.zeros(max(len(nl_final_states), 0), dtype=torch.int64).to(self.encoder.device)     
        labels = tensor[:len(nl_final_states)]

        loss = F.cross_entropy(logits, labels)

        return loss, probs, labels
    