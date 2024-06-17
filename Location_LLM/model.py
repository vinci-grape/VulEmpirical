import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, model, config, args):
        super(Model, self).__init__()
        self.model = model
        self.args = args
        self.classifier = nn.Linear(config.hidden_size, 2, dtype=torch.bfloat16).to(args.device)
    
    def forward(self, input_ids, attention_mask, nl_indices, labels=None):
        hidden_states = self.model(input_ids=input_ids, attention_mask=attention_mask)["hidden_states"]
        final_attention_states = hidden_states[-1]
        nl_final_states = final_attention_states[torch.arange(final_attention_states.size(0)), nl_indices]

        logits = self.classifier(nl_final_states)

        probs = nn.functional.softmax(logits, dim=-1)
        
        tensor = torch.zeros(max(len(nl_final_states), labels[0][-1]+1), dtype=torch.int64).to(self.model.device)
        tensor[labels] = True
        
        labels = tensor[:len(nl_final_states)]
        loss = F.cross_entropy(logits, labels)
        return loss, probs, labels