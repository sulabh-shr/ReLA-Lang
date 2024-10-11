import torch
from transformers import BertModel

gres_path = "..\output\gres_swin_base.pth"
gres_weights = torch.load(gres_path)['model']
bert = BertModel.from_pretrained("bert-base-uncased")
bert_weights = dict({k: v for k, v in bert.named_parameters()})

for gres_key, v in gres_weights.items():
    print(f'{gres_key:-<100s} : {v.shape}')
