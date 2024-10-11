import torch
from transformers import BertModel

gres_path = "..\output\gres_swin_base.pth"
gres_weights = torch.load(gres_path)['model']
bert = BertModel.from_pretrained("bert-base-uncased")
bert_weights = dict({k: v for k, v in bert.named_parameters()})

for gres_key, v in gres_weights.items():
    # gres_key = f'text_encoder.{bert_key}'
    if gres_key.startswith('text_encoder.'):
        bert_key = gres_key[len('text_encoder.'):]
        if bert_key not in bert_weights:
            print(F'NOT IN BERT: {bert_key}')
            continue
        bw = bert_weights[bert_key]
        gw = gres_weights[gres_key]
        assert bw.shape == gw.shape
        if not torch.all(bw == gw):
            print(f'WEIGHTS Differ | {gres_key} | {gw.shape}')
        else:
            print(f'SAME VALUES -----------------: {gres_key} | {gw.shape}')

# print(type(file))
# for k, v in file['model'].items():
#     print(k)
