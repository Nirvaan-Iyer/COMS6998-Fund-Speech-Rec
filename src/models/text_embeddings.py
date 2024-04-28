from transformers import BertTokenizer, BertModel
import torch
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

iemocap_data = [
    "I am feeling so happy right now.",
    "That movie made me really sad.",
    "She looked surprised when she saw me.",
    "His anger was palpable in his voice."
]

def get_bert_embeddings(texts, max_length=128):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
                            text,
                            add_special_tokens=True,
                            max_length=max_length,
                            pad_to_max_length=True,
                            return_attention_mask=True,
                            return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)
        last_hidden_states = outputs.last_hidden_state

    return last_hidden_states

embeddings = get_bert_embeddings(iemocap_data)

embeddings_np = embeddings.numpy()

print("Shape of BERT embeddings for IEMOCAP data:", embeddings_np.shape)
