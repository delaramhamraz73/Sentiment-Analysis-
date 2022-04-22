from transformers import BertModel, BertTokenizer
import torch
from torch import nn



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_names = ['negative', 'neutral', 'positive']

PRE_TRAINED_MODEL_NAME='bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

class SentimentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)

MAX_LEN = 180

model = SentimentClassifier(len(class_names))
model.load_state_dict(torch.load('best_model_state.bin', map_location=torch.device('cpu')))
model = model.to(device)

# phrase = "I love my company"
article = open(r'Toys RU.txt')
article = article.read()

encoded_article = tokenizer.encode_plus(
  text=article,
  max_length=MAX_LEN,
  truncation=True,
  add_special_tokens=True,
  return_token_type_ids=False,
  pad_to_max_length=True,
  return_attention_mask=True,
  return_tensors='pt',)
input_ids = encoded_article['input_ids'].to(device)
attention_mask = encoded_article['attention_mask'].to(device)
output = model(input_ids, attention_mask)
_, prediction = torch.max(output, dim=1)
print(f'article_text: {article}')
print(f'Sentiment  : {class_names[prediction]}')



