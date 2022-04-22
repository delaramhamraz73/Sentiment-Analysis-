from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from collections import defaultdict
from textwrap import wrap
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import  classification_report

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("Sentences_50Agree.csv")
print(df.head())
class_names = ['negative', 'neutral', 'positive']
# =====================================================================================================================

ax = sns.countplot(df.Sentiment)
plt.xlabel("Texts Score");
ax.set_xticklabels(class_names)
#plt.show()

# ====================Data Pre-processing==============================================================================
PRE_TRAINED_MODEL_NAME='bert-base-cased'

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

""" 
using a sample text to understand the tokenization process
"""
sample_txt = 'I work at Finwedge.'
tokens = tokenizer.tokenize(sample_txt)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

""" Some basic operations can convert the text to tokens and tokens to unique integers (ids),
[SEP] - marker for ending of a sentence
[CLS] - we must add this token to the start of each sentence, so BERT knows we’re doing classification
[PAD] - special token for padding
BERT understands tokens that were in the training set. Everything else can be encoded using the [UNK] (unknown) token
we use encode_plus for doing all of the above steps!
"""
encoding = tokenizer.encode_plus(
  sample_txt,
  max_length=32,
  add_special_tokens=True, # Add '[CLS]' and '[SEP]'
  return_token_type_ids=False,
  pad_to_max_length=True,
  return_attention_mask=True,
  truncation=True,
  return_tensors='pt',  # Return PyTorch tensors
)
"""
The token ids are now stored in a Tensor and padded to a length of 32
"""
print(encoding.keys())
print(len(encoding['input_ids'][0]))
print(encoding['input_ids'][0])

# ====================================================================================================================
"""
BERT works with fixed-length sequences. We have to choose the max length.
we store the token length of each review
"""
token_lens = []
for txt in df.Texts:
  tokens = tokenizer.encode(txt, max_length=180, truncation = True )
  token_lens.append(len(tokens))


"""
plot the distribution of tokens in reviews
"""
sns.distplot(token_lens)
plt.xlim([0, 256])
plt.xlabel('Token count')
plt.show()

# ==================== Length of tokens in a sentence in the database==================================================
MAX_LEN = 180
# ===========================Pytorch Dataset===========================================================================

"""
A pytorch dataset indicates a dataset object to load data from. In this code we are using a map-style dataset. 
A map-style dataset is one that implements the __getitem__() and __len__() protocols, and represents a map from 
indices/keys to data samples.
For example, such a dataset, when accessed with dataset[idx], could read the idx-th phrase and its corresponding label
from a folder on the disk.
"""


class GPReviewDataset(Dataset):
    def __init__(self, phrases, targets, tokenizer, max_len):
        self.phrases = phrases
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, item):
        phrase = str(self.phrases[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
          phrase,
          add_special_tokens=True,
          max_length=self.max_len,
          truncation = True,
          return_token_type_ids=False,
          pad_to_max_length=True,
          return_attention_mask=True,
          return_tensors='pt',
        )
        return {
          'Database_text': phrase,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long)
        }

# =====================# Splitting the data into train and test set====================================================


df_train, df_test = train_test_split(
    df,
    test_size = 0.1,
    random_state=RANDOM_SEED
)
df_val, df_test = train_test_split(
    df_test,
    test_size=0.5,
    random_state=RANDOM_SEED
)
print(df_train.shape)
print(df_val.shape)
print(df_test.shape)

# ==========================Creating Data Loader=======================================================================
"""
At the heart of PyTorch data loading utility is the torch.utils.data.DataLoader class. It represents
a Python iterable over a dataset.
"""


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(
      phrases=df.Texts.to_numpy(),
      targets=df.Sentiment.to_numpy(),
      tokenizer=tokenizer,
      max_len=max_len,
    )
    return DataLoader(
      ds,
      batch_size=batch_size,
      num_workers=0
    )


BATCH_SIZE = 16
train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

# =====================================================================================================================

data = next(iter(train_data_loader))
print(data.keys())
# print(data['input_ids'].shape)
# print(data['attention_mask'].shape)
# print(data['targets'].shape)

bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
last_hidden_state, pooled_output = bert_model(
    input_ids=encoding['input_ids'],
    attention_mask=encoding['attention_mask']
)

print(last_hidden_state.shape)
print(bert_model.config.hidden_size)

# ======================Classifier using BERT model==================================================================


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
# =====================================================================================================================


model = SentimentClassifier(len(class_names))
model = model.to(device)
input_ids = data['input_ids']#.to(device)
attention_mask = data['attention_mask']#.to(device)
print(input_ids.shape)
print(attention_mask.shape)

# =====================================================================================================================
"""
To reproduce the training procedure from the BERT paper, the AdamW optimizer provided by Hugging Face is used. 
It corrects weight decay. We’ll also use a linear scheduler with no warmup steps.
get_linear_schedule_with_warmup : Create a schedule with a learning rate that decreases linearly from the initial lr 
set in the optimizer to 0, after a warmup period during which it increases linearly from 0 to the initial lr set in 
the optimizer.
for hyper parameters, we use BERT fine tuning recommendations. 
- Batch size: 16, 32
- Learning rate (Adam): 5e-5, 3e-5, 2e-5
"""
EPOCHS = 10
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss()

# =========================Helper function to train the model==========================================================
"""
The scheduler gets called every time a batch is fed to the model. We’re avoiding exploding gradients by clipping the 
gradients of the model using clipgrad_norm.
"""


def train_epoch( model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
  model = model.train()
  losses = []
  correct_predictions = 0
  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)
    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)
    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
  return correct_predictions.double() / n_examples, np.mean(losses)

# ======================Helper function that evaluate the model on a given data loader================================


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
      for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)

# =======================Storing the Training History =================================================================


""" 
we will see the result (accuracy and validation) at this step after each epoch)
"""
history = defaultdict(list)
best_accuracy = 0
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    train_acc, train_loss = train_epoch(
      model,
      train_data_loader,
      loss_fn,
      optimizer,
      device,
      scheduler,
      len(df_train)
    )
    print(f'Train loss {train_loss} accuracy {train_acc}')
    val_acc, val_loss = eval_model(
      model,
      val_data_loader,
      loss_fn,
      device,
      len(df_val)
    )
    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc

# =================================training vs validation accuracy plot==============================================
plt.plot(history['train_acc'], label='train accuracy')
plt.plot(history['val_acc'], label='validation accuracy')
plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1])
plt.show()
# =================================a helper function to get the predictions from our model=============================


def get_predictions(model, data_loader):
    model = model.eval()
    Database_texts = []
    predictions = []
    prediction_probs = []
    real_values = []
    with torch.no_grad():
      for d in data_loader:
        texts = d["Database_text"]
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)
        Database_texts.extend(texts)
        predictions.extend(preds)
        prediction_probs.extend(outputs)
        real_values.extend(targets)
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return Database_texts, predictions, prediction_probs, real_values

# =================================================================================================================


y_Database_texts, y_pred, y_pred_probs, y_test = get_predictions(
    model, test_data_loader)

# ========================the classification report================================================================
print(classification_report(y_test, y_pred, target_names=class_names))
# ==================================examining a phrase from the dataset==============================================
idx = 2
Database_text = y_Database_texts[idx]
true_sentiment = y_test[idx]
pred_df = pd.DataFrame({
  'class_names': class_names,
  'values': y_pred_probs[idx]
})

print("\n".join(wrap(Database_text)))
print()
print(f'True Sentiment:{class_names[true_sentiment]}')

