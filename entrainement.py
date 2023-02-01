#Imports

import transformers

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          PreTrainedModel, DistilBertModel, DistilBertForSequenceClassification,
                          TrainingArguments, Trainer)
from transformers.modeling_outputs import SequenceClassifierOutput
import matplotlib.pyplot as plt



df["labels"] = df[labels].values.tolist()


#Create train / test datasets

mask = np.random.rand(len(df)) < 0.8
df_train = df[mask]
df_test = df[~mask]

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


train_encodings = tokenizer(df_train["text"].values.tolist(), truncation=True)
test_encodings = tokenizer(df_test["text"].values.tolist(), truncation=True)


train_labels = df_train["labels"].values.tolist()
test_labels = df_test["labels"].values.tolist()

class GoEmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = GoEmotionDataset(train_encodings, train_labels)
test_dataset = GoEmotionDataset(test_encodings, test_labels)

class DistilBertForMultilabelSequenceClassification(DistilBertForSequenceClassification):
    def __init__(self, config):
      super().__init__(config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.distilbert(input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        hidden_state = outputs[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.float().view(-1, self.num_labels))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions)


num_labels=len(labels)

model = DistilBertForMultilabelSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to('cpu')

#model = DistilBertForMultilabelSequenceClassification.from_pretrained(r"C:\Users\malos\OneDrive\Documents\2A\PSC\Modèles\Assistant virtuel\Classification\classification\checkpoint-51", num_labels=num_labels).to('cpu')

model.config.id2label = id2label
model.config.label2id = label2id

#Create trainer

def accuracy_thresh(y_pred, y_true, thresh=0.5, sigmoid=True):
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)
    if sigmoid:
      y_pred = y_pred.sigmoid()
    return ((y_pred>thresh)==y_true.bool()).float().mean().item()


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {'accuracy_thresh': accuracy_thresh(predictions, labels)}


batch_size = 32
# configure logging so we see training loss
logging_steps = len(train_dataset) // batch_size

args = TrainingArguments(
    output_dir="classification",
    evaluation_strategy = "epoch",
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=logging_steps
)


trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer)

def classify(s):
    y_pos = np.arange(len(labels))

    width = 0.5

    fig, ax = plt.subplots()

    maxl = 0

    model = DistilBertForMultilabelSequenceClassification.from_pretrained(r"C:\Users\malos\OneDrive\Documents\2A\PSC\Modèles\Assistant_virtuel\Classification\classification\checkpoint-246", num_labels=num_labels).to('cpu')

    input = torch.tensor([tokenizer(s)["input_ids"]])
    logits = model(input)[:2]

    output = torch.nn.Softmax(dim=1)(logits[0])

    output = output[0].tolist()

    result = labels[np.argmax(output)]

    print(result)

    hbars = ax.barh(y_pos , output,width, align='center')

    maxl = max(maxl,max(output))


    ax.set_yticks(y_pos, labels=labels)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.legend()

    # Label with specially formatted floats
    # ax.bar_label(hbars, fmt='%.2f')
    ax.set_xlim(right=min(1,maxl+0.1))  # adjust xlim to fit labels

    plt.show()