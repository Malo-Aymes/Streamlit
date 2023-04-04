#Imports

import transformers

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import requests
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          PreTrainedModel, DistilBertModel, DistilBertForSequenceClassification,
                          TrainingArguments, Trainer)
from transformers.modeling_outputs import SequenceClassifierOutput
import matplotlib.pyplot as plt

import huggingface_hub
import openpyxl


#Imports

import transformers
#import datasets

import torch
import pandas as pd
import numpy as np
from transformers import (AutoTokenizer, DistilBertForSequenceClassification)
from transformers.modeling_outputs import SequenceClassifierOutput
import matplotlib.pyplot as plt
import requests
import io

import streamlit as st
from google.cloud import firestore

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

#Setup

st.title("Classification : Virtual Assistant")

def classify(s,model):
        input = torch.tensor([tokenizer(user_input)["input_ids"]])
        logits = model(input)[:2]
        output = torch.nn.Softmax(dim=1)(logits[0])
        output = output[0].tolist()

        return output

@st.cache(allow_output_mutation = True)
def get_model():
    huggingface_hub.login(token = "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")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForMultilabelSequenceClassification.from_pretrained("Deopusi/classification_test",num_labels=14)
    return tokenizer,model


tokenizer,model = get_model()


labels = ['Weather', 'Clock', 'Calendar', 'Map', 'Phone', 'Email', 'Calculator', 'Translator', 'Web search', 'Social media', 'Small talk', 'Message', 'Reminders', 'Music']

id2label = {str(i):label for i, label in enumerate(labels)}
label2id = {label:str(i) for i, label in enumerate(labels)}

model.config.id2label = id2label
model.config.label2id = label2id


#Firestore

db = firestore.Client.from_service_account_json("firestore-key.json")

#Input

@st.cache_resource
def button_states():
    return {"pressed": None}

is_pressed = button_states()  # gets our cached dictionary

user_input = st.text_area("Enter sentence to classify :")
#values = st.checkbox("Show values")
button = st.button("Classify")

if button:
    # any changes need to be performed in place
    is_pressed.update({"pressed": True})

if is_pressed["pressed"]and user_input:
    # input = torch.tensor([tokenizer(user_input)["input_ids"]])
    # logits = model(input)[:2]
    # output = torch.nn.Softmax(dim=1)(logits[0])
    # output = output[0].tolist()
    output = classify(user_input,model)
    result = labels[np.argmax(output)]
    st.markdown(f"<h2 style='text-align: center; color: black;'>~~~~{result}~~~~</h2>", unsafe_allow_html=True)

    if False:
        y_pos = np.arange(len(labels))
        width = 0.5
        fig, ax = plt.subplots()
        maxl = 0
        hbars = ax.barh(y_pos , output,width, align='center')
        maxl = max(maxl,max(output))
        ax.set_yticks(y_pos, labels=labels)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.legend()
        # Label with specially formatted floats
        # ax.bar_label(hbars, fmt='%.2f')
        ax.set_xlim(right=min(1,maxl+0.1))  # adjust xlim to fit labels
        st.pyplot(fig)


    
    sat = st.radio('Is this correct ?' , ('Yes','No'))

    if sat=='No':
        option = st.selectbox('Class :', ('Weather', 'Clock', 'Calendar', 'Map', 'Phone', 'Email', 'Calculator', 'Translator', 'Web search', 'Social media', 'Small talk', 'Message', 'Reminders', 'Music'))
    else:
        option = result 
        # option == correct class
    send = st.button("Send")
    
    if send and sat!=None :
        doc_ref = db.collection("classification").document(user_input)
        doc_ref.set({"text":user_input,"class":option})
        st.write('Thank you for your input !')
    if send and sat == 'No':
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


        def accuracy_thresh(y_pred, y_true, thresh=0.5, sigmoid=True):
            y_pred = torch.from_numpy(y_pred)
            y_true = torch.from_numpy(y_true)
            if sigmoid:
                y_pred = y_pred.sigmoid()
            return ((y_pred>thresh)==y_true.bool()).float().mean().item()


        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            return {'accuracy_thresh': accuracy_thresh(predictions, labels)}



        df = pd.DataFrame({"text":[user_input],'Weather':[0], 'Clock':[0], 'Calendar':[0], 'Map':[0], 'Phone':[0], 'Email':[0], 'Calculator':[0], 'Translator':[0], 'Web search':[0], 'Social media':[0], 'Small talk':[0], 'Message':[0], 'Reminders':[0], 'Music':[0]})

        df[option].values[0] = 1

        df["labels"] = df[labels].values.tolist()

        df = pd.concat([df]*10,axis = 0,ignore_index=True)
        
    ##  
    
        df_test = pd.read_csv("https://raw.githubusercontent.com/Malo-Aymes/Streamlit/main/BdD1.csv",sep=";",encoding= 'unicode_escape',error_bad_lines=False)
      
       # st.write(df_test.columns[:12].tolist())
    ##

        df_test["labels"] = df_test[labels].values.tolist()

        df = pd.concat([df,df_test.sample(10)],axis = 0,ignore_index=True)

        df_test = df_test.sample(100)

        # print(df_test.columns[:14].tolist())


        train_encodings = tokenizer(df["text"].values.tolist(), truncation=True)
        test_encodings = tokenizer(df_test["text"].values.tolist(), truncation=True)


        train_labels = df["labels"].values.tolist()
        test_labels = df_test["labels"].values.tolist()


        train_dataset = GoEmotionDataset(train_encodings, train_labels)
        test_dataset = GoEmotionDataset(test_encodings, test_labels)

        args = TrainingArguments(
            output_dir="classification_test",
            evaluation_strategy = "epoch",
            learning_rate=2e-5,
            num_train_epochs=3,
            weight_decay=0.01
        )


        trainer = Trainer(
            model,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer)

        trainer.train()
        trainer.save_model()


        ## verify if the model is destroyed on the original dataset
        total_number_test = 100 ##number of samples to test after update
        df_test = df_test.sample(total_number_test)
                
        test_labels = df_test["labels"].values.tolist()        
        test_encodings = tokenizer(df_test["text"].values.tolist(), truncation=True)

        test_dataset = GoEmotionDataset(test_encodings, test_labels)

        count =0
        for i in range(total_number_test):
            ouput = classify(df_test["text"].values.tolist()[i],model)
            result = labels[np.argmax(output)]
            if test_labels[i] == result:
                count = count + 1
            st.write(df_test["text"].values.tolist()[i] , "->" , result)
            st.write(test_labels[i])
        st.write("The rate of correction after updating on the original dataset is:", count/total_number_test)
