#Imports

import transformers
#import datasets

import torch
import pandas as pd
import numpy as np
from transformers import (AutoTokenizer, DistilBertForSequenceClassification,AutoModelForQuestionAnswering)
from transformers.modeling_outputs import SequenceClassifierOutput
import matplotlib.pyplot as plt
import requests
import io

import streamlit as st

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

st.title("Virtual Assistant")

@st.cache(allow_output_mutation = True)
def get_models():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model_classification = DistilBertForMultilabelSequenceClassification.from_pretrained("Deopusi/virtual_assistant_classification",use_auth_token = "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",num_labels=14)
    model_extraction = AutoModelForQuestionAnswering.from_pretrained("Deopusi/extraction",use_auth_token = "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")
    return tokenizer,model_classification,model_extraction


tokenizer,model_classification,model_extraction = get_models()


labels = ['Weather', 'Clock', 'Calendar', 'Map', 'Phone', 'Email', 'Calculator', 'Translator', 'Web search', 'Social media', 'Small talk', 'Message', 'Reminders', 'Music']

id2label = {str(i):label for i, label in enumerate(labels)}
label2id = {label:str(i) for i, label in enumerate(labels)}

model_classification.config.id2label = id2label
model_classification.config.label2id = label2id

alt_questions = {'Where - Weather':'What is the location ?',
    'When - Weather':'What is the moment or time of interest ?',
    'What - Weather':'What do we want to know ?',
    'Type - Clock':'What function of a clock is to be used ?',
    'Time - Clock':'What is the time or duration ?',
    'What - Clock':'What do we want to do or know ?',
    'Where - Clock':'What is the place we are interested in ?',
    'When - Calendar':'When are we doing something ?',
    'What - Calendar':'What is our action in our calendar?',
    'Event/Person - Calendar':'What event or person are we interested in ?',
    'Start - Map':'What is our starting point ?',
    'End - Map':'What is our destination ?',
    'What - Map':'What are we accessing ?',
    'Type - Phone':'What function of a phone is to be used ?',
    'Information - Phone':'What information are we accessing ?',
    'Who - Phone':'Who is mentioned ?',
    'Type - Email':'What function of email is to used ?',
    'Information - Email':'What interests us ?',
    'Who - Email':'Who is involved ?',
    'Type - Calculator':'What function of a calculator is to be used ?',
    'Expression - Calculator':'What is the litteral expression ?',
    'Numbers - Calculator':'What is the numerical expression ?',
    'What - Translator':'What are we translating ?',
    'Start language - Translator':'What is our starting language ?',
    'End language - Translator':'What is our target language ?',
    'What - Web search':'What are we searching for ?',
    'Platform - Social media':'What social media platform are we accessing ?',
    'What - Social media':'What interests us ?',
    'Action - Social media':'What are we to do on social media ?',
    'type of sentence  - Small talk':'What type of sentence is this ?',
    'what - Small talk':'What are we talking about ?',
    'Action - Message':'What are we doing in our messages?',
    'Who - Message':'Who are we involving ?',
    'What - Message':'What are we sending or acting on ?',
    'Action - Reminders':'What are we doing in our reminders ?',
    'Time - Reminders':'When is the time mentioned ?',
    'What - Reminders':'What are we talking about ?',
    'Type of music - Music':'What is the type of music ?',
    'Author - Music':'Who is the author ?',
    'Title - Music':'What is the title ?',
    'Action - Music':'What are we to do ?'}

add_info = ['Default : here, now, weather',
    'Options : Clock, Alarm, Timer, World Clock, Stopwatch',
    'Operations: show, add, cancel, create event, delete, accept ; Default time : now',
    'Default location : here ; Operations: here, directions, locate, transport, distance',
    'Options : call, mute, charge, text, call history, contacts, Facetime, voicemail, do not disturb, end ',
    'Options : write, reply, forward, inbox, contacts, outbox, mark unread, mark read, unsubscribe, spam',
    'Options : Calculate, Set, Convert, Integrate, Solve, Geometry, Derivative',
    'Common languages : English, Mandarin Chinese, Spanish, Hindi, French, Arabic, Bengali, Russian, Portuguese, Urdu, Italian, German',
    ' ',
    'Platforms : Facebook, Instagram, Twitter, Pinterest, TikTok, LinkedIn, Snapchat, Whatsapp, Youtube, WeChat, Telegram',
    'Options : wh-question, command, exclamation, statement, yes/no question',
    'Operations : send, delete, block, read, search, forward, reply, list, mark as read, set up',
    'Operations : create, list, write, mark as complete, delete',
    ' ']
classes = ['Weather', 'Clock', 'Calendar', 'Map', 'Phone', 'Email', 'Calculator', 'Translator', 'Web search', 'Social media', 'Small talk', 'Message', 'Reminders', 'Music']


#Input

user_input = st.text_area("Enter sentence :")
button = st.button("Process")



if user_input and button:
    input = torch.tensor([tokenizer(user_input)["input_ids"]])
    logits = model_classification(input)[:2]
    output = torch.nn.Softmax(dim=1)(logits[0])
    output = output[0].tolist()
    result = labels[np.argmax(output)]
    st.write(result)

    text = f"{user_input} ; Class : {result} ; {add_info[classes.index(result)]} ;  "
    questions = [alt_questions[x] for x in alt_questions.keys() if result in x]
    for q in questions:
        inputs = tokenizer(q, text, return_tensors="pt")
        outputs = model_extraction(**inputs)

        start = np.argmax(outputs.start_logits[0].tolist())
        end = np.argmax(outputs.end_logits[0].tolist())

        tokens = inputs.input_ids[0, start : end + 1]

        answer = tokenizer.decode(tokens)

        if '[CLS]' in answer:
            answer = ""
        st.write(q+"   "+answer)