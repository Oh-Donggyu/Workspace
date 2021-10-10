# %% [markdown]

# ## Quick Tour
# [notebooks](https://colab.research.google.com/github/huggingface/notebooks/blob/master/transformers_doc/quicktour.ipynb#scrollTo=VVX2-GnQGKjt)
# - Let's have a quick look at the 🤗 Transformers library features. 
# - The library downloads pretrained models for, 
# - Natural Language Understanding (NLU) tasks 
#   - such as analyzing the sentiment of a text
# - Natural Language Generation (NLG),
#   - such as completing a prompt with new text or translating in another language.  
# 
# First we will see how to easily leverage the pipeline API to quickly use those pretrained models at inference. 
# Then, we will dig a little bit more and see how the library gives you access to those models 
# and helps you preprocess your data.
# %%
""" The easiest way to use pretrained model on a given task is to use pipeline """

from transformers import pipeline
classifier = pipeline('sentiment-analysis')

sentence = "We are very happy to show you the 🤗 Transformers library."
result = classifier(sentence)
print(result)
# [{'label': 'POSITIVE', 'score': 0.9997795224189758}]

sentences = [
    "We are very happy to show you the 🤗 Transformers library.",
    "We hope you don't hate it."
]
results = classifier(sentences)
for result in results:
    print(f"label: {result['label']}, with score: {result['score']:.4f}")
# label: POSITIVE, with score: 0.9998
# label: NEGATIVE, with score: 0.5309



""" Using different model in hugging face hub """

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# classifier = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
model_name = "nlptown/bert_base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)



""" Using the Tokenizer """
# you can learn more about tokenizers below
# https://huggingface.co/transformers/preprocessing.html

sentence = "We are very happy to show you the 🤗 Transformers library."
inputs = tokenizer(sentence)
print(inputs)

sentences = [
    "We are very happy to show you the 🤗 Transformers library.",
    "We hope you don't hate it."
]
batch = tokenizer(
    sentences,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors='pt'
)

for key, value in batch.items():
    print(f"{key}: {value.numpy().tolist()}")



""" Using the Mdoel """

# in hugging face transformers all outputs are objects that 
# contain the model's final activations along with other metadata
# These objects are described in greater detail below
# https://huggingface.co/transformers/main_classes/output.html
outputs = model(**batch)

print(outputs)
# SequenceClassifierOutput(loss=None, logits=tensor([[-4.0833,  4.3364],
#     [ 0.0818, -0.0418]], grad_fn=<AddmmBackward>), hidden_states=None, attentions=None)



""" Apply softmax activation to get predictions """

import torch
import torch.nn as nn

predictions = nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

# If you provide the model with labels in addition to inputs, the model output object will also contain a loss attribute
outputs = model(**batch, labels=[torch.tensor([1, 0])])
print(outputs)
# SequenceClassifierOutput(loss=tensor(0.3167, grad_fn=<NllLossBackward>), logits=tensor([[-4.0833,  4.3364],
#     [ 0.0818, -0.0418]], grad_fn=<AddmmBackward>), hidden_states=None, attentions=None)


""" Trainer API """
# Models are standard torch.nn.Module, so you can use them in your usual training loop.
# Transformers are also provides a Trainer class to help with your training
# (such as distributed training, mixed precision, etc ..) 
# See the tutorial for more details
# https://huggingface.co/transformers/training.html  =>  읽어보면 좋음.



""" Save and Load model with its Tokenizer """

save_dir = "./model"
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

# transformers의 cool features 중 하나는 torch, tf 각각에서 저장한 모델을 
# 서로 바꿔서 로드할 수 있다는 점이다.
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(save_dir)
model = AutoModel.from_pretained(save_dir)


""" return all hidden states and all attention weights if you need them """

outputs = model(**batch, output_hidden_states=True, output_attentions=True)
all_hidden_states = outputs.hidden_states
all_attentions = outputs.attentions



""" Accessing the code """
# The AutoModel and AutoTokenizer classes are just shortcuts 
# that will automatically work with any pretrained model. 
# Behind the scenes, the library has one model class per combination of architecture plus class, 
# so the code is easy to access and tweak if you need to.

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = DistilBertForSequenceClassification.from_pretrained(model_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)



""" Customizing the Model """
# If you want to change how the model itself is built, 
# you can define a custom configuration class. 
# Each architecture comes with its own relevant configuration. 
# For example, DistilBertConfig allows you to specify parameters 
# such as the hidden dimension, dropout rate, etc for DistilBERT. 
# If you do core modifications, like changing the hidden size, 
# you won't be able to use a pretrained model anymore and will need to train from scratch. 
# You would then instantiate the model directly from this configuration.

from transformers import DistilBertConfig, DistilBertTokenizerFast, DistilBertForSequenceClassification

config = DistilBertConfig(n_heads=8, dim=512, hidden_dim=4*512)
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification(config)

