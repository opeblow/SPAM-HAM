import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import TensorDataset,DataLoader,WeightedRandomSampler
from sklearn.model_selection import train_test_split
from utils.preprocessing import build_vocab,tokenize,augment_tokens,encode_tokens
from utils.train_utils import EarlyStopping
from models.model import SpamHamModel
import pandas as pd
import numpy as np
import re
import random
import json
import os

DATA_DIR="data"
SPAMHAM_CSV=r"C:\Users\user\Documents\spamham\spamham.csv"
EMAIL_CSV=r"C:\Users\user\Documents\spamham\email.csv"
VOCAB_DIR="vocab"
os.makedirs(VOCAB_DIR,exist_ok=True)

MAX_LENGTH=50
BATCH_SIZE=32
EMBED_DIM=64
HIDDEN_DIM=128
DROPOUT=0.3
LR=0.0005
WEIGHT_DECAY=1e-5
EPOCHS=40
PATIENCE=2


#Load CSVs
df=pd.read_csv(SPAMHAM_CSV)
extra_df=pd.read_csv(EMAIL_CSV)
#normalize column names
extra_df=extra_df.rename(columns={'Category':'Label','Message':'Text'})
extra_df['Label']=extra_df['Label'].map({'ham':0,'spam':1})
df['Label']=df['Label'].map({'ham':0,'spam':1})
#keep only spam from the extra datasets
extra_spam=extra_df[extra_df['Label']==1]
df=pd.concat([df,extra_spam],ignore_index=True)
df=df.sample(frac=1.0,random_state=42).reset_index(drop=True)#shuffle
new_spam_messages=[
     "you've won a brand new iphone!Visit bit.ly/freephone to claim now",
     "Congratulations you've been selected for a $1,000 walmart gift card.Visit now to redeem.",
     "Earn $500 daily working from home! Visit workfastmoney.com to get started",
     "Your number was randomly chosen!Visit luckywinner.net to see your price!",
     "Click here to remove all your debt instantly.Visit debtfree2025.org for details",
     "Want to lose 10kg in 10 days? Visit slimquickresults.xyz for a free trail ",
     "Urgent:Your account will be suspended.Visit secure-login-alert.net to verify info now",
     "Congratulations!You've been selected for a loan of $50,000 at 0% interest. click here to claim:loannow.xyz",
     "for account related queries visit our website or speak with a support agent",
     "dear customer please visit your nearest atm to update your kyc before 30th june",
     "you can view your transaction history when you visit your oline banking portal",
     "your insurance documents are ready for pickup. kindly visit the branch during working hours",
     "congratulations you've been selected for a loan of $50,000 at 0% interest. click here to claim:loannow.xyz",
     "your bank account has been suspended verify your details immediately at secure-login.bankverify.cc",
     "final warning pay your unpaid loan today and get 50% discount. click to avoid penalty",
     "congrats! you have won #1,000 in glo promo,call now!",
     "hi dear long time! let's connect on whatsapp +234xxxxxxxx",
     "hello friend,i have business for you! big money involved",
     "need quick loan? no bvn needed instant approval",
     "please donate to orphanage,account number 2663995050 access bank",
     "new year giveaway,just reply yes to win",
     "MTN:your sim has been upgraded click to activate",
     "hello,are you interested in forex trading? dm me now!",
     "this is from zenith bank reactivate your token now.",
     "your gtb account is restricted visit this link to reactivate",
     "click now to receive #2000 MTN data bonus",
     "i'm a soldier overseas. looking for honest partner",
     "we offer zero interest loan reply yes now",
     "you have been selected as lucky winner tap here",
     "sign up and get #10,000 daily doing surveys",
     "buy laptop at giveaway price limited offer!",
     "urgent message from cac your business is flagged",
     "i'll pay you 5k just to chat,interesred?",
     "visit this link to win your iphone 16 pro max",
     "visit this site to win 100k",
     "congratulations!you are the winner of a free gift.claim your prize now.",
     "urgent:your account has been selected for a bonus offer.click the link to verify",
     "get a loan with zer0 interest today!limited time promotion,call now",
     "buy cheap forex trading packages and earn huge income",
     "you have been preapproved for a secured credit card.activate your account today.",
     "reply yes to suscribe to our exclusive giveaway",
     "discount on all products!visit our website and save big",
     "investment opportunity with guaranteed returns. don't miss out",
     "your atm card will be deactivated soon. reactivate now!",
     "earn $5,000/month with this secret crypto method no skills required",
     "final notice!your account will be suspended.verify now",
     "act now and get a 90% discount on all electronics!",
     "unlock $1,000 instantly with this exclusive app-download now!",
     "we tried to deliver your package,but need more details.click to update",
     "limited time crypto investment with 200% guaranteed returns",
     "instant approval!apply now for $5,000 credit line",
     "don't miss your last chance to win a luxury vacation. enter here",
     "update required.your system is out of date.tap to install",
     "i'm opeyemi",
     "i'm ai architect"
]
new_spam_df=pd.DataFrame({
     'Text':new_spam_messages,
     'Label':[1]*len(new_spam_messages)
})
df=pd.concat([df,new_spam_df],ignore_index=True)
Texts=df['Text'].to_list()
Labels=df['Label'].to_list()

#Build vocab from all texts
vocab=build_vocab(Texts)


#split data
X_train_texts,X_val_texts,y_train,y_val=train_test_split(Texts,Labels,test_size=0.2,random_state=42)



#tokenize and augment training data
tokenized_X_train=[tokenize(text) for text in X_train_texts]
augmented_tokenized_X_train=[augment_tokens(tokens) for tokens in tokenized_X_train]
#encode augmented training data
X_train=torch.tensor([encode_tokens(tokens,vocab,max_length=50) for tokens in augmented_tokenized_X_train],dtype=torch.long)
y_train=torch.tensor(y_train,dtype=torch.float32)
#tokenize and encode validation
tokenized_X_val=[tokenize(text) for text in X_val_texts]
X_val=torch.tensor([encode_tokens(tokens,vocab,max_length=50)for tokens in tokenized_X_val],dtype=torch.long)
y_val=torch.tensor(y_val,dtype=torch.float32)
#calculate class weights safely
Labels_unique,counts=torch.unique(y_train,return_counts=True)
#create a dictionary of class weights
class_weights={}
for label,count in zip(Labels_unique,counts):
     count_val=count.item()
     if count_val==0 or count_val!=count_val:
          print(f"Skipping label {label} with invalid count:{count_val}")
          continue
     class_weights[int(label.item())]=1.0/count_val
pos_weight=torch.tensor([class_weights.get(1,1.0)/class_weights.get(0,1.0)])
#assign sample weights based on label
samples_weight=torch.tensor([class_weights.get(int(label),1.0)for label in y_train])
#create a weightedrandomsampler
sampler=WeightedRandomSampler(weights=samples_weight,num_samples=len(samples_weight),replacement=True)
#create a dataloader for batching
num_samples=len(samples_weight),replacement=True
train_loader=DataLoader(TensorDataset(X_train,y_train),batch_size=BATCH_SIZE,sampler=sampler)
val_loader=DataLoader(TensorDataset(X_val,y_val),batch_size=BATCH_SIZE,shuffle=False)
DataLoader(TensorDataset(X_val,y_val),batch_size=BATCH_SIZE,shuffle=False)


model=SpamHamModel(len(vocab),EMBED_DIM,HIDDEN_DIM,1,DROPOUT)
loss_fn=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer=optim.Adam(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
early_stopper=EarlyStopping(patience=PATIENCE)

best_val_loss=float('inf')#initialize best validation loss to infinity
best_model_state=None
best_epoch=-1
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x).squeeze(-1)
        loss = loss_fn(output, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train_loss += loss.item()
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_output = model(val_x).squeeze(-1)
            val_loss = loss_fn(val_output, val_y)
            total_val_loss += val_loss.item()
    avg_train_loss=total_train_loss/len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    if avg_val_loss < best_val_loss:
         best_val_loss=avg_val_loss
         best_epoch=epoch
         best_model_state=model.state_dict()
         print(f"New best model found at epoch {epoch} with val loss {avg_val_loss:.4f}")
    if early_stopper(avg_val_loss):
       print(f"Early stopping triggered at epoch {epoch}")
       break 

with open(os.path.join(VOCAB_DIR,"vocab.json"),"w",encoding="utf-8")as file:
    json.dump(vocab,file,ensure_ascii=False,indent=2)

if best_model_state is not None:
    print(f"Training complete.Best epoch:{best_epoch},Val Loss:{best_val_loss:.4f}")

else:
    print("Training complete.No model improvements were found to save")