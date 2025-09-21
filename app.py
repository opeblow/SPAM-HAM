import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
import json 
import uvicorn
class SpamHamModel(nn.Module):#model definition
    def __init__(self, vocab_size, embedded_dim,hidden_dim,output_dim,dropout_prob):
        super(SpamHamModel,self).__init__()
        self.embedding=nn.Embedding(vocab_size,embedded_dim)
        self.lstm=nn.LSTM(embedded_dim,hidden_dim,batch_first=True,bidirectional=True)
        self.dropout=nn.Dropout(dropout_prob)
        self.fc=nn.Linear(hidden_dim*2,output_dim)
    def forward(self,x):
            embedded=self.embedding(x)
            output,_=self.lstm(embedded)#passing embeddings through lstm
            last_output=output[:,-1,:]
            dropped=self.dropout(last_output)
            return self.fc(dropped)
with open("vocab.json","r")as file:
     vocab=json.load(file)
vocab_size=len(vocab)
embedded_dim=64
hidden_dim=128
output_dim=1 
dropout_prob=0.3
model=SpamHamModel(vocab_size,embedded_dim,hidden_dim,output_dim,dropout_prob)
model.load_state_dict(torch.load("best_spamham_model.pth",map_location=torch.device('cpu')))
model.eval()
def clean_text(Text):
       url_pattern=r'http[s]?://\s+|www\.s+|\b\w+\.(com|net|org|xyz|cc|biz|info|io|ly)\b'
       Text=re.sub(url_pattern,'<URL',Text,flags=re.IGNORECASE)
       return Text.lower()
def tokenize(Text):#tokenization
      Text=clean_text(Text)
      return re.findall(r'\b\w+\b',Text.lower())
def encode_tokens(tokens,vocab,max_length=50):
    idx=[vocab.get(word,vocab["<UNK>"]) for word in tokens]
    padded=idx + [0] * (max_length - len(idx))
    return padded[: max_length]
app=FastAPI(title="SPAM/HAM TEXT CLASSIFIERS API")
class message(BaseModel):
     text:str
@app.get("/")
def root():
     return{"message:WELCOME TO THE SPAM/HAM CLASSIFICATION API.POST/PREDICT WITH JSON"}
@app.post("/predict")
def predict_spamham(message):
     text=message.text
     if not text:
          raise HTTPException(status_code=400,detail="TEXT INPUT IS REQUIRED")
     tokens=tokenize(text)
     encoded=encode_tokens(tokens,vocab) 
     input_tensor=torch.tensor([encoded],dtype=torch.long)
     with torch.no_grad():
          output=model(input_tensor).squeeze(-1)
          prob=torch.sigmoid(output).item()
          label="spam" if prob > 0.5 else "ham"
          return label,prob
if __name__=="__main__":
     uvicorn.run("app:app",host="localhost",port=8000,reload=True)