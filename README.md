#  Spam/Ham Message Classifier (PyTorch)

A deep learning project built with *PyTorch* to classify SMS or email messages as *Spam* or *Ham (Not Spam)*.  
The model uses *custom tokenization, **vocabulary building, and a **Bidirectional LSTM* for accurate text classification.

---

##  Features

- *Custom Vocabulary* built from training data (vocab.json)
- *Text Preprocessing* (tokenization, lowercasing, cleaning)
- *Embedding + BiLSTM Neural Network*
- *Weighted Sampling* to handle class imbalance
- *Early Stopping* to prevent overfitting
- *Full Logging* for every training stage
- *Model Saving* (best_spamham_model.pth) and easy inference

---

##  Model Architecture

##  Project Structure

├── data/ │   └── messages.csv                # Dataset ├── vocab/ │   └── vocab.json                  # Token vocabulary (used for predictions) ├── models/ │   └── best_spamham_model.pth      # Trained model (ignored in .gitignore) ├── train.py                        # Training script ├── predict.py                      # Inference / prediction script ├── utils.py                        # Preprocessing and helper functions ├── requirements.txt                # Dependencies ├── .gitignore └── README.md


##  Installation

```bash
# Clone the repository
git clone https://github.com/opeblow/SPAM-HAM.git
cd <your-repo-name>

# Create and activate virtual environment (optional)
python -m venv myenv
source myenv/bin/activate      # On Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

 Training the Model

python train.py

This will:

Preprocess and encode the dataset

Train the BiLSTM model

Save the best weights to best_spamham_model.pth



---

 Making Predictions

python predict.py

Example:

> Enter a message: "Congratulations! You've won a free ticket!"
> Prediction: SPAM

 Author
 Mobolaji Opeyemi 
EMAIL:opeblow2021@gmail.com
