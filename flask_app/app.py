from utils import BERTDataset, infer, text_preprocessing_pipeline
from torch.utils.data import DataLoader
from flask import current_app
import torch
import numpy as np
import pandas as pd

from flask import Flask, render_template, url_for, request
device = 'cuda' if torch.cuda.is_available() else 'cpu'
labels = ["anger", "fear", "happy", "sarcasm", "irony", "sadness"]

model = None
tokenizer = None
server = Flask(__name__)


@server.route('/')
def home_endpoint():
    return 'You have invoked EmotionalClassifier App'


@server.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.get_json()['data']
        text = text_preprocessing_pipeline(review)

        # Creating dataframe from list
        df_infer = [[text, 0]]
        df_infer = pd.DataFrame(df_infer, columns=["Text", "Class"])

        infer_dataset = BERTDataset(
            df_infer, current_app.tokenizer, 128)
        infer_loader = DataLoader(infer_dataset, batch_size=64,
                                num_workers=4, shuffle=True, pin_memory=True)

        outputs, targets = infer(current_app.model, infer_loader)
        outputs = np.argmax(outputs, axis=-1)

        return labels[outputs[0]]
