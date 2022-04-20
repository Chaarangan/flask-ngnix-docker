from utils import BERTClass
from transformers import AutoTokenizer
from app import server
from flask import current_app
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == "__main__":
    with server.app_context():
        current_app.model = BERTClass(6)
        current_app.model.to(device)
        current_app.model.load_state_dict(torch.load(
            './model/model.bin', map_location=torch.device('cpu')))
        current_app.tokenizer = AutoTokenizer.from_pretrained(
            "./model/tokenizer/")
    # config param
    server.run(host='0.0.0.0', port=8000, debug=True)

