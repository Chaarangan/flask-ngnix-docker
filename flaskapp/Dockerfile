FROM python:3

RUN mkdir /emotional
WORKDIR /emotional/
ADD . /emotional/
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c 'from transformers import AutoTokenizer, AutoModel; tokenizer = AutoTokenizer.from_pretrained("./model/tokenizer/"); model = AutoModel.from_pretrained("roberta-base")'
RUN python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"