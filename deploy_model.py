import numpy as np
import joblib
import streamlit as st
import pandas as pd
from underthesea import word_tokenize
import re
from underthesea import sent_tokenize,word_tokenize
import re
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import Word
from nltk.util import ngrams
from wordcloud import WordCloud, STOPWORDS
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer


def processcing_text(text):
    tweet = text
    #lower
    tweet = tweet.lower()
    #convert any url link to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
    #convert any @Username to AT_USER
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet)

    #Remove not alphanumeric symbols white spaces
    tweet = re.sub(r'[^\w]',' ', tweet)
    #Removes # hashtag in front of a word
    tweet = re.sub(r'#([\w]+)', r'\1', tweet)
    tweet = re.sub(r'#([^\s]+)',r'\1',tweet)
    #remove :( or :)
    tweet = tweet.replace(':)','')
    tweet = tweet.replace(':(','')
    #remove numbers
    tweet = ''.join([i for i in tweet if not i.isdigit()])
    #remove multiple exclamation
    tweet = re.sub(r'(!)\1+', ' ', tweet)
    #remove multiple question marks
    tweet = re.sub(r'(\?)\1+','', tweet)
    #remove multistop
    tweet = re.sub(r'(\.)\1+','', tweet)
    #Remove additional whitespace
    tweet = re.sub(r'[\s]+',' ', tweet)
    tweet = re.sub(r'[\n]+',' ', tweet)
    #lemma 
    # tweet = " ".join([Word(word).lemmatize() for word in tweet.split()])
    #stemmer
    # st=  PorterStemmer()
    # tweet = " ".join([st.stem(word) for word in tweet.split()])
    #remove emoteicon from text
    # tweet = re.sub('')
    row = tweet
    return row

def toke(text):

    text=word_tokenize(text)
    # t=word_tokenize(t)
    return text


def prediction(model,tfidf, text):
    text = processcing_text(text)
    text = toke(text)
    text = tfidf.transform(text).toarray()
    prediction = model.predict(text)[0]
    if prediction == 0:
        return 'Nội dụng này thuộc thể loại chính trị ','https://bcp.cdnchinhphu.vn/zoom/670_420/334894974524682240/2023/9/10/cc92e5e8f5a419f634ec2b822b8634c6-1-1694344383733478186034-0-0-1250-2000-crop-16943503165941042667791.jpg'
    elif prediction == 1:
        return 'Nội dung nay thuộc thể loại giải trí ','https://ict-imgs.vgcloud.vn/2022/01/03/10/giai-tri-truc-tuyen-bung-no-tang-truong-vuot-bac-nho-dai-dich.jpg'
    elif prediction == 2:
        return 'Nội dung nay thuộc thể loại giao thông ','https://media.vneconomy.vn/images/upload/2023/08/21/bao-dam-trat-tu-an-toan-giao-thong-dip-quoc-khanh-2-9-va-thang-cao-diem-an-toan-giao-thong.png'
    elif prediction == 3:
        return 'Nội dung nay thuộc thể loại kinh doanh ','https://cdn.luatminhkhue.vn/lmk/articles/17/88199/ca-nhan-ho-gia-dinh-phai-dang-ky-kinh-doanh-88199.jpeg'
    else :
        return 'Nội dung nay thuộc thể loại thể thao ','https://www.itec.hcmus.edu.vn/edu/images/stories/tu-vung-tieng-han-ve-cac-mon-the-thao.jpg'

loaded_model = joblib.load('nb_classifier.pkl')
loaded_tfidf = joblib.load('tfidf_vectorizer.pkl')
def main():
    st.title('Prediction category paper')
    st.title('20116351 - Lê Thành Nghĩa')
    text_input = st.text_input('Nhập nội dung văn bản để predict')

    prediction_cate = ''
    cate_img = ''
    if st.button('Predict'):
        prediction_cate,cate_img = prediction(loaded_model,loaded_tfidf, text_input)
        st.success(prediction_cate)
        st.image(cate_img, width=300)

if __name__ == '__main__':
    main()



# X_new = np.array([[5.1, 3.5, 1.4, 0.2]])
# X_var = np.array([0.6856935123042505,0.18800402684563763,3.1131794183445156,0.5824143176733784])
# X_new = X_new.reshape(1,-1)
# X_var = X_var.reshape(1,-1)

# prediction = loaded_model.predict(X_new)
# prediction2 = loaded_model.predict(X_var)
# print(prediction)
# print(prediction2)

