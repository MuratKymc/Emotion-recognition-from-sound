import whisper
import os
import numpy as np
import pandas as pd
import unicodedata as ud
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils import to_categorical
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt



#-----FUNCTIONS-----#
latin_letters= {}
def is_latin(uchr):
    try: return latin_letters[uchr]
    except KeyError:
         return latin_letters.setdefault(uchr, 'LATIN' in ud.name(uchr))


def only_roman_chars(unistr):
    return all(is_latin(uchr)
           for uchr in unistr
           if uchr.isalpha())


def delete_nonlatin_rows(temp_df):
    for i in range(len(temp_df)):
       if not only_roman_chars(temp_df['Content'][i]):
          temp_df=temp_df.drop(i)


#Convert sound to text
def audio_transcribe(folder_name, transcribed_texts):
    for file in os.listdir(folder_name):
        file_path = f"{folder_name}/{file}"
        result = model.transcribe(file_path)
        transcribed_texts.append(result["text"])


#Preprocessing for NLTK process
def cleaning_text(text):
    nltk.download('stopwords')
    nltk.download('wordnet')

    stop_words=set(stopwords.words('english'))
    lemmatizer=WordNetLemmatizer()
    text = re.sub(r'[^\w\s]', '', text, re.UNICODE) 
    text = text.lower() 
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [word for word in text if not word in stop_words] 
    text = " ".join(text)
    return text


#NLTK training
def text_analysis_training(training_df):
    for i in training_df.index:
        training_df['Content'][i]=cleaning_text(training_df['Content'][i])
    
    #Vectorizing
    vectorizer=TfidfVectorizer(max_features=5000)
    data_features=vectorizer.fit_transform(training_df['Content'])

    #Splitting the dataset
    le = LabelEncoder()
    x=pd.DataFrame(data_features.toarray())
    y=pd.DataFrame(training_df['Sentiment'])
    y=le.fit_transform(y)
    x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.3, random_state=42)
    
    #Label encoder's real values
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_name_mapping)

    #Training XGBoost Classifier
    model= xgb.XGBClassifier(max_depth=10, n_estimators=1000, learning_rate=0.01)
    model.fit(x_train, y_train)
    predictions=model.predict(x_test)
    print(classification_report(y_test, predictions))
    return model


#Used to make predictions with NLTK model
def text_analysis_prediction_maker(temp_model, independant_vals_df):
    predictions=model.predict(independant_vals_df['Content'])
    return predictions

#Extracts the sound features
def extract_sound_feature(audio_file, max_length=100):
    audio, sr=librosa.load(audio_file, duration=4, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=max_length).T, axis=0)
    return mfcc

#Importing sound files
def import_sound_files(table_df, sound_df, folder_name):
    for i in range(len(table_df)):
       file_name = f"{folder_name}/{table_df['fileName'][i]}"
       features = extract_sound_feature(file_name, 40)

       temp_df=pd.DataFrame({
           'fileName': [table_df['fileName'][i]],
           'soundFile': [features]
           }) 
       sound_df=pd.concat([sound_df, temp_df], ignore_index=True)
    return sound_df

def lstm_training_preprocess(table_df):
    le_x = LabelEncoder()
    enc_y = OneHotEncoder()
    table_df['Sentiment'] = le_x.fit_transform(table_df['Sentiment']).astype(float)

    #Adding "Sentiment" and "soundFile" column together
    table_df['soundFile'] = np.array(table_df['soundFile'])
    for i in range(len(table_df['soundFile'])):
        table_df['soundFile'][i] = np.append(table_df['soundFile'][i], table_df['Sentiment'][i])
    
    x=table_df['soundFile']
    y=enc_y.fit_transform(table_df[['Emotion']])


    x = [i for i in x]
    x = np.array(x)
    x = np.expand_dims(x, -1)
    y = y.toarray()

    return x,y


#Training LSTM model
def lstm_model_training(table_df):
    x, y= lstm_training_preprocess(table_df)
    print(x.shape)
    print(y)
    
    x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.25, random_state=42)
    
    model = Sequential([
        LSTM(123, return_sequences=False, input_shape=(41,1)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(6, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    history = model.fit(x_train, y_train, validation_split=0.2, epochs=100, batch_size=512, shuffle=True)
    loss, accuracy=model.evaluate(x_test, y_test, verbose=2)
    print('\nTest accuracy: ', accuracy)
    plot_results(history)

    return model


def plot_results(history):
    epochs = list(range(100))
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    plt.plot(epochs, acc, label='train accuracy')
    plt.plot(epochs, val_acc, label='val accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    
    """loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.plot(epochs, loss, label='train loss')
    plt.plot(epochs, val_loss, label='val loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()"""


    
#-----MAIN-----#
transcribed_texts=[]
model = whisper.load_model("base.en")
dataset_folder = "dataset"
document_df=pd.read_excel('emotions.xlsx')
table_df=pd.DataFrame(columns=['fileName','Emotion','Content','Sentiment'])
sound_df=pd.DataFrame(columns=['fileName', 'soundFile'])
"""audio_transcribe(dataset_folder, transcribed_texts)

for i in range(len(document_df)):   
    new_fileName=document_df.loc[i]['fileName']
    new_Emotion=document_df.loc[i]['Emotion']
    new_Sentiment=document_df.loc[i]['Sentiment']
    new_Content=transcribed_texts[i].lower() 

    new_line=pd.DataFrame({
        'fileName': [new_fileName],
        'Emotion': [new_Emotion],
        'Content': [new_Content],
        'Sentiment': [new_Sentiment]
    })

    table_df=pd.concat([table_df,new_line], ignore_index=True)"""

#table_df.to_excel('result.xlsx', index=False) #delete this at the end   
table_df=pd.read_excel('result.xlsx') # Delete this at the end we don't need to read excel
document_df.drop(document_df.index, inplace=True)

#Preprocessing
table_df=table_df.dropna()
table_df=table_df.drop_duplicates()
delete_nonlatin_rows(table_df) 

#Training nltk model
nltk_model=text_analysis_training(table_df)

#Adding sound files to the dataframe
sound_df=import_sound_files(table_df, sound_df, dataset_folder)
table_df=pd.concat([table_df, sound_df], axis=1, join='outer')
sound_df.drop(sound_df.index, inplace=True)
lstm_model = lstm_model_training(table_df)

