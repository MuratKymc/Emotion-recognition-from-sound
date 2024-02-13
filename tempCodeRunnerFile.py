def extract_sound_feature(audio_file, max_length=100):
    audio, sr=librosa.load(audio_file, sr=None)
    mfccs=librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    mfccs_normalized=(mfccs - mfccs.mean()) / mfccs.std()
    mfccs_padded = pad_sequences([mfccs_normalized], maxlen=max_length, padding='post', truncating='post')[0]
    return mfccs_padded

#Importing sound files
def import_sound_files(table_df, sound_df, folder_name):
    for i in range(len(table_df)):
       file_name = f"{folder_name}/{table_df['fileName'][i]}"
       features = extract_sound_feature(file_name, 20)
       temp_df=pd.DataFrame({
           'fileName': [table_df['fileName'][i]],
           'soundFile': [features]
           }) 
       sound_df=pd.concat([sound_df, temp_df], ignore_index=True)
    return sound_df


#Training LSTM model
def lstm_model_training(table_df):
    le_x = LabelEncoder()
    le_y = LabelEncoder()
    table_df['Sentiment'] = le_x.fit_transform(table_df['Sentiment'])
    table_df['Emotion'] = le_y.fit_transform(table_df['Emotion'])

    #Adding "Sentiment" and "soundFile" column together
    table_df['soundFile'] = np.array(table_df['soundFile'])
    for i in range(len(table_df['soundFile'])):
        table_df['soundFile'][i] = np.append(table_df['soundFile'][i], table_df['Sentiment'][i])
    
    x=table_df['soundFile']
    y=np.array(table_df['Emotion'])

    
    x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.25, random_state=42)
    x_train, x_validation, y_train, y_validation= train_test_split(x_train, y_train, test_size=0.2)
    
    input_shape = (101, 1)  # Correct input shape
    model= keras.Sequential()
    #Adding layers
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))

    #Dense layers
    model.add(keras.layers.Dense(64, activation= 'relu'))
    model.add(keras.layers.Dropout(0.3))
    
    #Output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    #Compiling 
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    #Training
    model.fit(x_train, y_train, validation_data=(x_validation, y_validation), batch_size=32, epochs=30)
    loss, accuracy=model.evaluate(x_test, y_test, verbose=2)
    print('\nTest accuracy: ', accuracy)

    return model
