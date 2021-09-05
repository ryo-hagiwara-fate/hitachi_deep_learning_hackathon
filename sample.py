def build_model():
    # CNNのパラメータ設定
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=X_train.shape[1:]))  # レイヤー1
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))  # レイヤー2
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3)))  # レイヤー3
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding="same"))  # レイヤー4
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3)))  # レイヤー5
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3))) # レイヤー6
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, (3, 3), padding="same"))  # レイヤー7
    model.add(Activation("relu"))
    model.add(Conv2D(128, (3, 3)))  # レイヤー8
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3))) # レイヤー9
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())  # レイヤー10
    model.add(Dense(512))  # レイヤー11
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(5))  # 12
    model.add(Activation("softmax"))# ソフトマックス
    # コンパイル
    model.compile(loss="categorical_crossentropy", optimizer="ADAM", metrics=["accuracy"])
    
    return model