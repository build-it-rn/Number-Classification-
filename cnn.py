import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


mnist =tf.keras.datasets.mnist
(x_train, y_train), (x_test,y_test)= mnist.load_data()

x_train = x_train/255.0
x_test = x_test/255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


model=Sequential([
    Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),# how does 2 by 2 reduce the size 

    Conv2D(filters=64, kernel_size=(3,3),activation="relu"),
    MaxPooling2D(pool_size=(2,2)),


    Flatten(), #why are we flattening the data doesnt it lose the grid form 

    Dense(128,activation='relu'),

    Dense(10,activation="softmax"),

      ])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics =['accuracy']
    )

model.fit(x_train,y_train,epochs=10,validation_data=(x_test,y_test))

test_loss,test_acc=model.evaluate(x_test,y_test,verbose=2)

print(f"Test accuracy: {test_acc}")