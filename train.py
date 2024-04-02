import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report

dataset_dir = "Data"

images = []
labels = []

for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_dir):
        label = int(class_name)  
        print(label)

        for image_file in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_file)
            image = cv2.imread(image_path) 
            image = cv2.resize(image, (224, 224))  
            cv2.imshow("image",image) 
            cv2.waitKey(20)
            image = image / 255.0  
            images.append(image)
            labels.append(label)
        cv2.destroyAllWindows() 

images = np.array(images)
labels = np.array(labels)

print(images)

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
print(y_train)

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(6, activation='softmax') 
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

hist = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

model.save('keras_model.h5')


test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy}')

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes, average='weighted')
recall = recall_score(y_test, y_pred_classes, average='weighted')
f1 = f1_score(y_test, y_pred_classes, average='weighted')
confustion_matrix = confusion_matrix(y_test, y_pred_classes)

print("Confusion matrix:")
print(confustion_matrix)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

class_names = ['A','B','C','D','E','F']  
classification_rep = classification_report(y_test, y_pred_classes, target_names=class_names)
print("Classification Report:")
print(classification_rep)



epochs = [i for i in range(5)]
train_errors = [0 for i in range(5)]
vall_errors = [0 for i in range(5)]
fig,ax = plt.subplots(1,3)
train_acc = hist.history['accuracy']
for i in range(len(train_acc)):
    train_errors[i] = 1- train_acc[i]
train_loss = hist.history['loss']
vall_acc= hist.history['val_accuracy']
for i in range(len(train_acc)):
    vall_errors[i] = 1- vall_acc[i]
vall_loss=hist.history['val_loss']
fig.set_size_inches(10,7)

ax[0].plot(epochs, train_acc, color = 'red', marker = 'o', linestyle = '-', label='Train Acc')
ax[0].plot(epochs, vall_acc, color = 'blue', marker = 'o', linestyle = '--', label= 'Test Acc')
ax[0].set_title('Train and Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')

ax[1].plot(epochs, train_loss, color = 'red', marker = 'o', linestyle = '-', label='Train Loss')
ax[1].plot(epochs, vall_loss, color = 'blue', marker = 'o', linestyle = '--', label= 'Test Loss')
ax[1].set_title('Train and Validation Loss')
ax[1].legend()
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')


ax[2].plot(epochs, train_errors, color = 'red', marker = 'o', linestyle = '-', label='Train error')
ax[2].plot(epochs, vall_errors, color = 'blue', marker = 'o', linestyle = '--', label= 'Test error')
ax[2].set_title('Train and Validation errors')
ax[2].legend()
ax[2].set_xlabel('Epochs')
ax[2].set_ylabel('errors')

plt.show()



