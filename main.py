import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.src.applications.convnext import preprocess_input
from keras.models import Sequential
from keras import layers
from keras.callbacks import EarlyStopping
from keras.losses import SparseCategoricalCrossentropy
from keras.src.applications.mobilenet_v2 import MobileNetV2
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
from sklearn.utils import class_weight


main_path = "./fruits/"
img_size = (96, 96)
class_names = ["apple", "orange", "banana"]
class_to_label = {name: idx for idx, name in enumerate(class_names)}

def load_dataset(split_name):
    images = []
    labels = []
    split_path = os.path.join(main_path, split_name)
    for class_name in class_names:
        class_path = os.path.join(split_path, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path).convert("RGB")
            img = img.resize(img_size)
            img = np.array(img)
            images.append(img)
            labels.append(class_to_label[class_name])
    return np.array(images), np.array(labels)

Xtrain, y_train = load_dataset("train")
Xval, y_val = load_dataset("validation")
Xtest, y_test = load_dataset("test")

print("Train shape:", y_train.size)
print(f"apple: {np.sum(y_train==0)} orange: {np.sum(y_train==1)} banana: {np.sum(y_train==2)}")

print("Validation shape:", y_val.size)
print(f"apple: {np.sum(y_val==0)} orange: {np.sum(y_val==1)} banana: {np.sum(y_val==2)}")

print("Test shape:", y_test.size)
print(f"apple: {np.sum(y_test==0)} orange: {np.sum(y_test==1)} banana: {np.sum(y_test==2)}")
plt.figure()
plt.hist([y_train,y_val,y_test])
plt.title('Distribucija klasa - train')
plt.legend(['train','val','test'])
plt.grid()
plt.show()

Xtrain = preprocess_input(Xtrain)
Xval = preprocess_input(Xval)
Xtest = preprocess_input(Xtest)

def shuffle_data(X, y):
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]

Xtrain, y_train = shuffle_data(Xtrain, y_train)
Xval, y_val = shuffle_data(Xval, y_val)
Xtest, y_test = shuffle_data(Xtest, y_test)

plt.figure()
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(Xtrain[i])
    plt.title(class_names[y_train[i]])
    plt.axis("off")
plt.show()

data_augmentation = Sequential([
    layers.Input(shape=(img_size[0], img_size[1], 3)),
    layers.RandomFlip("horizontal"),
    layers.RandomFlip("vertical"),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.1),
])

plt.figure()
for i in range(10):

    img = np.expand_dims(Xtrain[i], axis=0)
    # augmentacija
    aug_img = data_augmentation(img, training=True)
    # prikaz
    plt.subplot(2,5,i+1)
    plt.imshow(aug_img[0].numpy().astype('uint8'))
    plt.axis("off")
plt.show()

def cnn_model(num_classes):
    model = Sequential([
        data_augmentation,
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model


model = cnn_model(num_classes=3)
model.summary()

es = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
weights = class_weight.compute_class_weight(class_weight='balanced',
                                            classes=np.unique(y_train),
                                            y=y_train)
print('Težine klasa su: K0 - ' + str(weights[0]) + ', K1 - ' + str(weights[1])+ ' K2 - ' + str(weights[2]))
history = model.fit(
    Xtrain, y_train,
    epochs=30,
    validation_data=(Xval, y_val),
    class_weight={0:weights[0], 1:weights[1], 2:weights[2]},
    callbacks=[es],
    verbose=1
)

plt.figure()
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss')
plt.legend()
plt.show()

y_pred_train = np.argmax(model.predict(Xtrain), axis=1)
y_true = y_train

acc = accuracy_score(y_train, y_pred_train)
print(f"\nTacnost: {acc:.4f}")

# Precision, Recall, F1 po klasi
print("\nDetaljni izveštaj po klasama(train):")
print(classification_report(y_train, y_pred_train))

cm = confusion_matrix(y_true, y_pred_train, normalize='true')
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot()
plt.show()

y_pred_test = np.argmax(model.predict(Xtest), axis=1)
y_true = y_test

acc = accuracy_score(y_test, y_pred_test)
print(f"\nTacnost: {acc:.4f}")

# Precision, Recall, F1 po klasi
print("\nDetaljni izveštaj po klasama(test):")
print(classification_report(y_test, y_pred_test))

cm = confusion_matrix(y_true, y_pred_test, normalize='true')
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot()
plt.show()

correct = np.where(y_test == y_pred_test)[0]

plt.figure(figsize=(10, 5))
for i in range(min(10, len(correct))):
    idx = correct[i]
    plt.subplot(2, 5, i + 1)
    plt.imshow(Xtest[idx])
    plt.title(f"Test:{class_names[y_test[idx]]}\nPredikcija:{class_names[y_pred_test[idx]]}")
    plt.axis("off")
plt.suptitle("Dobro klasifikovane slike")
plt.show()

wrong = np.where(y_test != y_pred_test)[0]

plt.figure(figsize=(10, 5))
for i in range(min(10, len(wrong))):
    idx = wrong[i]
    plt.subplot(2, 5, i + 1)
    plt.imshow(Xtest[idx])
    plt.title(f"Test:{class_names[y_test[idx]]}\nPredikcija:{class_names[y_pred_test[idx]]}")
    plt.axis("off")
plt.suptitle("Loše klasifikovane slike")
plt.show()

def transfer_learning(img_size, num_classes):
    base_model = MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False  # zamrzavamo bazni model

    model = Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model


tf_model = transfer_learning(img_size, 3)
tf_model.summary()


s = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)

history = tf_model.fit(Xtrain,y_train, validation_data=(Xval, y_val),
                       epochs=30,
                       callbacks=[s],
                       class_weight={0: weights[0], 1: weights[1], 2: weights[2]},
                       verbose=0
)
plt.figure()
plt.subplot(121)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.show()

y_pred_val = np.argmax(tf_model.predict(Xval), axis=1)
y_true_val = y_val

acc = accuracy_score(y_true_val, y_pred_val)
print(f"\nTacnost (val): {acc:.4f}")

print("\nDetaljni izveštaj po klasama (val):")
print(classification_report(y_true_val, y_pred_val))

cm = confusion_matrix(y_true_val, y_pred_val, normalize='true')
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot()
plt.show()

y_pred_test = np.argmax(tf_model.predict(Xtest), axis=1)
y_true_test = y_test

acc_test = accuracy_score(y_true_test, y_pred_test)
print(f"\nTacnost (test): {acc_test:.4f}")

print("\nDetaljni izveštaj po klasama (test):")
print(classification_report(y_true_test, y_pred_test))

cm_test = confusion_matrix(y_true_test, y_pred_test, normalize='true')
ConfusionMatrixDisplay(
    confusion_matrix=cm_test,
    display_labels=class_names
).plot()

plt.title("Konfuziona matrica - Test")
plt.show()
