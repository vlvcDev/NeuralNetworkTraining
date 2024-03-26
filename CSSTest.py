from keras.models import load_model
from keras.datasets import cifar10

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

reloaded_model = load_model("model_dropout")

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

test_loss, test_acc = reloaded_model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc*100:.3f}")

def evaluate_model(dataset, model):
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    num_rows = 3
    num_cols = 3

    data_batch = dataset[0 : num_rows * num_cols]

    predictions = model.predict(data_batch)

    plt.figure(figsize=(20,8))
    num_matches = 0

    for idx in range(num_rows * num_cols):
        ax = plt.subplot(num_rows, num_cols, idx + 1)
        plt.axis("off")
        plt.imshow(data_batch[idx])

        pred_idx = np.argmax(predictions[idx]).numpy()
        true_idx = np.nonzero(y_test[idx])

        title = str(class_names[true_idx[0][0]]) + " / " + str(class_names[pred_idx])
        title_obj = plt.title(title, fontdict={"fontsize": 13})

        if pred_idx == true_idx:
            num_matches += 1
            plt.setp(title_obj, color="g")
        else:
            plt.setp(title_obj, color="r")

        acc = num_matches / (idx+1)

    print("Prediction accuracy: ", int(100 * acc) / 100)

    return

evaluate_model(X_test, reloaded_model)

predictions = reloaded_model.predict(X_test)

predicted_labels = [np.argmax(i) for i in predictions]

y_test_integer_labels = tf.argmax(y_test, axis=1)

cm = tf.math.confusion_matrix(labels=y_test_integer_labels, predictions=predicted_labels)

plt.figure(figsize=[12,6])

import seaborn as sn

sn.heatmap(cm, annot=True, fmt="d", annot_kws={"size": 12})

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
