import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import image_dataset_from_directory
import os
from PIL import Image

# Ограничение потоков TensorFlow для снижения нагрузки на CPU
os.environ["TF_NUM_INTEROP_THREADS"] = "3"
os.environ["TF_NUM_INTRAOP_THREADS"] = "3"
os.environ["OMP_NUM_THREADS"] = "4"

# Пути к данным
DATASET_PATH = "./dataset"  # Укажи путь к директории, содержащей папки Cat и Dog
BATCH_SIZE = 16
IMG_SIZE = (800, 600)

# Включение mixed precision для ускорения вычислений (если поддерживается)
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

# Загрузка данных
train_ds = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.3,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
)

val_test_ds = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.3,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
)

val_size = len(val_test_ds) // 2
val_ds = val_test_ds.take(val_size)
test_ds = val_test_ds.skip(val_size)

# Нормализация изображений
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=2)
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=2)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=2)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# Конфигурация сети (Conf6)
model = Sequential(
    [
        Input(shape=(800, 600, 3)),  # Явное указание входного слоя
        Conv2D(64, (4, 4), strides=(1, 1), activation="tanh", padding="same"),
        Conv2D(64, (2, 2), strides=(2, 2), activation="tanh", padding="same"),
        Conv2D(32, (3, 3), strides=(1, 1), activation="tanh", padding="same"),
        Conv2D(32, (3, 3), strides=(1, 1), activation="tanh", padding="same"),
        MaxPooling2D(pool_size=(2, 2)),
        Dense(64, activation="tanh"),
        Dense(64, activation="tanh"),
        Dropout(0.2),
        Flatten(),
        Dense(1, activation="sigmoid"),
    ]
)

# Компиляция модели
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Обучение модели
EPOCHS = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# Визуализация результатов
plt.figure(figsize=(12, 6))
plt.plot(history.history["val_loss"], label="Test Loss")
plt.xlabel("Эпохи")
plt.ylabel("Ошибка")
plt.legend()
plt.title("Ошибка тестирования")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history["loss"], label="Train Loss")
plt.xlabel("Эпохи")
plt.ylabel("Ошибка")
plt.legend()
plt.title("Ошибка обучения")
plt.show()

# Оценка на тестовых данных
test_loss, test_acc = model.evaluate(test_ds)
print(f"Точность на тестовых данных: {test_acc:.4f}")


# Вывод предсказаний на 5 изображениях каждого класса
def load_and_preprocess_image(file_path):
    img = Image.open(file_path).resize(IMG_SIZE)
    img = np.array(img) / 255.0  # Нормализация
    return img


sample_images = []
sample_labels = []

for category in ["Cat", "Dog"]:
    folder_path = os.path.join(DATASET_PATH, category)
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path)[:5]]
    sample_images.extend([load_and_preprocess_image(img) for img in images])
    sample_labels.extend([category] * 5)

sample_images = np.array(sample_images)
predictions = model.predict(sample_images)

plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample_images[i])
    plt.title(f"Predicted: {'Dog' if predictions[i] > 0.5 else 'Cat'}")
    plt.axis("off")
plt.show()
