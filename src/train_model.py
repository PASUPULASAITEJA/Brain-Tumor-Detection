import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
DATASET_PATH = 'dataset'
MODEL_PATH = 'model/brain_tumor_model.h5'

def compute_class_weights(dataset_path: str) -> dict:
    yes_path = os.path.join(dataset_path, 'yes')
    no_path = os.path.join(dataset_path, 'no')
    yes_count = len(os.listdir(yes_path)) if os.path.exists(yes_path) else 0
    no_count = len(os.listdir(no_path)) if os.path.exists(no_path) else 0
    total = yes_count + no_count
    if total == 0: return {0: 1.0, 1: 1.0}
    weight_0 = (1 / no_count) * (total / 2.0) if no_count > 0 else 1.0
    weight_1 = (1 / yes_count) * (total / 2.0) if yes_count > 0 else 1.0
    return {0: weight_0, 1: weight_1}

def create_model() -> tf.keras.Model:
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    data_augmentation = Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.2),
        RandomZoom(0.2)
    ], name='data_augmentation')

    model = Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        data_augmentation,
        tf.keras.layers.Rescaling(1./127.5, offset=-1), 
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train() -> None:
    os.makedirs("model", exist_ok=True)
    print("Loading datasets...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH, validation_split=0.2, subset="training", seed=123,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH, validation_split=0.2, subset="validation", seed=123,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE
    )

    val_batches = tf.data.experimental.cardinality(val_ds)
    test_ds = val_ds.take(val_batches // 2)
    val_ds = val_ds.skip(val_batches // 2)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    class_weights = compute_class_weights(DATASET_PATH)
    model = create_model()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy')
    ]

    print("Starting training...")
    model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS,
        class_weight=class_weights, callbacks=callbacks
    )
    print(f"✅ Model saved to {MODEL_PATH}")
    print("\nEvaluating model on isolated test set...")
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test Accuracy: {accuracy*100:.2f}%\n")

if __name__ == '__main__':
    train()