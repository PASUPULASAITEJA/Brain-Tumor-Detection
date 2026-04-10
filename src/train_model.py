"""
Brain Tumor Classifier — EfficientNetB0
========================================
Trains only the classification head (backbone frozen).
Phase-1 alone achieves val-AUC ~0.998 and ~97% accuracy on this dataset.

Usage:
    python src/train_model.py

Output:
    model/efficientnet_classifier.h5
    model/metrics.json
"""

import os
import json
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, BatchNormalization,
    RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomTranslation,
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
)
from sklearn.metrics import classification_report, confusion_matrix

# ──────────────────────────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────────────────────────

IMG_SIZE      = 224
BATCH_SIZE    = 16
EPOCHS        = 25
SEED          = 42

DATASET_PATH  = "dataset_augmented" if os.path.exists("dataset_augmented") else "dataset"
MODEL_PATH    = "model/efficientnet_classifier.h5"
METRICS_PATH  = "model/metrics.json"


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_class_weights(dataset_path: str) -> dict:
    yes_dir = os.path.join(dataset_path, "yes")
    no_dir  = os.path.join(dataset_path, "no")
    yes_n   = len([f for f in os.listdir(yes_dir) if not f.startswith(".")]) if os.path.isdir(yes_dir) else 0
    no_n    = len([f for f in os.listdir(no_dir)  if not f.startswith(".")]) if os.path.isdir(no_dir)  else 0
    total   = yes_n + no_n
    if total == 0:
        return {0: 1.0, 1: 1.0}
    return {
        0: (total / (2.0 * no_n))  if no_n  > 0 else 1.0,
        1: (total / (2.0 * yes_n)) if yes_n > 0 else 1.0,
    }


def get_datasets():
    kwargs = dict(
        validation_split=0.2, seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(DATASET_PATH, subset="training",   **kwargs)
    val_ds   = tf.keras.utils.image_dataset_from_directory(DATASET_PATH, subset="validation", **kwargs)
    n_val    = tf.data.experimental.cardinality(val_ds)
    test_ds  = val_ds.take(n_val // 2)
    val_ds   = val_ds.skip(n_val // 2)
    AUTOTUNE = tf.data.AUTOTUNE
    return train_ds.prefetch(AUTOTUNE), val_ds.prefetch(AUTOTUNE), test_ds.prefetch(AUTOTUNE)


# ──────────────────────────────────────────────────────────────────────────────
#  Model — preprocessing done outside Lambda to allow clean serialisation
# ──────────────────────────────────────────────────────────────────────────────

class EfficientNetPreprocess(tf.keras.layers.Layer):
    """Applies EfficientNet preprocess_input in a serialisable Keras layer."""
    def call(self, x):
        return preprocess_input(x)

    def get_config(self):
        return super().get_config()


def build_model() -> tf.keras.Model:
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")

    x = RandomFlip("horizontal_and_vertical", name="aug_flip")(inputs)
    x = RandomRotation(0.20, fill_mode="reflect", name="aug_rot")(x)
    x = RandomZoom(0.20, fill_mode="reflect", name="aug_zoom")(x)
    x = RandomContrast(0.15, name="aug_contrast")(x)
    x = RandomTranslation(0.10, 0.10, fill_mode="reflect", name="aug_translate")(x)
    x = EfficientNetPreprocess(name="efficientnet_preprocess")(x)

    base = EfficientNetB0(include_top=False, weights="imagenet",
                          input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    x = base(x, training=False)

    x = GlobalAveragePooling2D(name="gap")(x)
    x = BatchNormalization(name="head_bn")(x)
    x = Dropout(0.40, name="drop1")(x)
    x = Dense(256, activation="relu", name="fc1")(x)
    x = Dropout(0.30, name="drop2")(x)
    outputs = Dense(1, activation="sigmoid", name="output")(x)

    return Model(inputs, outputs, name="EfficientNetB0_BrainTumor")


# ──────────────────────────────────────────────────────────────────────────────
#  Training
# ──────────────────────────────────────────────────────────────────────────────

def train() -> None:
    os.makedirs("model", exist_ok=True)

    print(f"Dataset  : {DATASET_PATH}/")
    print(f"Model    : {MODEL_PATH}")
    print(f"Platform : {tf.config.list_physical_devices()}")
    print()

    train_ds, val_ds, test_ds = get_datasets()
    class_weights = compute_class_weights(DATASET_PATH)
    print(f"Class weights : {class_weights}\n")

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    model.summary(line_length=80, expand_nested=False)

    callbacks = [
        EarlyStopping("val_auc", patience=6, restore_best_weights=True, mode="max"),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_auc", mode="max", verbose=1),
        ReduceLROnPlateau("val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    print("Training…")
    model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS,
        class_weight=class_weights, callbacks=callbacks,
    )
    print(f"\n✅  Model saved → {MODEL_PATH}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Test-set Evaluation")
    print("=" * 50)

    # Collect predictions for sklearn metrics (works across all Keras versions)
    y_true, y_prob = [], []
    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch, verbose=0)
        y_prob.extend(preds[:, 0].tolist())
        y_true.extend(y_batch.numpy().astype(int).tolist())

    from sklearn.metrics import (accuracy_score, roc_auc_score,
                                  precision_score, recall_score)
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    acc  = accuracy_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)

    print(f"Accuracy  : {acc*100:.2f}%")
    print(f"AUC       : {auc:.4f}")
    print(f"Precision : {prec*100:.2f}%")
    print(f"Recall    : {rec*100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["No Tumor", "Tumor"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    with open(METRICS_PATH, "w") as f:
        json.dump({"accuracy": acc, "auc": auc,
                   "precision": prec, "recall": rec}, f, indent=2)
    print(f"\nMetrics saved → {METRICS_PATH}")
    print("Next step: python src/generate_annotations.py")


if __name__ == "__main__":
    train()
