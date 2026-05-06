"""
Fixed Training script for ETD-Hybrid GRU + Attention model
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from loguru import logger

from model_architecture import build_model_with_mixed_precision, build_model

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "final_dataset"
MODEL_DIR = BASE_DIR / "backend" / "model"
LOG_DIR = BASE_DIR / "logs" / "training"

SEQUENCE_LENGTH = 20
BATCH_SIZE = 32
EPOCHS = 150
VAL_SPLIT = 0.15
TEST_SPLIT = 0.05
RANDOM_SEED = 42
USE_MIXED_PRECISION = True


# SAFE GPU CONFIG
def configure_gpu():
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"✅ GPU enabled: {[g.name for g in gpus]}")
            return True
    except Exception as e:
        logger.warning(f"GPU config failed: {e}")

    logger.warning(" Running on CPU")
    return False


# OAD DATASET
def load_dataset():
    if not DATASET_DIR.exists():
        logger.error("Dataset not found")
        sys.exit(1)

    class_dirs = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()])

    X, y, labels = [], [], []
    feature_dim = None

    for idx, class_dir in enumerate(class_dirs):
        kp_file = class_dir / "keypoints.npy"

        if not kp_file.exists():
            continue

        keypoints = np.load(str(kp_file))

        if keypoints.ndim == 2:
            keypoints = keypoints.reshape(1, keypoints.shape[0], keypoints.shape[1])

        if keypoints.ndim != 3:
            continue

        n_frames = keypoints.shape[1]
        fdim = keypoints.shape[2]

        if feature_dim is None:
            feature_dim = fdim
            logger.info(f"FEATURE_DIM = {feature_dim}")

        if n_frames != SEQUENCE_LENGTH:
            continue

        labels.append(class_dir.name)

        for seq in keypoints:
            X.append(seq)
            y.append(len(labels) - 1)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    logger.info(f"Loaded {len(X)} samples, {len(labels)} classes")
    return X, y, labels, feature_dim


# ✅ AUGMENTATION
def augment_sequence(seq):
    aug = seq.copy()
    aug += np.random.normal(0, 0.01, aug.shape)

    if np.random.rand() < 0.3:
        drop_idx = np.random.choice(SEQUENCE_LENGTH, np.random.randint(1, 4), replace=False)
        aug[drop_idx] = 0

    if np.random.rand() < 0.2:
        aug *= np.random.uniform(0.9, 1.1)

    return aug


def augment_dataset(X, y):
    aug_X = [X]
    aug_y = [y]

    for _ in range(2):
        aug_X.append(np.array([augment_sequence(s) for s in X]))
        aug_y.append(y)

    return np.concatenate(aug_X), np.concatenate(aug_y)


# ✅ MAIN TRAINING
def main():
    has_gpu = configure_gpu()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    X, y, labels, feature_dim = load_dataset()
    num_classes = len(labels)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    X_train, y_train = augment_dataset(X_train, y_train)

    class_weights_arr = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights_arr))

    # ✅ MODEL BUILD
    if USE_MIXED_PRECISION and has_gpu:
        model = build_model_with_mixed_precision(num_classes, SEQUENCE_LENGTH, feature_dim)
    else:
        model = build_model(num_classes, SEQUENCE_LENGTH, feature_dim)

    model.summary()

    # FIXED CHECKPOINT (.h5)
    checkpoint_path = MODEL_DIR / "best_model.h5"

    cb_list = [
        callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=5),
    ]

    # TRAIN
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=cb_list,
    )

    # SAVE FINAL MODEL (VERY IMPORTANT)
    final_model_path = MODEL_DIR / "final_model.h5"
    model.save(str(final_model_path))

    # EVALUATE
    test_results = model.evaluate(X_test, y_test, return_dict=True)
    logger.info(f"Test metrics: {test_results}")
    logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")

    # SAVE LABELS
    (MODEL_DIR / "labels.txt").write_text("\n".join(labels))

    logger.info(f"Model saved at: {final_model_path}")
    print("model trained successfully")


if __name__ == "__main__":
    main()
