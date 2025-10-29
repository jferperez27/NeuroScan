import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L, Model, optimizers as opt
import nibabel as nib
import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

import tensorflow as tf
tf.keras.backend.clear_session()
try: del model
except: pass


# all training files
basedir = pathlib.Path(__file__).resolve().parent.parent
training_data_dir = basedir / "preprocessing" / "training_data"
training_files = sorted(str(p) for p in training_data_dir.glob("*.nii.gz"))

training_key_csv = training_data_dir / "training_data_key.csv"

BATCH_SIZE = 2  # Increased from 1 for more stable gradients
EPOCHS = 2

SHAPE = (128, 128, 128)  # current shape for lightweight VRAM usage
df = pd.read_csv(training_key_csv)
classes = sorted(df["condition"].unique())
class_to_id = {c:i for i,c in enumerate(classes)}
df["y"] = df["condition"].map(class_to_id).astype(np.int32)

filenames = df["file_name"].tolist()
labels    = df["y"].to_numpy()

file_paths = [training_data_dir / f for f in filenames]
files_paths = [str(file) for file in file_paths]
#print(file_paths)

fp_tr, fp_va, y_tr, y_va = train_test_split(
    files_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

def load_to_tensor(file_path):
    """
    Load a NIfTI image from file_path and convert it to a TensorFlow tensor.
    """
    img = nib.load(file_path)
    dim = img.get_fdata(dtype=np.float32)
    nz = dim != 0

    if np.any(nz):
        m, s = dim[nz].mean(), dim[nz].std()
        if s > 1e-6:
            dim[nz] = (dim[nz] - m) / s
    
    dim = np.expand_dims(dim, axis=-1)  # add tensor dimension
    assert dim.shape[:3] == SHAPE, f"Got shape {dim.shape}"
    return dim


def py_load(path):
    if isinstance(path, np.ndarray):
        path = path.item()  # unwrap numpy scalar
    if isinstance(path, bytes):
        path = path.decode("utf-8")
    elif isinstance(path, Path):
        path = str(path)

    vol4d = load_to_tensor(path)   # returns (D,H,W,1)
    if vol4d.ndim == 4 and vol4d.shape[-1] == 1:
        vol3d = vol4d[..., 0]      # -> (D,H,W)
    elif vol4d.ndim == 3:
        vol3d = vol4d
    else:
        raise ValueError(f"Unexpected shape from load_to_tensor: {vol4d.shape}")

    if vol3d.dtype != np.float32:
        vol3d = vol3d.astype(np.float32, copy=False)
    if not np.isfinite(vol3d).all():
        raise ValueError("NaN/Inf in volume")
    assert vol3d.shape == SHAPE, (vol3d.shape, SHAPE)
    return vol3d

def tf_load(path, label):
    # numpy_function -> (D,H,W) float32
    tensor = tf.numpy_function(py_load, [path], tf.float32)

    # 1) fix the static 3D shape
    tensor.set_shape(SHAPE)  # e.g., (128,128,128)

    # 2) add channel dim -> (D,H,W,1)
    tensor = tf.expand_dims(tensor, -1)

    # (optional) assert after expand if you want:
    # tensor.set_shape(SHAPE + (1,))

    # label -> int32 scalar for classification
    label = tf.cast(label, tf.int32)
    # label should be a scalar, not shape (1,)
    label = tf.squeeze(label)

    return tensor, label

# --- normalize inputs for tf.data ---
# file_paths: list[Path] -> list[str]
file_paths_str = [str(p) for p in file_paths]
# labels -> float32 array (or int32 if you prefer)
labels_arr = np.asarray(labels, dtype=np.float32)

dataset = tf.data.Dataset.from_tensor_slices((file_paths_str, labels_arr))

# Create training and validation datasets
train_ds = tf.data.Dataset.from_tensor_slices((fp_tr, y_tr))
train_ds = train_ds.map(tf_load, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(buffer_size=len(fp_tr))
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((fp_va, y_va))
val_ds = val_ds.map(tf_load, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# Number of classes for classification
num_classes = len(classes)
print(f"Number of classes: {num_classes}")
print(f"Classes: {classes}")
print(f"Training samples: {len(fp_tr)}, Validation samples: {len(fp_va)}")

# Build a simple 3D CNN model
def build_3d_cnn(input_shape=(128, 128, 128, 1), num_classes=2):
    inputs = keras.Input(shape=input_shape)
    
    # First conv block
    x = L.Conv3D(32, kernel_size=3, activation='relu', padding='same')(inputs)
    x = L.MaxPooling3D(pool_size=2)(x)
    x = L.BatchNormalization()(x)
    
    # Second conv block
    x = L.Conv3D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = L.MaxPooling3D(pool_size=2)(x)
    x = L.BatchNormalization()(x)
    
    # Third conv block
    x = L.Conv3D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = L.MaxPooling3D(pool_size=2)(x)
    x = L.BatchNormalization()(x)
    
    # Global pooling and dense layers
    x = L.GlobalAveragePooling3D()(x)
    x = L.Dropout(0.5)(x)
    x = L.Dense(128, activation='relu')(x)
    x = L.Dropout(0.3)(x)
    
    # Output layer
    if num_classes == 2:
        outputs = L.Dense(1, activation='sigmoid')(x)
    else:
        outputs = L.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Create and compile model
model = build_3d_cnn(input_shape=SHAPE + (1,), num_classes=num_classes)

# Choose appropriate loss and metrics based on number of classes
if num_classes == 2:
    loss = 'binary_crossentropy'
    metrics = ['accuracy', keras.metrics.AUC(name='auc')]
else:
    loss = 'sparse_categorical_crossentropy'
    metrics = ['accuracy']

model.compile(
    optimizer=opt.Adam(learning_rate=1e-4),  # Lower learning rate for stability
    loss=loss,
    metrics=metrics
)

model.summary()

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1
)

# Save the model
model.save(basedir / "train" / "trained_model.h5")
print("Model saved to trained_model.h5")
