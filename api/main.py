import csv
import datetime
import json
import os
import random
import string
import sys
import time
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from keras.utils import load_img, img_to_array
from pure_eval import Evaluator
from pydantic import BaseModel
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import backend as K, mixed_precision
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, BatchNormalization,
                                     GlobalAveragePooling2D, Dense, Dropout)
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Federated averaging utility


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.setrecursionlimit(1500)

# Mixed precision policy
# tf.config.experimental.enable_mlir_graph_optimization(False)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# GPU setup
physical_gpus = tf.config.list_physical_devices('GPU')
if physical_gpus:
    try:
        tf.config.experimental.set_memory_growth(physical_gpus[0], True)
    except RuntimeError:
        pass
    print("✅ GPU memory growth enabled.")
else:
    print("⚠️ No GPU detected, using CPU.")

# Clear any existing Keras session
K.clear_session()
print("🗑️ Keras session cleared.")

# =============================================================================
# APPLICATION INITIALIZATION
# =============================================================================

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


# =============================================================================
# DATA MODELS (Pydantic)
# =============================================================================

class SunExposure(BaseModel):
    alta: Optional[float] = None
    media: Optional[float] = None


class User(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    password: Optional[str] = None
    sunExposure: Optional[SunExposure] = None


class Credentials(BaseModel):
    id: str
    password: str


# =============================================================================
# DATABASE LOADING
# =============================================================================

USERS_PATH = os.path.join("json", "users.json")
with open(USERS_PATH, "r") as f:
    usersDB = {u['id']: {'name': u['name'], 'password': u['password'], 'sunExposure': u['sunExposure']}
               for u in json.load(f)['users']}

PATIENTS_PATH = os.path.join("json", "patients.json")
with open(PATIENTS_PATH, "r") as f:
    patientsDB = {p['id']: {'id': p['id'], 'name': p['name'], 'age': p['age'], 'gender': p['gender']}
                  for p in json.load(f)['patients']}


# =============================================================================
# CALLBACKS
# =============================================================================

class LrPrinter(Callback):
    """Prints the learning rate at the end of each epoch."""

    def on_epoch_end(self, epoch, logs=None):
        lr = float(self.model.optimizer.lr)
        print(f" — lr: {lr:.2e}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

globalThreshold = 0.70


def generateRandomId(length: int):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


def getPatient(pid: str):
    return patientsDB.get(pid)


def upload_image(user_id: str, array_or_file, folder: Optional[str]):
    base_dir = os.path.join("images", user_id, folder) if folder else os.path.join("images", user_id)
    os.makedirs(base_dir, exist_ok=True)
    filename = f"ISIC_{len(os.listdir(base_dir))}.jpg"
    path = os.path.join(base_dir, filename)
    if isinstance(array_or_file, np.ndarray):
        cv2.imwrite(path, array_or_file)
    else:
        array_or_file.save(path)
    return path


def getProcessedImage(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.6, heat, 0.4, 0)


def addSunExposureFactor(pred: float, exposure: str, uid: str) -> float:
    factor = usersDB[uid]['sunExposure']
    if exposure == 'Media':
        pred *= factor['media']
    elif exposure == 'Alta':
        pred *= factor['alta']
    return min(pred, 100.0)


# =============================================================================
# MODEL ARCHITECTURE & LOADING
# =============================================================================

def generateModelColor() -> Sequential:
    CNNModel = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(450, 600, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return CNNModel


def loadModel(uid: Optional[str] = None) -> Sequential:
    base_weights = "models/fed/melanoma_weights.h5"
    user_weights = f"checkpoints/users/{uid}/user.weights.h5" if uid else None
    if not os.path.exists(base_weights):
        raise FileNotFoundError(f"Base weights not found: {base_weights}")
    weights = user_weights if uid and os.path.exists(user_weights) else base_weights
    model = generateModelColor()
    model.load_weights(weights)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', Precision(name='prec'), Recall(name='rec'), AUC(name='auc')]
    )
    return model


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generateInform(user_id: str, patient: str, sunExposure: str, zone: str,
                   orig_path: str, proc_path: str, pred: float):
    inform_dir = os.path.join("json", "informs", user_id)
    os.makedirs(inform_dir, exist_ok=True)
    inf_file = os.path.join(inform_dir, "informs.json")
    informs = []
    if os.path.exists(inf_file):
        with open(inf_file) as f:
            informs = json.load(f).get('informs', [])
    p = getPatient(patient)
    informs.append({
        "id": generateRandomId(10),
        "Paciente": p['name'], "idPaciente": p['id'],
        "Edad": p['age'], "Sexo": p['gender'],
        "Zona": zone, "ExposicionSolar": sunExposure,
        "ImagenOriginal": orig_path, "ImagenProcesada": proc_path,
        "Fecha": datetime.datetime.now().strftime("%d/%m/%Y"),
        "Prediccion": pred
    })
    with open(inf_file, 'w') as f:
        json.dump({"informs": informs}, f, indent=4)


# =============================================================================
# TRAINING & EVALUATION UTILITIES
# =============================================================================

def plot_loss_curves(
        history: dict,
        save_dir: str,
        filename: str = "loss_plot.png",
        csv_name: str = "loss_data.csv"
):
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    import numpy as np

    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    epochs_range = list(range(1, len(loss) + 1))

    df_loss = pd.DataFrame({
        "epoch": epochs_range,
        "train_loss": loss,
        "val_loss": val_loss
    })
    df_loss.to_csv(os.path.join(save_dir, csv_name), index=False)

    plt.figure(figsize=(12, 4))
    plt.plot(epochs_range, loss, "-o", label="Train Loss")
    plt.plot(epochs_range, val_loss, "-o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0, 1.25)


    if len(epochs_range) <= 15:
        step = 1
    elif len(epochs_range) <= 40:
        step = 2
    else:
        step = 5

    max_epoch = epochs_range[-1]
    plt.xlim(0, max_epoch + 1)
    plt.xticks(np.arange(0, max_epoch + 2, step))

    plt.legend()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

def plot_loss_from_csv(
        csv_path: str,
        save_dir: str,
        filename: str = "loss_plot.png"
):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os

    df = pd.read_csv(csv_path)
    epochs = df["epoch"]
    loss = df["train_loss"]
    val_loss = df["val_loss"]

    plt.figure(figsize=(12, 4))
    plt.plot(epochs, loss, "-o", label="Train Loss")
    plt.plot(epochs, val_loss, "-o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0, 1.25)

    if len(epochs) <= 15:
        step = 1
    elif len(epochs) <= 40:
        step = 2
    else:
        step = 5

    max_epoch = epochs.iloc[-1]
    plt.xlim(0, max_epoch + 1)
    plt.xticks(np.arange(0, max_epoch + 2, step))

    plt.legend()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


def binarize_labels(labels_csv_path: str, output_csv_path: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(labels_csv_path)
    df['binary_label'] = df['MEL'].apply(lambda x: 1 if x == 1.0 else 0)
    result = df[['image', 'binary_label']].copy()
    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        result.to_csv(output_csv_path, index=False)
    return result


def create_and_train_base_model(
        labels_csv: str,
        images_dir: str,
        output_model_path: str,
        output_weights_path: str,
        img_size: tuple[int, int] = (450, 600),
        batch_size: int = 32,
        epochs: int = 50,
        validation_split: float = 0.2
):
    start_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join("evaluations", "trains", timestamp)
    os.makedirs(eval_dir, exist_ok=True)

    df = pd.read_csv(labels_csv, dtype={"binary_label": int})
    df["filepath"] = df["image"].map(lambda fn: os.path.join(images_dir, fn + ".jpg"))
    df = df[df["filepath"].map(os.path.exists)]

    train_df, val_df = train_test_split(
        df,
        test_size=validation_split,
        stratify=df["binary_label"],
        random_state=42
    )

    def stats(sub, name):
        tot = len(sub)
        pos = sub["binary_label"].sum()
        neg = tot - pos
        print(f"{name}: total={tot}, 1={pos} ({pos / tot * 100:.1f}%), 0={neg} ({neg / tot * 100:.1f}%)")

    stats(train_df, "🏋️‍♂️ Train set")
    stats(val_df, "🔬 Val set")

    y_train = train_df["binary_label"]
    classes = np.unique(y_train)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = dict(zip(classes, cw))
    print("class_weight:", class_weight)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_df["binary_label"] = train_df["binary_label"].astype(str)
    val_df["binary_label"] = val_df["binary_label"].astype(str)

    train_gen = train_datagen.flow_from_dataframe(
        train_df, x_col="filepath", y_col="binary_label",
        target_size=img_size, batch_size=batch_size, class_mode="binary"
    )
    val_gen = val_datagen.flow_from_dataframe(
        val_df, x_col="filepath", y_col="binary_label",
        target_size=img_size, batch_size=batch_size, class_mode="binary"
    )

    x0, y0 = next(train_gen)
    print(f" → x0 range: {x0.min():.2e} – {x0.max():.2e}")
    print(f" → y0 unique: {np.unique(y0)} dtype={y0.dtype}")

    CNNModel = generateModelColor()
    CNNModel.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", Precision(name="prec"), Recall(name="rec"), AUC(name="auc")]
    )

    cbs = [
        EarlyStopping(monitor="val_auc", mode="max", patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.3, patience=3, min_lr=1e-6),
        LrPrinter()
    ]

    hist_start = time.time()
    history = CNNModel.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        class_weight=class_weight,
        callbacks=cbs
    )
    hist_end = time.time()

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    CNNModel.save(output_model_path)
    print(f"✅ Modelo completo guardado en {output_model_path}")
    CNNModel.save_weights(output_weights_path)
    print(f"✅ Pesos base guardados en {output_weights_path}")

    h = history.history
    plot_loss_curves(h, save_dir=eval_dir)

    y_true, y_pred = [], []
    val_gen.reset()
    for _ in range(len(val_gen)):
        x_batch, y_batch = val_gen.next()
        y_true.extend(y_batch.astype(int))
        y_pred.extend(CNNModel.predict(x_batch).ravel())
    prec, rec, th = precision_recall_curve(y_true, y_pred)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    best_thr = th[np.argmax(f1)]

    total_time = time.time() - start_time
    avg_epoch_time = (hist_end - hist_start) / len(h["loss"])

    with open(os.path.join(eval_dir, "metrics_report.txt"), "w") as f:
        f.write(f"Final Metrics:\n")
        f.write(f"  Accuracy:  {h['accuracy'][-1]:.4f}\n")
        f.write(f"  Precision: {h['prec'][-1]:.4f}\n")
        f.write(f"  Recall:    {h['rec'][-1]:.4f}\n")
        f.write(f"  AUC:       {h['auc'][-1]:.4f}\n")
        f.write(f"\nOptimal Threshold (F1-max): {best_thr:.3f}\n")
        f.write(f"\nTraining time summary:\n")
        f.write(f"  Total time: {total_time:.2f} seconds\n")
        f.write(f"  Avg time/epoch: {avg_epoch_time:.2f} seconds\n")

    print(f"🔍 Optimal threshold (F1-max): {best_thr:.3f}")
    print(f"✅ Report saved in {eval_dir}")

    return history


def averageWeightsModel():
    w_files, users = getAllWeightsFiles()
    if not w_files:
        return

    avg = loadModel()
    base_weights = avg.get_weights()
    weights_sum = [np.zeros_like(w) for w in base_weights]

    sample_counts = []
    total_samples = 0
    for user in users:
        csv_path = os.path.join("labels", user, "labels_binary.csv")
        if os.path.exists(csv_path):
            try:
                n = len(pd.read_csv(csv_path))
            except Exception:
                n = 1
        else:
            n = 1
        sample_counts.append(n)
        total_samples += n

    if total_samples == 0:
        sample_counts = [1] * len(w_files)
        total_samples = len(w_files)

    for wf, n in zip(w_files, sample_counts):
        m = generateModelColor()
        m.load_weights(wf)
        client_ws = m.get_weights()
        for i, w in enumerate(client_ws):
            weights_sum[i] += w * n

    new_weights = [ws / total_samples for ws in weights_sum]
    avg.set_weights(new_weights)

    os.makedirs("checkpoints/average", exist_ok=True)
    avg.save_weights("checkpoints/average/average.weights.h5")
    create_average_model()


def getAllWeightsFiles():
    base = "checkpoints/users"
    files, users = [], []
    if os.path.isdir(base):
        for u in os.listdir(base):
            for f in os.listdir(os.path.join(base, u)):
                if f.endswith("weights.h5"):
                    files.append(os.path.join(base, u, f));
                    users.append(u)
    return files, users


def create_average_model():
    """
    Creates a model with the architecture defined in generateModelColor,
    loads the averaged weights, compiles it, and saves it as a complete .h5 model.
    """
    try:
        avg_model = generateModelColor()
        avg_weights_path = os.path.join("checkpoints", "average", "average.weights.h5")
        if not os.path.exists(avg_weights_path):
            print("❌ Averaged weights not found. Please run federatedLearning first.")
            return

        avg_model.load_weights(avg_weights_path)
        avg_model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', Precision(name='prec'), Recall(name='rec'), AUC(name='auc')]
        )

        output_path = os.path.join("models", "average", "melanoma_average.h5")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        avg_model.save(output_path)
        print(f"✅ Federated model successfully saved at {output_path}")

    except Exception as e:
        print(f"❗ Error creating federated model: {e}")


def evaluate_model_from_path(
        model_path: str,
        images_dir: str,
        labels_csv_path: str,
        img_size: tuple[int, int] = (450, 600),
        threshold: float = 0.7
):
    """
    Evaluates a Keras model loaded from a path on a validation set.
    Saves metrics, ROC and PR curves, confusion matrix, and all the required
    data to build it in a CSV file.

    Args:
        model_path: Path to the model (.h5 or .keras).
        images_dir: Directory containing images.
        labels_csv_path: CSV file with columns 'image' and 'binary_label'.
        img_size: Input size of the model.
        threshold: Decision threshold for classification.
    """

    print(f"📦 Loading model from {model_path} ...")
    model = load_model(model_path)

    df = pd.read_csv(labels_csv_path)
    df = df[df["image"].map(lambda x: os.path.exists(os.path.join(images_dir, x + ".jpg")))]

    y_true, y_pred, y_scores = [], [], []
    image_names = []

    for _, row in df.iterrows():
        img_path = os.path.join(images_dir, row["image"] + ".jpg")
        img = load_img(img_path, target_size=img_size)
        arr = img_to_array(img) / 255.0
        pred_score = model.predict(np.expand_dims(arr, axis=0))[0][0]
        label = int(row["binary_label"])

        y_true.append(label)
        y_scores.append(pred_score)
        y_pred.append(1 if pred_score >= threshold else 0)
        image_names.append(row["image"])

    # Save confusion matrix data
    results_df = pd.DataFrame({
        "image": image_names,
        "true_label": y_true,
        "predicted_label": y_pred,
        "score": y_scores
    })

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join("evaluations", timestamp)
    os.makedirs(base_dir, exist_ok=True)

    results_df.to_csv(os.path.join(base_dir, "confusion_data.csv"), index=False)
    print(f"📝 Confusion matrix data saved at {os.path.join(base_dir, 'confusion_data.csv')}")

    report = classification_report(y_true, y_pred, output_dict=True)
    report["roc_auc"] = auc(*roc_curve(y_true, y_scores)[:2])

    with open(os.path.join(base_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=4)
    print(f"✅ Metrics saved at {os.path.join(base_dir, 'metrics.json')}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.plot(fpr, tpr, label=f"AUC = {report['roc_auc']:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(base_dir, "roc_curve.png"))
    plt.clf()

    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.savefig(os.path.join(base_dir, "precision_recall.png"))
    plt.clf()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No MEL", "MEL"], yticklabels=["No MEL", "MEL"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(base_dir, "confusion_matrix.png"))
    plt.close()

    print(f"📊 Charts saved at {base_dir}")



def compare_models_weights(path1: str, path2: str):
    print(f"🔍 Comparing models:\n - Model 1: {path1}\n - Model 2: {path2}")

    if not os.path.exists(path1):
        print(f"❌ Model 1 not found: {path1}")
        return
    if not os.path.exists(path2):
        print(f"❌ Model 2 not found: {path2}")
        return

    try:
        model1 = load_model(path1)
        model2 = load_model(path2)
    except Exception as e:
        print(f"⚠️ Error loading models: {e}")
        return

    weights1 = model1.get_weights()
    weights2 = model2.get_weights()

    if len(weights1) != len(weights2):
        print("❗️ The models have a different number of layers or weights.")
        return

    diffs = []
    for i, (w1, w2) in enumerate(zip(weights1, weights2)):
        diff = np.mean(np.abs(w1 - w2))
        diffs.append(diff)

    total_diff = np.sum(diffs)
    max_diff = np.max(diffs)
    print(f"\n📊 Mean difference per layer: {['%.6f' % d for d in diffs]}")
    print(f"🔬 Total difference: {total_diff:.6f}")
    print(f"📈 Maximum difference per layer: {max_diff:.6f}")

    if total_diff == 0.0:
        print("✅ The models have *identical* weights.")
    elif total_diff < 1e-3:
        print("⚠️ The models are *very similar*, possible minimal changes.")
    else:
        print("✅ The models are *different*, training had an effect.")


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get('/')
def index():
    return {"status": "running"}


@app.post('/login/')
async def login(creds: Credentials):
    user = usersDB.get(creds.id)
    if not user or user['password'] != creds.password:
        raise HTTPException(401, "Incorrect credentials")
    loadModel(creds.id)
    return {'id': creds.id, 'name': user['name'], 'sunExposure': user['sunExposure']}



@app.get('/patients/')
async def get_patients():
    return patientsDB


@app.post('/predictMelanom/{user_id}/{patient}/{zone}/{sunExposure}')
async def predict_melanom(user_id: str, patient: str, sunExposure: str, zone: str,
                          image: UploadFile = File(...)):
    raw = await image.read()
    img = tf.image.decode_image(raw, channels=3).numpy()
    img = cv2.resize(img, (600, 450))
    img_norm = img.astype('float32') / 255.0
    model = loadModel(user_id)
    prob = model.predict(np.expand_dims(img_norm, 0), verbose=0)[0][0]
    prediction = addSunExposureFactor(float(prob * 100), sunExposure, user_id)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    image_name = f"{timestamp}_{user_id}_{patient}"
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Save original and labels
    orig_dir = os.path.join('images', user_id)
    os.makedirs(orig_dir, exist_ok=True)
    orig_path = os.path.join(orig_dir, f"{image_name}.jpg")
    cv2.imwrite(orig_path, img_bgr)

    labels_dir = os.path.join('labels', user_id)
    os.makedirs(labels_dir, exist_ok=True)
    csv_path = os.path.join(labels_dir, 'labels_binary.csv')
    label = '1' if prediction >= 70 else '0'
    mode = 'a' if os.path.exists(csv_path) else 'w'
    with open(csv_path, mode, newline='') as f:
        writer = csv.writer(f)
        if mode == 'w': writer.writerow(['image', 'binary_label'])
        writer.writerow([image_name, label])

    proc_img = getProcessedImage(img)
    proc_path = upload_image(patient, cv2.cvtColor(proc_img, cv2.COLOR_RGB2BGR), 'processed')
    upload_image(user_id, img_bgr, None)

    generateInform(user_id, patient, sunExposure, zone, orig_path, proc_path, prediction)
    return JSONResponse({'prediction': {'result': round(prediction, 2)}, 'image': orig_path})


@app.post("/retrain_model/{user_id}")
async def retrain_model(user_id: str):
    import time, datetime, os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.metrics import Precision, Recall, AUC

    images_dir = os.path.join("images", user_id)
    labels_csv = os.path.join("labels", user_id, "labels_binary.csv")

    df = pd.read_csv(labels_csv, dtype={"binary_label": int})
    df["filepath"] = df["image"].map(lambda fn: os.path.join(images_dir, fn + ".jpg"))
    df = df[df["filepath"].map(os.path.exists)]

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["binary_label"],
        random_state=42
    )

    def stats(sub, name):
        tot = len(sub)
        pos = sub["binary_label"].sum()
        neg = tot - pos
        print(f"{name}: total={tot}, 1={pos} ({pos / tot * 100:.1f}%), 0={neg} ({neg / tot * 100:.1f}%)")

    stats(train_df, "🏋️‍♂️ Train set")
    stats(val_df, "🔬 Val set")

    y_train = train_df["binary_label"]
    classes = np.unique(y_train)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = dict(zip(classes, cw))
    print("class_weight:", class_weight)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_df["binary_label"] = train_df["binary_label"].astype(str)
    val_df["binary_label"] = val_df["binary_label"].astype(str)

    BATCH = 8
    train_gen = train_datagen.flow_from_dataframe(
        train_df, x_col="filepath", y_col="binary_label",
        target_size=(450, 600), batch_size=BATCH, class_mode="binary"
    )
    val_gen = val_datagen.flow_from_dataframe(
        val_df, x_col="filepath", y_col="binary_label",
        target_size=(450, 600), batch_size=BATCH, class_mode="binary", shuffle=False
    )

    model = loadModel(user_id)
    for layer in model.layers[:-3]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', Precision(name='prec'), Recall(name='rec'), AUC(name='auc')]
    )

    ckpt_dir = os.path.join("checkpoints", "users", user_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join("evaluations", "retrain", timestamp + f"_{user_id}")
    os.makedirs(eval_dir, exist_ok=True)

    early = EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True)
    red_lr = ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=3, min_lr=1e-6)

    print("🚀 Training model...")
    start_time = time.time()
    history = model.fit(
        train_gen,
        epochs=20,
        validation_data=val_gen,
        callbacks=[early, red_lr],
        class_weight=class_weight
    )
    end_time = time.time()

    model.save_weights(os.path.join(ckpt_dir, "user.weights.h5"))

    # 📊 Métricas
    h = history.history
    loss, val_loss = h["loss"], h["val_loss"]
    acc, val_acc = h["accuracy"], h["val_accuracy"]
    prec, rec, auc = h["prec"], h["rec"], h["auc"]

    # 📈 Gráfico de pérdida
    plot_loss_curves(h, save_dir=eval_dir)

    # 🔍 Umbral óptimo
    y_true, y_pred = [], []
    val_gen.reset()
    for _ in range(len(val_gen)):
        x_batch, y_batch = val_gen.next()
        y_true.extend(y_batch.astype(int))
        y_pred.extend(model.predict(x_batch).ravel())
    from sklearn.metrics import precision_recall_curve
    p, r, t = precision_recall_curve(y_true, y_pred)
    f1 = 2 * p * r / (p + r + 1e-8)
    best_thr = t[np.argmax(f1)]

    # 📝 Guardar informe
    total_time = end_time - start_time
    avg_epoch_time = total_time / len(loss)
    with open(os.path.join(eval_dir, "metrics_report.txt"), "w") as f:
        f.write("Final Metrics:\n")
        f.write(f"  Accuracy:  {acc[-1]:.4f}\n")
        f.write(f"  Precision: {prec[-1]:.4f}\n")
        f.write(f"  Recall:    {rec[-1]:.4f}\n")
        f.write(f"  AUC:       {auc[-1]:.4f}\n")
        f.write(f"\nOptimal Threshold (F1-max): {best_thr:.3f}\n")
        f.write(f"\nTraining time summary:\n")
        f.write(f"  Total time: {total_time:.2f} seconds\n")
        f.write(f"  Avg time/epoch: {avg_epoch_time:.2f} seconds\n")

    print(f"✅ Report and plot saved in {eval_dir}")
    return {
        "result": float(acc[-1]),
        "best_threshold": float(best_thr)
    }


def model_from_user_weights_model(
        user_id: str
):
    """
    Evaluates a set of custom weights using the base architecture.

    Args:
        weights_path: Path to the user's .h5 weights file.
    """

    user_model_path =  os.path.join("checkpoints", "users", user_id, "user.weights.h5")

    if not os.path.exists(user_model_path):
        print(f"❌ Weights not found at: {user_model_path}")
        return

    # 1. Create base architecture
    model = generateModelColor()

    # 2. Load custom weights
    try:
        model.load_weights(user_model_path)
    except Exception as e:
        print(f"⚠️ Error loading weights: {e}")
        return

    # 3. Compile model
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', Precision(name='prec'), Recall(name='rec'), AUC(name='auc')]
    )

    # 4. Save complete model
    model_path = os.path.join("models", "fed", f"melanoma_model_{user_id}.h5")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)



@app.get('/informs/{user_id}')
async def get_informs(user_id: str):
    path = os.path.join('json', 'informs', user_id, 'informs.json')
    if not os.path.exists(path):
        raise HTTPException(404, 'Report not found')
    with open(path) as f:
        return json.load(f)


@app.get('/images/{user_id}/{image:path}')
async def get_inform_image(user_id: str, image: str = Path(..., description="Image path")):
    path = os.path.join('images', user_id, image)
    if not os.path.exists(path):
        raise HTTPException(404, 'Image not found')
    return FileResponse(path)

@app.post('/federatedLearning')
async def federated_learning():
    try:
        averageWeightsModel()
    except Exception as e:
        print('FL error:', e)

    return {'status': 'ok', 'message': 'Federated averaging complete'}


@app.put('/users/settings/{user_id}')
def update_user_settings(user_id: str, user: User):
    if user_id not in usersDB:
        raise HTTPException(404, 'User not found')
    if user.sunExposure:
        usersDB[user_id]['sunExposure'] = user.sunExposure.dict()
    with open(USERS_PATH, 'w') as f:
        json.dump({'users': [{'id': uid, **data} for uid, data in usersDB.items()]}, f, indent=4)
    return {'message': 'Solar exposure settings updated successfully'}


#if __name__ == '__main__':

# =============================================================================
# Here you can train models, evaluate...