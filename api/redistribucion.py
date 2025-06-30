import os
import shutil
import pandas as pd

# Ruta origen
base_dir = "data/ISIC2018/train"
images_dir = os.path.join(base_dir, "images")
labels_path = os.path.join(base_dir, "labels_binary.csv")

# Rutas destino
output_base = "data/isic2018Fed"
splits = {
    "train": 4000,
    "medico1": 2000,
    "medico2": 2000,
    "medico3": None  # El resto
}

# Cargar el CSV original
df = pd.read_csv(labels_path)
df = df[df["image"].map(lambda x: os.path.exists(os.path.join(images_dir, f"{x}.jpg")))]
total = len(df)

# Mezclar aleatoriamente
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Dividir
start = 0
split_dfs = {}

for name, count in splits.items():
    if count is None:
        end = total
    else:
        end = start + count
    split_dfs[name] = df.iloc[start:end].copy()
    start = end

# Copiar imágenes y guardar CSV por conjunto
for name, subset_df in split_dfs.items():
    img_dest = os.path.join(output_base, name, "images")
    os.makedirs(img_dest, exist_ok=True)

    # Copiar imágenes
    for img_name in subset_df["image"]:
        src_path = os.path.join(images_dir, f"{img_name}.jpg")
        dst_path = os.path.join(img_dest, f"{img_name}.jpg")
        shutil.copy2(src_path, dst_path)

    # Guardar CSV correspondiente
    labels_out = os.path.join(output_base, name, "labels_binary.csv")
    subset_df.to_csv(labels_out, index=False)

    # Resumen de clases
    print(f"\n📁 Conjunto '{name}'")
    print(subset_df["binary_label"].value_counts().sort_index().to_string())
    print(f"Total: {len(subset_df)} imágenes")

print("\n✅ División completada y etiquetas alineadas.")
