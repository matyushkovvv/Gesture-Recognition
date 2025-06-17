import pandas as pd
import os
import shutil

# 📥 Ввод: train или val
train_or_val = input("Enter 'train' or 'val': ").strip()

# 📁 Пути
source_img_dir = "./raw_dataset/train/train"  # где лежат исходные изображения
output_img_dir = f"./dataset/images/{train_or_val}"
output_lbl_dir = f"./dataset/labels/{train_or_val}"

# 📌 Создаём папки, если их нет
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_lbl_dir, exist_ok=True)

# 📄 Чтение CSV
data = pd.read_csv("./raw_dataset/train/train/_annotations.csv")
data.dropna(inplace=True)

# 🔢 Замена классов на числа
label_map = {'Rock': 0, 'Paper': 1, 'Scissors': 2}
data['class'] = data['class'].map(label_map)

# 🧠 Группировка по изображению
grouped = data.groupby("filename")

# 📦 Индекс для переименования
for idx, (filename, group) in enumerate(grouped):

    # 📤 Новые имена
    new_img_name = f"{idx}.jpg"
    new_lbl_name = f"{idx}.txt"

    # 📁 Путь к исходному файлу
    src_path = os.path.join(source_img_dir, filename)
    dst_path = os.path.join(output_img_dir, new_img_name)

    # ✅ Копируем изображение
    shutil.copy2(src_path, dst_path)

    # 📝 Создаём .txt файл для YOLO
    with open(os.path.join(output_lbl_dir, new_lbl_name), 'w') as f:
        for _, row in group.iterrows():
            x_center = ((row['xmin'] + row['xmax']) / 2) / row['width']
            y_center = ((row['ymin'] + row['ymax']) / 2) / row['height']
            bbox_width = (row['xmax'] - row['xmin']) / row['width']
            bbox_height = (row['ymax'] - row['ymin']) / row['height']
            class_id = int(row['class'])

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
