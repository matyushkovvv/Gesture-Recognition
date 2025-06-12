import pandas as pd

import shutil

# filename,width,height,class,xmin,ymin,xmax,ymax

train_or_val = input("Enter 'train' or 'val': ")

# load data
data = pd.read_csv("./raw_dataset/test/test/_annotations.csv")
data.dropna(inplace=True)


# Rock = 0
# Paper = 1
# Scissors = 2

# changes classes to numbers
data.loc[data['class'] == 'Rock', 'class'] = 0
data.loc[data['class'] == 'Paper', 'class'] = 1
data.loc[data['class'] == 'Scissors', 'class'] = 2

for index, row in data.iterrows():

    # create annotation file
    with open(f'./dataset/labels/{train_or_val}/{row["filename"]}.txt', 'w') as annotation_file:

        # calculate center of bounding box
        x_center = ((row['xmin'] + row['xmax']) / 2) / row['width']
        y_center = ((row['ymin'] + row['ymax']) / 2) / row['height']

        # calculate height and width of bounding box
        width = (row['xmax'] - row['xmin']) / row['width']
        height = (row['ymax'] - row['ymin']) / row['height']

        # write to annotation file
        annotation_file.write(f"{row['class']} {x_center} {y_center} {width} {height}")
    