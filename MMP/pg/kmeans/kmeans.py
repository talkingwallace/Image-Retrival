from sklearn.cluster import KMeans,MeanShift,AgglomerativeClustering,Birch
from sklearn.decomposition import PCA

import numpy as np
import os
import matplotlib.pyplot as plt
import math
from PIL import Image

# FILENAME = './1000images.txt'
# PREDICT_FILENAME = './7images.txt'
FILENAME = './features.txt'
PREDICT_FILENAME = './features2.txt'
LABEL_NAME = ['beach','building','bus','dinasour','flower','horse','man']
IMAGEPATH = './image.orig'

def process_string(s):
    e = s.strip().split(' ')
    return e

def read(filename):
    content = []
    with open(filename,'r') as f:
        for e in f.readlines():
            s = process_string(e)
            content.append(s)
    content = np.array(content).astype('float32')
    return content

def statistic(classification,label):
    classification[label] += 1

def display():
    print('-'*30)
    print('\n0.beach\n1.building\n2.bus\n3.dinasour\n4.flower\n5.horse\n6.man')
    print('-'*30)

if __name__ == '__main__':
    display()
    selection = int(input('Please input the number: '))
    print('\nextracting from the file\n')

    features = read(FILENAME)
    predict_features = read(PREDICT_FILENAME)

    pca = PCA(n_components=0.99)
    pca.fit(features)
    features = pca.transform(features)
    predict_features = pca.transform(predict_features)

    model = Birch(n_clusters=10).fit(features)
    # model = KMeans(n_clusters=10,max_iter=1000).fit(features)
    # model = MeanShift(bandwidth=3).fit(features)
    # model = AgglomerativeClustering(n_clusters=10).fit(features)
    y_pred = model.labels_
    # ['beach', 'building', 'bus', 'dinasour', 'flower', 'horse', 'man']

    output_labels = model.predict(predict_features)
    # print(set(y_pred))
    classification = list()
    for i in range(10):
        classification.append(dict())
    for e in classification:
        for i in range(10):
            e.setdefault(i,0)
    for index,label in enumerate(y_pred):
        # 0 黑人
        if index >=0 and index <= 100:
            statistic(classification[0],label)
        # 1 沙滩
        elif index <= 199:
            statistic(classification[1], label)
        # 2 建筑
        elif index <= 299:
            statistic(classification[2], label)
        # 3 公交车
        elif index <= 399:
            statistic(classification[3], label)
        # 4 恐龙
        elif index <= 499:
            statistic(classification[4], label)
        # 5 大象
        elif index <= 599:
            statistic(classification[5], label)
        # 6 花
        elif index <= 699:
            statistic(classification[6], label)
        # 7 马
        elif index <= 799:
            statistic(classification[7], label)
        # 8 雪山
        elif index <= 899:
            statistic(classification[8], label)
        # 9 食物
        else:
            statistic(classification[9], label)

    data_label = output_labels[selection]
    images = list()
    for index,y in enumerate(y_pred):
        if y == data_label:
            images.append(os.path.join(IMAGEPATH,'%s.jpg' %index))

    image_nums = len(images)
    rows = int(np.ceil(math.sqrt(image_nums)))
    fig = plt.figure(figsize=(30, 30))
    for index,e in enumerate(images):
        n = index % rows
        ax = fig.add_subplot(rows, rows, index+1)
        picture = Image.open(e)
        picture = np.array(picture) / 255.
        ax.imshow(picture)

    for i in range(rows):
        for j in range(rows):
            idx = (n*i)+j

    plt.show()



    print(classification)
    print(output_labels)
    print()