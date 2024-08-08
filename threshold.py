import os
import numpy as np

def make_buckets(filepath='/home/aarush/CircNet/congestion/data/base_maps/label', destpath='/home/aarush/CircNet/congestion/data/2buckets/label', thresholds=[0.2]):
    my_labels = sorted(os.listdir(filepath))
    for i in my_labels:
        label = np.load(os.path.join(filepath, i))
        label[label<=thresholds[0]]=0
        label[label>thresholds[0]]=1
        np.save(os.path.join(destpath, i), label)
        
