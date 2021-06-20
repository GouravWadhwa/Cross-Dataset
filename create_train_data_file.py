import os
import random

mapping = {
    'dog' : 0,
    'elephant' : 1,
    'giraffe' : 2,
    'guitar' : 3,
    'horse' : 4,
    'house' : 5,
    'person' : 6
}

labelled_files = []
unlabelled_files = []

train_file = open ('train_file.txt', 'w+')
val_file = open ('val_file.txt', 'w+')

for root, dirs, files in os.walk ("PACS/") :
    for file in files :
        class_ = root.split ("/")[-1]
        domain = root.split ("/")[2]

        if domain == 'sketch' :
            unlabelled_files.append ("unlabelled " + os.path.join (root, file) + " " + str (mapping[class_]) + "\n")
        else :
            labelled_files.append ("labelled " + os.path.join (root, file) + " " + str (mapping[class_]) + "\n")

val_datapoints = random.sample (labelled_files, int (len (labelled_files) * 0.1))

for val_data in val_datapoints :
    val_file.write (val_data)

for data in labelled_files :
    if data not in val_datapoints :
        train_file.write (data)

for data in unlabelled_files :
    train_file.write (data)