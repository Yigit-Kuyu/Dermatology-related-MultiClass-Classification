import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import glob
import PIL
import pathlib
import splitfolders
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras import regularizers
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
import Augmentor
import os
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import *



class_names=["Acne", "Atopic Dermatitis", "Bullous Dieases", "cancer", "other"]
path_to_all_dataset=""


'''

## Convert whole images to RGB format 
from PIL import Image
import os

for i in class_names:
    folder_path = os.path.join(path_to_all_dataset, i)
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        try:
            img = Image.open(img_path)
            img = img.convert("RGB")  # Convert to RGB format if not already
            img.save(img_path)  # Overwrite with the correct format
        except Exception as e:
            print(f"Error processing {img_path}: {e}")



path_to_training_dataset="/Test_Val_Data/train/"
path_to_val_dataset="/Test_Val_Data/val/"


## Convert whole image names to suitable names
folder_path = path_to_training_dataset 
# Adjust this path to point to the directory that contains your class folders
base_path = folder_path
for class_name in class_names:
    # Replace spaces with underscores to make file names cleaner
    class_name_underscore = class_name.replace(" ", "_")
    
    # Build the folder path for the images of this class
    folder_path = os.path.join(base_path, class_name)
    
    # Make sure the folder exists
    if not os.path.isdir(folder_path):
        print(f"Warning: {folder_path} does not exist!")
        continue
    
    counter = 1
    for filename in os.listdir(folder_path):
        old_path = os.path.join(folder_path, filename)
        
        # Only rename if it's a file (not a directory)
        if os.path.isfile(old_path):
            # Get the file extension (.jpg, .png, etc.)
            _, ext = os.path.splitext(filename)
            
            # Construct the new filename: e.g., "Acne_1.jpg"
            new_filename = f"{class_name_underscore}_{counter}{ext}"
            new_path = os.path.join(folder_path, new_filename)
            
            # Rename the file
            os.rename(old_path, new_path)
            
            counter += 1
    
    print(f"Renamed files in {class_name} -> format: {class_name_underscore}_#.ext")

'''


'''
# Creating training and test data based on test folder
path_to_train_val_dataset="/Test_Val_Data/"
splitfolders.ratio(path_to_all_dataset, output=path_to_train_val_dataset, ratio=(.8, .2))
'''


path_to_training_dataset="/Test_Val_Data/train/"
path_to_val_dataset="/Test_Val_Data/val/"

data_dir_train = pathlib.Path(path_to_training_dataset)
img_height,  img_width  = 224, 224

train_ds = image_dataset_from_directory(data_dir_train,
                                        seed = 123,
                                        image_size=(img_height, img_width))

class_names = train_ds.class_names

print(class_names)

'''
# Ploting sample
plt.figure(figsize=(15, 10))
for i, class_ in enumerate(list(class_names)):
    plt.subplot(3, 3, i+1)
    data_path = os.path.join(str(data_dir_train), class_)
    file_path = glob.glob(os.path.join(data_path,'*.*'))[0]
    img = PIL.Image.open(file_path)
    plt.imshow(img)
    plt.title(class_)
    plt.axis("off")
plt.show()
'''


class_size = {}
for name in class_names:
    class_size[name] = len(list(data_dir_train.glob(name+'/*.*')))

print(class_size)

class_df = pd.DataFrame(class_size.items(),index=list(class_size), columns = ['ClassName', 'NumberOfSamples'])
class_df.drop(['ClassName'], axis = 1, inplace=True)
print(class_df)


'''
# Removing corrupted images 
from PIL import Image
import os

def remove_corrupted_images(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                try:
                    img = Image.open(os.path.join(root, file))
                    img.close()
                except OSError as e:
                    print(f"Removing corrupted image: {os.path.join(root, file)}")
                    os.remove(os.path.join(root, file))

remove_corrupted_images(path_to_val_dataset)
remove_corrupted_images(path_to_training_dataset)

# Printing image extensions
def list_file_extensions(path_to_val_dataset):
    extensions = {}
    for root, dirs, files in os.walk(path_to_val_dataset):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in extensions:
                extensions[ext] = 0
            extensions[ext] += 1
    print(extensions)

list_file_extensions(path_to_val_dataset)
list_file_extensions(path_to_training_dataset)
'''


'''
# offline augmentation
# Increase 500 samples for each class in train folder 
for i in class_names:
    p = Augmentor.Pipeline(path_to_training_dataset + i,output_directory='')
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.sample(500)
'''



# Collect image paths and labels, and filter valid images
def create_dataframe_from_directory(directory, class_names, target_size=(224, 224)):
    data = []
    print(f"Scanning directory: {directory}")
    for root, dirs, files in os.walk(directory):
        subfolder_name = os.path.basename(root)
        # Only process subfolders that match class_names
        if subfolder_name in class_names:
            print(f"Processing class: {subfolder_name}")
            for file in files:
                # Check for image file extensions
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    try:
                        # Verify image integrity
                        img = Image.open(file_path)
                        img.verify()
                        img.close()
                        # Reopen to check resizing (optional, ensures compatibility)
                        img = Image.open(file_path)
                        img.resize(target_size)
                        img.close()
                        # Add valid image path and class label to data
                        data.append({"filename": file_path, "class": subfolder_name})
                    except Exception as e:
                        print(f"Skipping invalid image: {file_path}, Error: {e}")
    if not data:
        print(f"Warning: No valid images found in {directory}")
    return pd.DataFrame(data)

train_dir= path_to_training_dataset
val_dir= path_to_val_dataset
df_train  = create_dataframe_from_directory(train_dir, class_names)
df_val  = create_dataframe_from_directory(val_dir, class_names)

# Verify the DataFrames
print("Training DataFrame:")
print(df_train.head())
print("Validation DataFrame:")
print(df_train.head())


batch_size = 32
epochs = 500
img_size = 224

# Online augmentation
# Does not create additional image files on disk. 
# Instead, it applies random transformations (like rotation, zoom, flips, etc.) to images each time they are loaded in a training batch. 
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    width_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.2, 0.8]
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    width_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.2, 0.8]
)

test_datagen_no_aug = ImageDataGenerator(rescale=1./255) # test_datagen without augmentation


# Optional: Use "flow_from_directory" to automatically labels each image based on its subfolder name.
training_set = train_datagen.flow_from_dataframe(
                                                df_train,
                                                target_size=(img_size, img_size), # 224X224
                                                batch_size=32,
                                                class_mode='categorical'  # for multi-class
                                                )
val_set = val_datagen.flow_from_dataframe(      df_val,
                                                target_size=(img_size, img_size), # 224X224
                                                batch_size=32,
                                                class_mode='categorical'  # for multi-class
                                                )
test_set = test_datagen_no_aug.flow_from_dataframe(
                                                df_val,
                                                target_size=(img_size, img_size), # 224X224
                                                batch_size=32,
                                                class_mode='categorical'  # for multi-class
                                                )


# "loss" is CategoricalCrossentropy defined in "model.compile"
# If the training loss doesn’t decrease (improve) for 3 consecutive epochs (patience=5), training stops.
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Reduces the learning rate according to monitored metric (val_loss)
# If val_loss does not improve for 3 epochs, new_learning_rate = current_learning_rate * factor
reduce_lr =  keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, min_lr=0.00001)


saved_checkpoints= '/best_checkpoint_model.h5'
checkpoint = ModelCheckpoint(saved_checkpoints, save_best_only= True, monitor = 'val_loss')



def Xception():

    engine = tf.keras.applications.Xception(
        # Freezing the weights of the top layer in the InceptionResNetV2 pre-traiined model
        include_top = False,
        # Use Imagenet weights
        weights = 'imagenet',
        # Define input shape to 224x224x3
        input_shape = (img_size , img_size , 3),

    )


    x = tf.keras.layers.GlobalAveragePooling2D(name = 'avg_pool')(engine.output)
    x =Dropout(0.75)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Dense(5, activation = 'softmax', name = 'dense_output')(x)

    # Build the Keras model
    model = tf.keras.models.Model(inputs = engine.input, outputs = out)
    # Compile the model
    model.compile(
        
        optimizer = tf.keras.optimizers.Adam(learning_rate= 1e-4),
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics = ['accuracy'] # Set metrics to accuracy
    )

    return model


def train():
    time_start = time.time()

    model = Xception()

    model.summary()
    history = model.fit(training_set, epochs= 500 ,validation_data = val_set,
                       callbacks=[early_stopping, reduce_lr ,checkpoint]
                        )


    final_model_path= '/final_model.h5'
    #model.save_weights(final_model_path)
    model.save(final_model_path)
    print('Model saved.')

    time_end = time.time()
    print('Training Time:', time_end - time_start)
    print('\n')

    return history

def test():
    
    key=class_names
    
    print('Testing:')
    mod =  keras.models.load_model(saved_checkpoints)
    mod.evaluate(test_set)

    prob = mod.predict(test_set)
    predIdxs = np.argmax(prob, axis=1)


    print('\nClassification Report:')
    print(classification_report(test_set.labels, predIdxs,target_names = key, digits=5))

    cm = confusion_matrix(test_set.labels, predIdxs)
    cm_df = pd.DataFrame(cm,index = key,columns =key)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Values")
    plt.xlabel("Predicted Values")
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()


    # Convert integer labels to one-hot
    test_labels_one_hot = tf.keras.utils.to_categorical(test_set.labels, num_classes=len(key))

    # Overall AUC (macro-average)
    macro_auc_ovo = roc_auc_score(test_labels_one_hot, prob, multi_class='ovo', average='macro')
    print("Macro-average One-vs-One ROC AUC:", macro_auc_ovo)

    # Per-class AUC
    for i, class_name in enumerate(key):
        y_true = test_labels_one_hot[:, i]
        y_prob = prob[:, i]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        class_auc = auc(fpr, tpr)
        print(f"{class_name} AUC: {class_auc:.4f}")

    
   
if __name__ == "__main__":
    
    
    #train_history = train()
    test()
    


  
