from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

# set paths to train and test directories and their annotation files
#train_dir = "train/"
#test_dir = "test/"
#train_annotations_file = "train_ann.csv"
test_annotations_file = "test/_annotations.csv"
test_df = pd.read_csv(test_annotations_file)
#valid_annotations_file = "val_ann.csv"
num_classes = 30

# define the data generator for training data
# train_datagen = ImageDataGenerator(rescale=1./255)
# train_df = pd.read_csv(train_annotations_file)
# train_generator = train_datagen.flow_from_dataframe(
#         dataframe=train_df,
#         directory=None,
#         x_col="filepaths",
#         y_col="class",
#         target_size=(216, 216),
#         batch_size=32,
#         class_mode='categorical')

train_df, val_df = train_test_split(test_df, test_size=0.25, random_state=42)

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col="filename",
    y_col="class",
    target_size=(216, 216),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=None,
    x_col="filename",
    y_col="class",
    target_size=(216, 216),
    batch_size=32,
    class_mode='categorical'
)

# val_datagen = ImageDataGenerator(rescale=1./255)
# val_df = pd.read_csv(valid_annotations_file)

# val_generator = val_datagen.flow_from_dataframe(
#         dataframe=val_df,
#         directory=None,
#         x_col="filepaths",
#         y_col="class",
#         target_size=(216, 216),
#         batch_size=32,
#         class_mode='categorical')

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(216, 216, 3)))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3), activation='selu'))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dense(256, activation='sigmoid'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

# compile the model
model.compile(optimizer="adam",
              loss=keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# train the model
model.fit(train_generator, epochs=10, validation_data = (val_generator))

# evaluate the model on test data
test_loss, test_acc = model.evaluate(val_generator, verbose=2)
print('\nTest accuracy:', test_acc)

#model.save("ballsV10.h5")

