import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Load dataset
def load_traffic_dataset(images_directory, labels_directory):
    images = []
    labels = []
    class_names = sorted(os.listdir(labels_directory))
    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(images_directory, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img = cv2.imread(os.path.join(class_dir, filename))
                img = cv2.resize(img, (128, 128))
                images.append(img)
                labels.append(idx)
    images = np.array(images)
    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=len(class_names))
    return images, labels

# Load dataset from downloaded folders
images_directory = r"D:\NULLCLASS- AGE AND GENDER DETECTOR\Traffic Dataset"
labels_directory = r"D:\NULLCLASS- AGE AND GENDER DETECTOR\Traffic Dataset"
train_images, train_labels = load_traffic_dataset(images_directory, labels_directory)

# Split the data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Create the pre-trained model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_labels.shape[1], activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Create ImageDataGenerator instances for data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator()

# Create the generators
train_generator = train_datagen.flow(train_images, train_labels, batch_size=32)
val_generator = val_datagen.flow(val_images, val_labels, batch_size=32)

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=val_generator)

# GUI setup
class TrafficAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Analyzer")
        self.model = model
        self.label = tk.Label(root, text="Upload an image to analyze")
        self.label.pack()
        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()
        self.panel = None
        self.img_ref = None

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv2.imread(file_path)
            resized_image = cv2.resize(image, (400, 400), interpolation=cv2.INTER_AREA)
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            image = ImageTk.PhotoImage(image)
            self.img_ref = ImageTk.PhotoImage(image)
            if self.panel is None:
                self.panel = tk.Label(image=image)
                self.panel.image = image
                self.panel.pack(side="bottom", fill="both", expand="yes")
            else:
                self.panel.configure(image=image)
                self.panel.image = image
            self.analyze_image(file_path)

   # Update the analyze_image method in the TrafficAnalyzerApp class
    def analyze_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (128, 128))
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        prediction = self.model.predict(img)
        prediction = np.argmax(prediction, axis=1)

    # Assuming class_names are sorted in alphabetical order
        class_names = sorted(os.listdir(labels_directory))
        label = class_names[prediction[0]]

        self.label.configure(text=label)

   

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficAnalyzerApp(root)
    root.mainloop()
