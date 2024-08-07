import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

def create_simulated_dataset(num_samples=100):
    os.makedirs('dataset/red_car', exist_ok=True)
    os.makedirs('dataset/blue_car', exist_ok=True)
    os.makedirs('dataset/person', exist_ok=True)
    os.makedirs('dataset/other_vehicle', exist_ok=True)

    for i in range(num_samples):
        # Create a red car image
        red_car = np.zeros((128, 128, 3), dtype=np.uint8)
        red_car[:, :, 2] = 255
        cv2.imwrite(f'dataset/red_car/red_car_{i}.png', red_car)

        # Create a blue car image
        blue_car = np.zeros((128, 128, 3), dtype=np.uint8)
        blue_car[:, :, 0] = 255
        cv2.imwrite(f'dataset/blue_car/blue_car_{i}.png', blue_car)

        # Create a person image
        person = np.zeros((128, 128, 3), dtype=np.uint8)
        person[:, :, 1] = 255
        cv2.imwrite(f'dataset/person/person_{i}.png', person)

        # Create an other vehicle image
        other_vehicle = np.zeros((128, 128, 3), dtype=np.uint8)
        other_vehicle[:, :, 0] = 128
        other_vehicle[:, :, 1] = 128
        cv2.imwrite(f'dataset/other_vehicle/other_vehicle_{i}.png', other_vehicle)

create_simulated_dataset()
def load_and_preprocess_images(directory):
    images = []
    labels = []
    class_names = ['red_car', 'blue_car', 'person', 'other_vehicle']
    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(directory, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith('.png'):
                img = cv2.imread(os.path.join(class_dir, filename))
                img = cv2.resize(img, (128, 128))
                images.append(img)
                labels.append(idx)
    images = np.array(images)
    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=len(class_names))
    return images, labels

# Load dataset
train_images, train_labels = load_and_preprocess_images('dataset')
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow(train_images, train_labels, batch_size=32, subset='training')
validation_generator = train_datagen.flow(train_images, train_labels, batch_size=32, subset='validation')

history = model.fit(train_generator, epochs=10, validation_data=validation_generator)

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2





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

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            # Open the image
            image = cv2.imread(file_path)

# Resize the image using OpenCV with anti-aliasing
            resized_image = cv2.resize(image, (400, 400), interpolation=cv2.INTER_AREA)

# Convert the image to RGB format (required for displaying with Tkinter)
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

# Convert the resized image to a Tkinter-compatible format
            image = ImageTk.PhotoImage(image=Image.fromarray(resized_image))
           # image = Image.open(file_path)
            #image.thumbnail((400, 400), Image.ANTIALIAS)
            #image = image.resize((400, 400), Image.ANTIALIAS)
            #image = ImageTk.PhotoImage(image)
            if self.panel is None:
                self.panel = tk.Label(image=image)
                self.panel.image = image
                self.panel.pack(side="bottom", fill="both", expand="yes")
            else:
                self.panel.configure(image=image)
                self.panel.image = image
            self.analyze_image(file_path)

    def analyze_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (128, 128))
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        prediction = self.model.predict(img)
        prediction = np.argmax(prediction, axis=1)

        if prediction == 0:
            self.label.configure(text="Red car (marked as blue)")
        elif prediction == 1:
            self.label.configure(text="Blue car (marked as red)")
        elif prediction == 2:
            self.label.configure(text="Person detected")
        else:
            self.label.configure(text="Other vehicle detected")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficAnalyzerApp(root)
    root.mainloop()
