from tensorflow import keras as k
# from k.preprocessing.image import ImageDataGenerator

train_datagen = k.preprocessing.imageImageDataGenerator(rescale=1./255.0, width_shift_range=0.1, height_shift_range=0.1)
train_generator = train_datagen.flow_from_directory(
        r'charRecognition/data/train',  # this is the target directory
        target_size=(28,28),  # all images will be resized to 28x28
        batch_size=1,
        class_mode='categorical')

validation_generator = train_datagen.flow_from_directory(
        r'charRecognition/data/val',  # this is the target directory
        target_size=(28,28),  # all images will be resized to 28x28        
        batch_size=1,
        class_mode='categorical')


model = k.models.Sequential()
model.add(k.layers.Conv2D(32, (24,24), input_shape=(28, 28, 3), activation='relu', padding='same'))
# model.add(Conv2D(32, (20,20), input_shape=(28, 28, 3), activation='relu', padding='same'))
# model.add(Conv2D(32, (20,20), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(k.layers.Dropout(0.4))
model.add(k.layers.Flatten())
model.add(k.layers.Dense(128, activation='relu'))
model.add(k.layers.Dense(36, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=k.optimizers.Adam(lr=0.0001), metrics=['accuracy'])

class stop_training_callback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('val_accuracy') > 0.992):
      self.model.stop_training = True

batch_size = 1
callbacks = [stop_training_callback()]
history = model.fit_generator(
      train_generator,
      steps_per_epoch = train_generator.samples // batch_size,
      validation_data = validation_generator, 
      validation_steps = validation_generator.samples // batch_size,
      epochs = 80, callbacks=callbacks, verbose =1)


model.save(r'charRecognition/trained_model.h5')