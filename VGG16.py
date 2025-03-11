import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
print(tf.config.list_physical_devices("GPU"))
# location of data files
all_data_loc = 'All_data'

# train blocks
train_blocks = ['block_0101', 'block_0102', 'block_0203', 'block_0301']

# get the density maps and the corresponding image subwindows so that they can be stacked for the train data
all_feature_subwindow_names = []
all_density_map_names =[]

for folder_name in train_blocks:
    path_name = os.path.join(all_data_loc, folder_name)
    density_map = [file for file in os.listdir(path_name)if file[:7] == 'density']
    feature_map = [file for file in os.listdir(path_name)if file[:7] == 'subwind']
    all_feature_subwindow_names.append(feature_map)
    all_density_map_names.append(density_map)

# load all feature maps
all_feature_maps = []
counter = 0
for item in np.array(all_feature_subwindow_names).flatten():
    load_map = np.load(os.path.join(all_data_loc, train_blocks[counter], item))
    all_feature_maps.append(load_map)
    counter = counter + 1

# load density maps
all_density_maps = []
counter = 0
for map in np.array(all_density_map_names).flatten():
    load_density_map = np.load(os.path.join(all_data_loc, train_blocks[counter], map))
    all_density_maps.append(load_density_map)
    counter = counter + 1

# check the shape of an item
print(all_density_maps[0].shape, all_feature_maps[0].shape)

# stack all data together?

# I think we need to stack vertically
all_train_density_maps = np.vstack(all_density_maps)

all_train_feature_maps = np.vstack(all_feature_maps)

# check these shapes
all_train_density_maps.shape, all_train_feature_maps.shape

# we may need to do a sanity check to know if we have correctly stacked
for i in range(4):
    print(np.mean(all_density_maps[i] == all_train_density_maps[12288*i: 12288 + 12288*i,:]))

# for feature maps
for i in range(4):
    print(np.mean(all_feature_maps[i] == all_train_feature_maps[12288*i: 12288 + 12288*i,:]))

# Also need to load the validation data

# list the contents in the validation folder
all_valid_contents = os.listdir("All_data/block_0204")

validation_feature_maps = np.load(os.path.join("All_data/block_0204", 'subwindow_seqs_0204.npy'))

validation_density_map = np.load(os.path.join("All_data/block_0204", 'density_maps_seqs_0204.npy'))

inputs = tf.keras.Input(shape = (None, 32, 32, 3))
resized_input = tf.keras.layers.TimeDistributed(tf.keras.layers.Resizing(224,224))(inputs)
base_model = tf.keras.applications.VGG16(weights = 'imagenet', include_top = False, input_shape = (224,224,3))
base_model.summary()
# freeze all model layers here
for layer in base_model.layers:
    layer.trainable = False

base_model.summary()

# 3. Add custom layers on top of the VGG16 base model
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)  # Apply global average pooling
x = tf.keras.layers.Dense(1024, activation='relu')(x)  # Add a fully connected layer
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)

new_model = tf.keras.models.Model(inputs = base_model.input, outputs = x)

new_model.summary()

# time dist layer for the entire model
time_dist_layer = tf.keras.layers.TimeDistributed(new_model)

time_dist_output = time_dist_layer(resized_input)
time_dist_output

lstm_layer = tf.keras.layers.LSTM(64)
lstm_output = lstm_layer(time_dist_output)
lstm_output

# pred_layer_td = tf.keras.layers.TimeDistributed()
pred_layer = tf.keras.layers.Dense(7, activation = "relu")
# pred_time_dist_layer = tf.keras.layers.TimeDistributed(pred_layer)
pred_layer_out = pred_layer(lstm_output)
pred_layer_out

cnn_lstm = tf.keras.models.Model(inputs, pred_layer_out)
cnn_lstm.summary()

# Define and add the generator function
def data_generator(x_data, y_data, batch_size, shuffle=False, peek=False, verbose=False):
    num_samples = len(x_data)
    indices = np.arange(num_samples)
    
    if peek:    # Give first batch unshuffled and don't change start index when peeking for training
        end = min(batch_size, num_samples)
        if verbose:
            print(f"Generating peeking batch up to index {end}")
        batch_x = x_data[:end]
        batch_y = y_data[:end]
        peek = False
        yield (batch_x, batch_y)

    while True:    # Loop indefinitely for epochs
        # Shuffle indices at the start of each epoch after the peek, if shuffle is enabled
        if shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]

            # Print batch indices if verbose
            if verbose:
                # Warning: calling verbose when shuffling will usually clutter output
                if shuffle:
                    if len(batch_indices) < 16:
                        print(f"Batch indices: {np.sort(batch_indices)}")
                    else:
                        print(f"Printing batch indices would clutter output. Skipped.")
                    print(f"Length: {len(batch_indices)}")
                else:
                    print(f"Generating batch from index {start} to {end}")
                # traceback.print_stack()

            # Generate batches
            batch_x = x_data[batch_indices]
            batch_y = y_data[batch_indices]

            # Yield the current batch
            yield (batch_x, batch_y)


batch_size = 64
generator_batch_size = 64
train_gen = data_generator(all_train_feature_maps, all_train_density_maps, batch_size, shuffle=False, peek=True, verbose=False)
val_gen = data_generator(validation_feature_maps, validation_density_map, generator_batch_size, shuffle=False, peek=True, verbose=False)


steps_per_epoch = int(np.ceil(len(all_train_feature_maps) / batch_size))
train_steps = int(np.ceil(len(all_train_feature_maps) / generator_batch_size))
validation_steps = int(np.ceil(len(validation_feature_maps) / generator_batch_size))


# compile the model
opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
cnn_lstm.compile(loss='mean_squared_error', optimizer=opt, metrics = ['mean_absolute_error'])
    
# add early stopping
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights = True, verbose=1, patience=10)

# fit the generator - the steps are a must here
history = cnn_lstm.fit(train_gen,
          validation_data = val_gen, steps_per_epoch=steps_per_epoch, 
                                  validation_steps=validation_steps,           
          epochs = 30, callbacks = [es])


# save this model
model_name_frozen = 'VGG16_CNN_LSTM_frozen.keras'
cnn_lstm.save('models' + '/' + model_name_frozen)


for layer in base_model.layers[-4:]:
    layer.trainable = True

new_model.summary()

cnn_lstm.summary()

# compile the model
opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
cnn_lstm.compile(loss='mean_squared_error', optimizer=opt, metrics = ['mean_absolute_error'])
    
# add early stopping
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights = True, verbose=1, patience=10)

# fit the generator - the steps are a must here
history_new = cnn_lstm.fit(train_gen,
          validation_data = val_gen, steps_per_epoch=steps_per_epoch, 
                                  validation_steps=validation_steps,           
          epochs = 30, callbacks = [es])


# save this model
model_name_finetuned = 'VGG16_CNN_LSTM_finetuned.keras'
cnn_lstm.save('models' + '/' + model_name_finetuned)




























