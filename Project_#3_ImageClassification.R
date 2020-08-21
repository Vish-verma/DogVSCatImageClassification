# Cleaning the memory
rm(list=ls())

#Set Working Directory
setwd('E:/Data Science(Edwisor)/Project/Project #3/Deep Learning Dataset - Cat vs Dog Classifier/dog-cat-full-dataset-master/data')

#Loading Libraries
#install.packages('tfdatasets')
library(keras)


#Setting Training and Test Path
train_image_files_path="E:/Data Science(Edwisor)/Project/Project #3/Deep Learning Dataset - Cat vs Dog Classifier/dog-cat-full-dataset-master/data/train/"
valid_image_files_path="E:/Data Science(Edwisor)/Project/Project #3/Deep Learning Dataset - Cat vs Dog Classifier/dog-cat-full-dataset-master/data/test/"

# IMAGE CLASSIFICATION USING CNN MODEL

# image size 
img_width=180
img_height=180
target_size=c(img_width, img_height)
channels=3

#Importing Images and doing Augmentation
train_data_gen = image_data_generator(
  rescale = 1/255,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE)

train_image_array_gen=flow_images_from_directory(train_image_files_path, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "binary",
                                                    seed = 42)

# Validation data (NO Augmentation)
valid_data_gen=image_data_generator(
  rescale = 1/255
)  

valid_image_array_gen=flow_images_from_directory(valid_image_files_path, 
                                                    valid_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "binary",
                                                     seed = 42)

# number of training samples
train_samples=train_image_array_gen$n
# number of validation samples
valid_samples=valid_image_array_gen$n

# define batch size and number of epochs
batch_size <- 32
epochs <- 10


# initialise model
model=keras_model_sequential()

# add layers
model %>%
  layer_conv_2d(filter = 16, kernel_size = c(3,3), padding = "same", input_shape = c(img_width, img_height, channels)) %>%
  layer_activation("relu") %>%
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same") %>%
  layer_activation("relu") %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  # Flatten max filtered output into feature vector and feed into dense layer
  layer_flatten() %>%
  layer_dense(128) %>%
  layer_activation("relu") %>%
  
  # Outputs from dense layer are projected onto output layer
  layer_dense(1) %>% 
  layer_activation("sigmoid")

# compile
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = 'adam',
  metrics = "accuracy"
)


# training the model on the dataset with validation set
hist = model %>% fit_generator(
  # training data
  train_image_array_gen,
  
  # epochs
  epochs = epochs, 
  steps_per_epoch = as.integer(train_samples / batch_size), 
  
  # validation data
  validation_data = valid_image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
  
  # print progress
  verbose = 2)

#Graph of Loss vs Validation Loss and Accuracy vs Validation Accuracy
plot(hist)


#IMAGE CLASSIFICATION USING TRANSFER LEARNING

#Getting a Pre trained Mobilenet V2 model without the output layer
conv_base = application_mobilenet_v2(weights = "imagenet",include_top = FALSE,input_shape = c(160, 160, 3))

# image size 
img_width=160
img_height=160
target_size=c(img_width, img_height)
channels=3

#Adding Our Output Layer to the MobileNetV2 Model

model_transfer=keras_model_sequential() %>% 
  conv_base %>% 
  layer_flatten() %>% 
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

#Getting Trainable Parameters
length(model_transfer$trainable_weights)

#Freezing the weights of pre trained model
freeze_weights(conv_base)
length(model_transfer$trainable_weights)


#Importing Images and doing Augmentation
train_data_gen = image_data_generator(
  rescale = 1/255,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE)

train_image_array_gen=flow_images_from_directory(train_image_files_path, 
                                                 train_data_gen,
                                                 target_size = target_size,
                                                 class_mode = "binary",
                                                 seed = 42)

# Validation data (Augmentation)
valid_data_gen=image_data_generator(
  rescale = 1/255
)  

valid_image_array_gen=flow_images_from_directory(valid_image_files_path, 
                                                 valid_data_gen,
                                                 target_size = target_size,
                                                 class_mode = "binary",
                                                 seed = 42)

#Compiling Model with parameters
model_transfer %>% compile(
  loss = "binary_crossentropy",
  optimizer = 'adam',
  metrics = c("accuracy")
)

#Training the model on dataset 
history_tranfer <- model_transfer %>% fit_generator(
  train_image_array_gen,
  steps_per_epoch = 100,
  epochs = 10,
  validation_data = valid_image_array_gen,
  validation_steps = 50
)

#Graph of Loss vs Validation Loss and Accuracy vs Validation Accuracy
plot(history_tranfer)


#IMAGE CLASSIFICATION USING AUTO ENCODERS


#Creating an Auto Encoder Model

# image size 
img_width=64
img_height=64
target_size=c(img_width, img_height)
channels=3

#Encoding Layer
#encoder_model = keras_model_sequential()
enc_input = layer_input(c(64, 64, 3))
# add layers to encoder
encoder_model = enc_input %>%
  layer_conv_2d(filter = 48, kernel_size = c(3,3), padding = "same") %>%
  layer_activation("relu") %>%
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 96, kernel_size = c(3,3), padding = "same") %>%
  layer_activation("relu") %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  # Third hidden layer
  layer_conv_2d(filter = 192, kernel_size = c(3,3), padding = "same") %>%
  layer_activation("relu") %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  # Fourth hidden layer
  layer_conv_2d(filter = 32, kernel_size = c(1,1), padding = "same")

#Latent Image Size
Lt_width=8
Lt_height=8
Lt_size=c(img_width, img_height)

#Decoding Layers
#decoder_model = keras_model_sequential()

# add layers to decoder
decoder_model = encoder_model %>%
  layer_conv_2d(filter = 192, kernel_size = c(1,1), padding = "same") %>%
  layer_activation("relu") %>%
  # Use up sampling
  layer_upsampling_2d(c(2,2)) %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 192, kernel_size = c(3,3), padding = "same") %>%
  layer_activation("relu") %>%
  
  # Use up sampling
  layer_upsampling_2d(c(2,2)) %>%
  
  # Third hidden layer
  layer_conv_2d(filter = 96, kernel_size = c(3,3), padding = "same") %>%
  layer_activation("relu") %>%
  
  # Use up sampling
  layer_upsampling_2d(c(2,2)) %>%
  
  # Fourth hidden layer
  layer_conv_2d(filter = 48, kernel_size = c(3,3), padding = "same")%>%
  
  #Last Layer
  layer_conv_2d(filter = 3, kernel_size = c(3,3), padding = "same")


# Creating Model from both layers
auto_Encoder = keras_model(enc_input,decoder_model)

# compile
auto_Encoder %>% compile(
  loss = "binary_crossentropy",
  optimizer = 'adam'
)

#Importing Images and doing Augmentation for Autoencoder class mode should be input
train_data_gen = image_data_generator(
  rescale = 1/255,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE)

train_image_array_gen=flow_images_from_directory(train_image_files_path, 
                                                 train_data_gen,
                                                 target_size = target_size,
                                                 class_mode = "input",
                                                 seed = 42)

# Validation data (NO Augmentation)
valid_data_gen=image_data_generator(
  rescale = 1/255
)  

valid_image_array_gen=flow_images_from_directory(valid_image_files_path, 
                                                 valid_data_gen,
                                                 target_size = target_size,
                                                 class_mode = "binary",
                                                 seed = 42)


#Training the model on dataset 
history_auto_encoder <- auto_Encoder %>% fit_generator(
  train_image_array_gen,
  steps_per_epoch = 100,
  epochs = 10,
  verbose= 2)



#Plotting few images to see encoders ouput
one_batch = train_image_array_gen[0]
pred = auto_Encoder %>% predict(one_batch) 

n = 10
for (i in 1:n) 
{
  plot(as.raster(pred[i,,,]))
}


#Creating the Final model using the AutoEncoders weights
enc_input = layer_input(c(64, 64, 3))
#finalModel = keras_model_sequential()
finalModel_enc = enc_input %>%
  layer_conv_2d(filter = 48, kernel_size = c(3,3), padding = "same") %>%
  layer_activation("relu") %>%
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 96, kernel_size = c(3,3), padding = "same") %>%
  layer_activation("relu") %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  # Third hidden layer
  layer_conv_2d(filter = 192, kernel_size = c(3,3), padding = "same") %>%
  layer_activation("relu") %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  # Fourth hidden layer
  layer_conv_2d(filter = 32, kernel_size = c(1,1), padding = "same")

finalModel_nn = finalModel_enc %>%
  layer_flatten() %>%
  layer_dense(256) %>%
  layer_activation("relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(0.5) %>%
  # Outputs from dense layer are projected onto output layer
  layer_dense(1) %>% 
  layer_activation("sigmoid")

finalModel = keras_model(enc_input,finalModel_nn)

# compile
finalModel %>% compile(
  loss = "binary_crossentropy",
  optimizer = 'adam',
  metrics = c("accuracy")
)

#Importing Images and doing Augmentation for final Model
train_data_gen = image_data_generator(
  rescale = 1/255,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE)

train_image_array_gen=flow_images_from_directory(train_image_files_path, 
                                                 train_data_gen,
                                                 target_size = target_size,
                                                 class_mode = "binary",
                                                 seed = 42)

#Copying the weights of autoencoder model to Full Model as it have already been trained.
for(n in 1:10){
  layer_final = finalModel$get_layer(index = n)
  layer_encoder = auto_Encoder$get_layer(index = n)
  weight = layer_encoder$get_weights()
  layer_final$set_weights(weight)
  freeze_weights(layer_final)
}

# number of training samples
train_samples=train_image_array_gen$n
# number of validation samples
valid_samples=valid_image_array_gen$n

# define batch size and number of epochs
batch_size <- 32
epochs <- 10

# training the model on the dataset with validation set
hist_encoder = finalModel %>% fit_generator(
  # training data
  train_image_array_gen,
  
  # epochs
  epochs = 20, 
  steps_per_epoch = as.integer(train_samples / batch_size), 
  
  # validation data
  validation_data = valid_image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
  
  # print progress
  verbose = 2)

#Graph of Loss vs Validation Loss and Accuracy vs Validation Accuracy
plot(hist_encoder)
