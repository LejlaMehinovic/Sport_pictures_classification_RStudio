
install.packages("devtools")
library(devtools)
devtools::install_github("rstudio/keras")
install.packages("keras")
install.packages("Rcpp")

library(keras)
install_keras()
install.packages("EBImage")
library(EBImage)
library(keras)
library(tensorflow)
use_condaenv("r-tensorflow")


start <- Sys.time()
#Kategorije
categories <- c("2", "34", "35", "36", "37", "38", "39", "40", "41",
                "42", "43")
# Broj kategorija
output_n <- length(categories)
output_n

#Velièina slika za treniranje i testiranje
img_width <- 20
img_height <- 20
target_size <- c(img_width, img_height)

# RGB
channels <- 3

#Put do slika za klasificiranje
path <- "C:/Users/Lejla/Desktop/SP"
train_image_files_path <- "C:/Users/Lejla/Desktop/SP/TRENING"
test_image_files_path <- "C:/Users/Lejla/Desktop/SP/TEST"
validation_image_files_path <- "C:/Users/Lejla/Desktop/PREZENTACIJA/TEST"

# Skaliranje vrijednosti od 0-1
train_data_gen = image_data_generator(
  rescale = 1/255
)
test_data_gen <- image_data_generator(
  rescale = 1/255
)  
valid_data_gen <- image_data_generator(
  rescale = 1/255
)

# Skup za treniranje
train_image_array_gen <- flow_images_from_directory(train_image_files_path, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    seed = 42)
#Skup za testiranje
test_image_array_gen <- flow_images_from_directory(test_image_files_path, 
                                                   test_data_gen,
                                                   target_size = target_size,
                                                   class_mode = "categorical",
                                                   seed = 42)
valid_image_array_gen <- flow_images_from_directory(validation_image_files_path, 
                                                    valid_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    seed = 42)

length(valid_image_files_path)

table(factor(train_image_array_gen$classes))
table(factor(test_image_array_gen$classes))
table(factor(valid_image_array_gen$classes))

train_image_array_gen$class_indices
classes_indices <- train_image_array_gen$class_indices
save(classes_indices, file = "C:/Users/Lejla/Desktop/SP/classes_indices.RData")

# Broj slika za treniranje
train_samples <- train_image_array_gen$n
# Broj slika za testiranje
test_samples <- test_image_array_gen$n

train_samples
test_samples

# Parametri za model
batch_size <- 32
epochs <- 20

#MODEL :
model <- keras_model_sequential()

# definiranje modela
model %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3),
                padding = "same", 
                input_shape = c(img_width, img_height, channels)) %>%
  layer_activation("relu") %>%
  layer_conv_2d(filter = 16, kernel_size = c(3,3),
                padding = "same") %>%
  layer_activation_leaky_relu(0.5) %>%
  layer_batch_normalization() %>%
  # max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  layer_flatten() %>%
  layer_dense(100) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(output_n) %>% 
  layer_activation("softmax")

# compile
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)  

#h5 file za spremanje modela
if (!requireNamespace("BiocManager", quietly = TRUE))
install.packages("BiocManager")
BiocManager::install("rhdf5")
library("rhdf5")
h5createFile("checkpoints2.h5")


# Fit model
hist <- model %>% fit_generator(
  # podaci za treniranje
  train_image_array_gen,
  # epohe
  steps_per_epoch = as.integer(train_samples / batch_size), 
  epochs = epochs, 
  # podaci za testiranje
  validation_data = test_image_array_gen,
  validation_steps = as.integer(test_samples / batch_size),
  # Prikazati napredak modela
  verbose = 2,
  callbacks = list(
    # spremiti najbolji model nakon svake epohe
  callback_model_checkpoint("C:/Users/Lejla/Desktop/SP/checkpoints2.h5",
                              save_best_only = TRUE)
  )
)

df_out <- hist$metrics %>% 
{data.frame(acc = .$acc[epochs], val_acc = .$val_acc[epochs], elapsed_time = as.integer(Sys.time()) - as.integer(start))}


#model2 <- load_model_hdf5(filepath = "C:/Users/Lejla/Desktop/SP/checkpoints2.h5")
#model2

test_datagen <- image_data_generator(rescale = 1/255)
test_generator <- flow_images_from_directory(
  test_image_files_path,
  test_datagen,
  target_size = c(20, 20),
  class_mode = 'categorical')


validation_image_files_path <- "C:/Users/Lejla/Desktop/ValidationSlike"

validationdf <- as.data.frame(validation_image_files_path)

test_generatorValid <- flow_images_from_directory(
  validation_image_files_path,
  test_datagen,
  target_size = c(20, 20),
  class_mode = 'categorical')

#Predikcija



predictionsForTest<- as.data.frame(predict_generator(model, test_generatorValid, steps = 1))

colnames(predictionsForTest) <- c("2","34","35","36","37","38","39","40","41","42","43")


predictionsNEW<-as.data.frame(cbind(row.names(predictionsForTest),apply(a,1,function(x) names(a)[which(x==max(x))])))





predictions <- as.data.frame(predict_generator(model, test_generator, steps = 1))



#Uèitati najbolji model
load("C:/Users/Lejla/Desktop/SP/classes_indices.RData")
classes_indices_df <- data.frame(indices = unlist(classes_indices))
classes_indices_df <- classes_indices_df[order(classes_indices_df$indices), , drop = FALSE]
colnames(predictions) <- rownames(classes_indices_df)

#Ispisati predikciju
t(round(predictions, digits = 2))
for (i in 1:nrow(predictions)) {
  cat(i, ":")
  print(unlist(which.max(predictions[i, ])))
}

nrow(predictions)

#Priprema slika za testiranje 
image_prep2 <- function(x) {
  arrays <- lapply(x, function(path) {
    img <- image_load(path, target_size = c(20, 20))
    x <- image_to_array(img)
    x <- reticulate::array_reshape(x, c(1, dim(x)))
    x <- x / 255
  })
  do.call(abind::abind, c(arrays, list(along = 1)))
}

#Putanja slika za validaciju
img_path4<-"C:/Users/Lejla/Desktop/SP/VALIDATION/COCO_val2014_000000010684.jpg"
img_path5<-"C:/Users/Lejla/Desktop/SP/VALIDATION/COCO_val2014_000000020001.jpg"
img_path6<-"C:/Users/Lejla/Desktop/SP/VALIDATION/COCO_val2014_000000026982.jpg"

#Testiranje modela na slikama za validaciju
res4 <- predict(model, image_prep2(c(img_path4)))
res4
#[,1]       [,2]      [,3]      [,4]       [,5]      [,6]        [,7]       [,8]       [,9]     [,10]      [,11]
#[1,] 0.008056561 0.01382886 0.3248067 0.1921888 0.01196285 0.1149128 0.002128333 0.00161347 0.00732727 0.3009606 0.02221371


validation_image_files_path <- "C:/Users/Lejla/Desktop/PREZENTACIJA/TEST"






res7 <- predict(model, image_prep2(c(img_path6)))
res7
res5<-predict(model,image_prep2(c(img_path5)) )
res5
res6<-predict(model,image_prep2(c(img_path6)) )
res6


#ZA TESTIRANJE

slika1<-"C:/Users/Lejla/Desktop/PREZENTACIJA/TEST/COCO_train2014_000000000419.jpg"
rez1<-predict(model, image_prep2(c(slika1)))
rez1
max.col(rez1)


slika2<-"C:/Users/Lejla/Desktop/PREZENTACIJA/TEST/COCO_train2014_000000000149.jpg"
rez2<-predict(model, image_prep2(c(slika2)))
rez2

max.col(rez2)

slika3<-"C:/Users/Lejla/Desktop/PREZENTACIJA/TEST/COCO_train2014_000000000201.jpg"
rez3<-predict(model, image_prep2(c(slika3)))
rez3
max.col(rez3)

slika4<-"C:/Users/Lejla/Desktop/PREZENTACIJA/TEST/COCO_train2014_000000000307.jpg"
rez4<-predict(model, image_prep2(c(slika4)))
rez4
max.col(rez4)

slika5<-"C:/Users/Lejla/Desktop/PREZENTACIJA/TEST/COCO_train2014_000000000322.jpg"
rez5<-predict(model, image_prep2(c(slika5)))
rez5
max.col(rez5)

slika6<-"C:/Users/Lejla/Desktop/PREZENTACIJA/TEST/COCO_train2014_000000000368.jpg"
rez6<-predict(model, image_prep2(c(slika6)))
rez6
max.col(rez6)


slika7<-"C:/Users/Lejla/Desktop/PREZENTACIJA/TEST/COCO_train2014_000000000382.jpg"
rez7<-predict(model, image_prep2(c(slika7)))
rez7
max.col(rez7)


slika8<-"C:/Users/Lejla/Desktop/PREZENTACIJA/TEST/COCO_train2014_000000000394.jpg"
rez8<-predict(model, image_prep2(c(slika8)))
rez8
max.col(rez8)

slika9<-"C:/Users/Lejla/Desktop/PREZENTACIJA/TEST/COCO_train2014_000000000419.jpg"
rez9<-predict(model, image_prep2(c(slika9)))
rez9
max.col(rez9)

slika10<-"C:/Users/Lejla/Desktop/PREZENTACIJA/TEST/COCO_train2014_000000000431.jpg"
rez10<-predict(model, image_prep2(c(slika10)))
rez10
max.col(rez10)


for(i in 1:281){
  slika[i]<-"C:/Users/Lejla/Desktop/PREZENTACIJA/TEST/COCO_train2014_000000000077.jpg"
}

#Nazivi klasa
classes_indices_l <- rownames(classes_indices_df)
names(classes_indices_l) <- unlist(classes_indices)
classes_indices_l
