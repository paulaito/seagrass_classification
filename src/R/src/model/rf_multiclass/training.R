# Split into train and test

library(caret)
library(randomForest)
library(dplyr)

# Ria Formosa dataset
ria_ground_truth_dataset <- './data/processed/datasets/ria_formosa_truth_data.csv'
ria_truth_data <- read.csv(ria_ground_truth_dataset)

# Output
cm_out <- './reports/confusion_matrix/ria_formosa/R/cm_ria_formosa_seagrass_R.csv'
classes_stats_out <- './reports/confusion_matrix/ria_formosa/R/stats_classes_ria_formosa_seagrass_R.csv'
overall_stats_out <- './reports/confusion_matrix/ria_formosa/R/stats_overall_ria_formosa_seagrass_R.csv'

model_rds = './R/model/RF_ria_formosa_model.RDS'

# Eliminate seagrass unknown records
ria_truth_data_clean <- ria_truth_data %>% filter(habitatclass != 'seagrass unknown')
unique(ria_truth_data_clean$habitatclass)

# Add seagrass class column (target)
ria_truth_data_clean <- ria_truth_data_clean %>%
    mutate(seagrass_class = case_when(habitatclass == 'seagrass intertidal' ~ '1',
                                    habitatclass == 'seagrass subtidal' ~ '2',
                                    TRUE ~ '0'))

# Preparing dataset
features = c('blue', 'green', 'red', 'nir', 'ndvi', 'dem')

X_ria <- ria_truth_data_clean %>% select(features)
y_ria <- ria_truth_data_clean$seagrass_class

# Splitting train/test 80/20 stratified (keeping proportion of samples)
train.index <- createDataPartition(ria_truth_data_clean$seagrass_class, p = .8, list = FALSE)
train <- ria_truth_data_clean[ train.index,]
test  <- ria_truth_data_clean[-train.index,]

# Hyperparameter tuning
tunegrid <- expand.grid(mtry=c(1:20))

train_control <- trainControl(method = "cv", number = 5, search = "grid")

rf <- train(train[,features],
                  train$seagrass_class,
                  method ="rf",
                  metric = "Kappa",
                  tuneGrid = tunegrid,
                  trControl=trainControl(method="cv",
                                         number = 5,
                                         search = "grid"))

pred_y <- predict(rf, test)
test_y <- test$seagrass_class %>% as.factor()

# Save best model
saveRDS(rf, model_rds)

# Get statistics
cm <- confusionMatrix(data=pred_y, reference = test_y)

confusion_matrix <- as.table(cm)
classes_stats <- as.matrix(cm,what="classes")
overall_stats <- as.matrix(cm,what="overall")

write.csv(confusion_matrix, cm_out)
write.csv(classes_stats, classes_stats_out)
write.csv(overall_stats, overall_stats_out)
