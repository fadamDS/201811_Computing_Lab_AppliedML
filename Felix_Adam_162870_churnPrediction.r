
# Load libraries.
library(caret)
library(plyr)
library(dplyr)
library(ggplot2)
library(pROC)

# Load source files.
source("io.R")
source('classDistribution.R')

# Set seed.
set.seed(321)

# Set path to data set.
path <- "Data/churn.csv"

# Count number of lines in input file.
lines <- readChar(path, file.info(path)$size)
total.rows <- length(gregexpr("\n",lines)[[1L]])
rm(lines)

# Load data.
df <- load.data(p.path = path,
				p.header = TRUE,
				p.dec = ".",
				p.sep = ",",
				p.blank.lines.skip = TRUE,
				p.stringsAsFactors = FALSE,
				p.comment.char = "",
				p.initial.rows = 100,
				p.total.nrows = total.rows,
				p.id = FALSE)

# Start time for performance measures later
start.time <- as.numeric(as.POSIXct(Sys.time()))

# Remove missing cases.
df <- df[complete.cases(df),]

# Remove redundant features -> bcookie
df$bcookie <- NULL
df$timestamp <- NULL

# Tranform the class values to factors.
df$cluster <- as.factor(df$cluster)

### Preprocessing and feature engineering ### 

# Extracing the column names, for easier filtering on certain attributes later
columns <- colnames(df)

# Extracting different columns for active_days, dwell_time and so on
active_day_cols <- columns[grepl('ST_act', columns)]
dwell_time_cols <- columns[grepl('ST_dwelltime', columns)]
session_cols <- columns[grepl('ST_num_sessions', columns)]
pageviw_cols <- columns[grepl('ST_pageviews', columns)]
click_cols <- columns[grepl('ST_cliks', columns)]

## Feature Engeneering ##
# My feature engeneering is mostly based on summarizing columns over days # 

# Take sum of active days, similar to average active days
df$total_active_days <- df %>% select(active_day_cols) %>% rowSums()

# Total dwell time
df$total_dwell_time <- df %>% select(dwell_time_cols) %>% rowSums()

# Number of sessions
df$total_session <- df %>% select(session_cols) %>%  rowSums()

# Average dwell time per session
df$avg_dwell_session <- df$total_dwell_time / df$total_session

# Total pageviews
df$total_pageviews <- df%>% select(pageviw_cols) %>% rowSums()

# Avg. Pageviews per session
df$views_per_session <- df$total_pageviews / df$total_active_days

# Total nr of clicks
df$total_clicks <- df %>% select(click_cols) %>% rowSums()

# Average Clicks per Session
df$avg_click_session <- df$total_pageviews / df$total_session

# Maximum dwell time using row-wise maximum and apply 
df$max_dwell <- apply(df[dwell_time_cols], 1, function(x) max(x))

# Set the class.
class <- 'cluster'

# Select cols to use 
use_cols <- c(class,'total_active_days','total_dwell_time','total_session','avg_dwell_session','total_pageviews','views_per_session','total_clicks', 
                'avg_click_session','max_dwell')


# Remove unwanted columns
df <- df%>% select(use_cols)

# Position of class
class_col <- which(colnames(df) == class)

### Setting up training and testing sets ###

# Perform stratified bootstrapping (keep 60% of observations for training and 40% for testing).
indices.training <- createDataPartition(df[,class_col], 
                                        times = 1, 
                                        p = .60, 
                                        list = FALSE)

# Get training and test set.
training <- df[indices.training[,1],]
test  <- df[-indices.training[,1],]

# Print class distribution to verify correct stratified sampling
classDistribution(dataset.name = "df",
                  table = df,
                  class = class_col)

classDistribution(dataset.name = "training",
                  table = training,
                  class = class_col)

classDistribution(dataset.name = "test",
                  table = test,
                  class = class_col)

# Get column names for use in the formula
train_cols <- colnames(training)

formula <- as.formula(paste("cluster ~ ", paste(train_cols[-which(train_cols == 'cluster')], collapse= "+")))

print(formula)

# Parameters for Grid Search, expand grid creates a dtaframe from all combinations of supplied vectors
xgboostGrid <- expand.grid(nrounds = c(100,200), # Number of boosting rounds
                           eta = c(0.9,1), # Learning Rate
                           gamma = c(0.8,0.9,1), # Minimum loss reduction for split
                           colsample_bytree = c(0.5,1.0), # Fraction of cols to be randomly sampled
                           max_depth = c(4,6,8,10), # Maximum depth of a tree
                           min_child_weight = c(1,4,8), # Minimum Sum of weights 
                           subsample = 1) # Fraction of observations to be randomly sampled for each tree

# Setting up the values for the train function later
xgboostControl = trainControl(method = "cv", # Cross validation 
                              number = 10, # Using 10-folds
                              classProbs = TRUE, # Class probabilities will be computed
                              search = "grid", # Using grid search
                              allowParallel = TRUE) # Enable parallelization

# Set the maximum number of threads to be used, on my machine this is 4.
max.threads <- 4

# Train the model using carets train function
# Handing over the specified parameters
model.training <- caret::train(formula, 
                          data = training,
                          method = "xgbTree",
                          trControl = xgboostControl,
                          tuneGrid = xgboostGrid,
                          verbose = TRUE,
                          metric = "Accuracy",
                          maximize = TRUE,
                          nthread = max.threads)

# Print basic information about our model.
print(model.training)
print(model.training$finalModel)

# The plot function can be used to examine the relationship between 
# the estimates of performance and the tuning parameters.
ggplot(model.training, output = "ggplot") + theme(legend.position = "top")

# Evaluate model on test set.
end.time <- as.numeric(as.POSIXct(Sys.time()))
tt <- end.time - start.time
print(paste(c("tt: ", tt), sep = "", collapse = ""))

# Make Class Predictions on 
model.test.pred <- predict(model.training, 
                           test, 
                           type = "raw",
                           norm.votes = TRUE)

# Make probabilty predictions for computing AUC later
model.test.prob <- predict(model.training, 
                           test, 
                           type = "prob",
                           norm.votes = TRUE)

# Compute confusion matrix.
performance <- confusionMatrix(model.test.pred, test$cluster)
print(performance)

# Compute AUC
model.roc <- plot.roc(predictor = model.test.prob[,2],  
                      test[,class],
                      levels = rev(levels(test[,class])),
                      legacy.axes = FALSE,
                      percent = TRUE,
                      mar = c(4.1,4.1,0.2,0.3),
                      identity.col = "red",
                      identity.lwd = 2,
                      smooth = FALSE,
                      ci = TRUE, 
                      print.auc = TRUE,
                      auc.polygon.border=NULL,
                      lwd = 2,
                      cex.lab = 2.0, 
                      cex.axis = 1.6, 
                      font.lab = 2,
                      font.axis = 2,
                      col = "blue")

# Confidence Interval for ROC
ciobj <- ci.se(model.roc, specificities = seq(0, 100, 5))

# Plotting confidence interval and optimal threshold
plot(ciobj, type = "shape", col = "#1c61b6AA")
plot(ci(model.roc, of = "thresholds", thresholds = "best"))

