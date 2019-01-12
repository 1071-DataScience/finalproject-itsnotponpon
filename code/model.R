library('rpart')
library('argparser')

parser <- arg_parser('Final Project on Duolingo SLAM dataset')
parser <- add_argument(parser, '--train', help='Train file')
parser <- add_argument(parser, '--test', help='Test file')
parser <- add_argument(parser, '--pred', help='Output prediction file name', default='')
argv <- parse_args(parser)
train_f <- argv$train
test_f <- argv$test
pred_f <- argv$pred
if (pred_f == '') {
  pred_f <- sub('test', 'pred', test_f)
}
cat(train_f, test_f, pred_f)
# Somehow get train and test file names...
train_f <- '../data_fr_en/fr_en.slam.20171218.train'
test_f <- '../data_fr_en/fr_en.slam.20171218.test'

# The dataset format is almost SSV, but not quite
load_data <- function(filename, is_train) {
  raw_lines <- readLines(filename)
  # Read the SSV part first
  df <- read.table(filename, header=F, as.is=c(1, 4))
  if (is_train) {
    colnames(df) <- c('uid', 'token', 'pos', 'morph', 'depLabel', 'depHead', 'label')
  } else {
    colnames(df) <- c('uid', 'token', 'pos', 'morph', 'depLabel', 'depHead')
  }
  # Filter out the exercise header
  wh <- which(startsWith(raw_lines, '#'))
  # Count where they originally belong
  exercise_len <- c()
  for (i in 1:length(wh)-1) {
    exercise_len <- c(exercise_len, wh[i+1] - wh[i])
  }
  exercise_len <- sapply(exercise_len, function(x) {x - 2})
  exercise_len <- c(exercise_len, nrow(df) - sum(exercise_len))
  headers <- raw_lines[wh]
  l <- strsplit(headers, '\\s+|:')
  l <- rep.int(l, exercise_len)
  # Fill in various variables, then return it
  df$user <- sapply(l, function(x) {x[[3]]}, simplify=T)
  df$countries <- sapply(l, function(x) {x[[5]]}, simplify=T)
  df$days <- as.numeric(sapply(l, function(x) {x[[7]]}, simplify=T))
  df$client <- as.factor(sapply(l, function(x) {x[[9]]}, simplify=T))
  df$session <- as.factor(sapply(l, function(x) {x[[11]]}, simplify=T))
  df$format <- as.factor(sapply(l, function(x) {x[[13]]}, simplify=T))
  df$time <- as.numeric(sapply(l, function(x) {x[[15]]}, simplify=T))
  # Extra preprocessing
  df$time[is.na(df$time)] <- 0
  df$time[df$time < 0] <- 0
  df
}
# test_d <- load_data(test_f, F)

train_d <- read.csv('data/train_d.csv', as.is=c(1, 4, 9))
test_d <- read.csv('data/test_d.csv', as.is=c(1, 4, 8))
# write.csv(train_d, 'train_d.csv', row.names=F)
# write.csv(test_d, 'test_d.csv', row.names=F)

# 'Fixing' features
train_d <- train_d[train_d$days > 1,]
train_d$time[train_d$time < 0] <- 0
test_d$time[test_d$time < 0] <- 0
train_d$time[is.na(train_d$time)] <- 0
test_d$time[is.na(test_d$time)] <- 0
train_d$countries <- factor(sapply(strsplit(train_d$countries, split='|', fixed=T), function(x) {x[1]}, simplify=T))
test_d$countries <- factor(sapply(strsplit(test_d$countries, split='|', fixed=T), function(x) {x[1]}, simplify=T))

# rp_control <- rpart.control(minsplit=10L, xval=10)
glm_model <- glm(label ~ pos + depHead + depLabel + days +
                 countries + session + format + time,
                data=train_d,
                family=binomial())
# rp_model <- rpart(label ~ pos + depHead + depLabel + days +
#                   countries + session + format + time,
#                   data=train_d,
#                   control=rp_control)

print(summary(glm_model))

prediction <- predict(glm_model, test_d, type='response')
# prediction[is.na(prediction)] <- 0.0001
out_d <- data.frame(uid=test_d$uid, pred=prediction)
write.table(out_d, 'pred.csv', quote=F, col.names=F, row.names=F)
