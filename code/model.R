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

# The dataset format is almost SSV, but not quite
load_data <- function(filename, is_train) {
  raw_lines <- readLines(filename)
  # Read the SSV part first
  # `uid`, `token`, and `morph` is kept as string for preprocessing
  df_part <- read.table(filename, header=F, as.is=c(1, 2, 4))
  if (is_train) {
    colnames(df_part) <- c('uid', 'token', 'pos', 'morph', 'depLabel', 'depHead', 'label')
  } else {
    colnames(df_part) <- c('uid', 'token', 'pos', 'morph', 'depLabel', 'depHead')
  }
  # Filter out the exercise header
  wh <- which(startsWith(raw_lines, '#'))
  # Count where they originally belong
  exercise_len <- vector(mode='integer', length=length(wh))
  for (i in 1:length(wh)-1) {
    exercise_len[i] <- wh[i+1] - wh[i]
  }
  exercise_len <- exercise_len - 2
  exercise_len[i+1] <- nrow(df_part) - sum(exercise_len) - 2
  # Read and replicate headers
  headers <- raw_lines[wh]
  l <- strsplit(headers, '\\s+|:')
  l <- rep.int(l, exercise_len)
  # Fill in various variables, then return it
  # l[[1]] is '#', even numbers are feature names
  ex_user      <- sapply(l, function(x) {x[[3]]}, simplify=T)
  ex_countries <- sapply(l, function(x) {x[[5]]}, simplify=T)
  ex_days      <- as.numeric(sapply(l, function(x) {x[[7]]}, simplify=T))
  ex_client    <- as.factor(sapply(l, function(x) {x[[9]]}, simplify=T))
  ex_session   <- as.factor(sapply(l, function(x) {x[[11]]}, simplify=T))
  ex_format    <- as.factor(sapply(l, function(x) {x[[13]]}, simplify=T))
  ex_time      <- as.numeric(sapply(l, function(x) {x[[15]]}, simplify=T))
  exercises <- data.frame(user=ex_user,
                          countries=ex_countries,
                          days=ex_days,
                          client=ex_client,
                          session=ex_session,
                          format=ex_format,
                          time=ex_time)

  cbind(exercises, df_part)
}
# After another round of optimization we can read original files with acceptable speed
train_d <- load_data(train_f, is_train=T)
test_d <- load_data(test_f, is_train=F)

# 'Fixing' features, including:
# * filter out less than 1 day of usage
# * Lowercase tokens
# * Split exercise and token indices
# * Replacing negative and `NA` time with zero (not the best, better guess them)
train_d <- train_d[train_d$days > 1,]
train_d$token <- factor(tolower(train_d$token))
train_d$exerciseId <- factor(sapply(train_d$uid, function(x) {substr(x, 1, 8)}))
train_d$exerciseIndex <- as.numeric(sapply(train_d$uid, function(x) {substr(x, 9, 10)}))
train_d$tokenIndex <- as.numeric(sapply(train_d$uid, function(x) {substr(x, 11, 12)}))
train_d$time[train_d$time < 0] <- 0
train_d$time[is.na(train_d$time)] <- 0
# train_d$countries <- factor(sapply(strsplit(train_d$countries, split='|', fixed=T), function(x) {x[1]}, simplify=T))

test_d$token <- factor(tolower(test_d$token))
test_d$exerciseId <- factor(sapply(test_d$uid, function(x) {substr(x, 1, 8)}))
test_d$exerciseIndex <- as.numeric(sapply(test_d$uid, function(x) {substr(x, 9, 10)}))
test_d$tokenIndex <- as.numeric(sapply(test_d$uid, function(x) {substr(x, 11, 12)}))
test_d$time[test_d$time < 0] <- 0
test_d$time[is.na(test_d$time)] <- 0
# test_d$countries <- factor(sapply(strsplit(test_d$countries, split='|', fixed=T), function(x) {x[1]}, simplify=T))

# rpart model (currently unused)
# rp_control <- rpart.control(minsplit=10L, xval=10)
# rp_model <- rpart(label ~ pos + depHead + depLabel + days +
#                   countries + session + format + time,
#                   data=train_d,
#                   control=rp_control)

glm_model <- glm(label ~ pos + depHead + depLabel + days +
                 session + format + time +
                tokenIndex + exerciseIndex,
                data=train_d,
                family=binomial())

# print(summary(glm_model))

prediction <- predict(glm_model, test_d, type='response')
# prediction[is.na(prediction)] <- 0.0001
out_d <- data.frame(uid=test_d$uid, pred=prediction)
write.table(out_d, pred_f, quote=F, col.names=F, row.names=F)
