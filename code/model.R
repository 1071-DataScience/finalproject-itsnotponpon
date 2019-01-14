library('gbm3')
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
  pred_f <- paste0(test_f, '.pred')
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
  # Find the position of the exercise headers
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
  ex           <- list()
  ex$user      <- factor(sapply(l, function(x) {x[[3]]}, simplify=T))
  ex$countries <- sapply(l, function(x) {x[[5]]}, simplify=T)
  ex$days      <- as.numeric(sapply(l, function(x) {x[[7]]}, simplify=T))
  ex$client    <- factor(sapply(l, function(x) {x[[9]]}, simplify=T))
  ex$session   <- factor(sapply(l, function(x) {x[[11]]}, simplify=T))
  ex$format    <- factor(sapply(l, function(x) {x[[13]]}, simplify=T))
  ex$time      <- as.numeric(sapply(l, function(x) {x[[15]]}, simplify=T))
  exercises <- as.data.frame(ex, stringsAsFactors=F)

  cbind(exercises, df_part)
}
# After another round of optimization we can read original files with acceptable speed
train_d <- load_data(train_f, is_train=T)
test_d <- load_data(test_f, is_train=F)

# 'Cleaning' features, including:
# * filter out less than 1 day of usage (user needs to familiarize the app)
# * Lowercase tokens
# * Split exercise and token indices
# * Replacing negative and `NA` time with zero (not the best, better guess them)
train_d <- train_d[train_d$days > 1,]
train_d$token <- factor(tolower(train_d$token))
train_d$exerciseId <- factor(substr(train_d$uid, 1, 8))
train_d$exerciseIndex <- as.numeric(substr(train_d$uid, 9, 10))
train_d$tokenIndex <- as.numeric(substr(train_d$uid, 11, 12))
train_d$time[is.na(train_d$time)] <- median(train_d$time, na.rm=T)
train_d$time[train_d$time < 0] <- median(train_d$time)
train_d$countries <- factor(sapply(strsplit(train_d$countries, split='|', fixed=T), function(x) {x[1]}, simplify=T))

test_d$token <- factor(tolower(test_d$token))
test_d$exerciseId <- factor(substr(test_d$uid, 1, 8))
test_d$exerciseIndex <- as.numeric(substr(test_d$uid, 9, 10))
test_d$tokenIndex <- as.numeric(substr(test_d$uid, 11, 12))
test_d$time[is.na(test_d$time)] <- median(test_d$time, na.rm=T)
test_d$time[test_d$time < 0] <- median(test_d$time)
test_d$countries <- factor(sapply(strsplit(test_d$countries, split='|', fixed=T), function(x) {x[1]}, simplify=T))

# train_params <- training_params(num_trees=6400,
#                                 interaction_depth=3,
#                                 shrinkage=0.001)
# gbmt_model <- gbmt(label ~ depLabel +
#                 days + time +
#                 tokenIndex + exerciseIndex,
#                 distribution=gbm_dist('Bernoulli'),
#                 data=train_d,
#                 train_params=train_params,
#                 cv_folds=25,
#                 is_verbose=F)
glm_model <- glm(label ~ depLabel + client *
                session * format + days * time +
                tokenIndex * exerciseIndex,
                data=train_d,
                family=binomial())
# best_iter <- gbmt_performance(gbmt_model, method='cv')
# print(best_iter)
# print(summary(gbmt_model, num_trees=best_iter))
print(summary(glm_model))
prediction <- predict(glm_model, test_d, type='response')
out_d <- data.frame(uid=test_d$uid, pred=prediction)
write.table(out_d, pred_f, quote=F, col.names=F, row.names=F)
