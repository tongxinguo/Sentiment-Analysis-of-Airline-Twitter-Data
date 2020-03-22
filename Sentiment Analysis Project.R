rm(list=ls())
library(tm)
library(e1071)
library(slam)
library(text2vec)

complaint <- read.csv("~/UR/Social Media/complaint1700.csv", 
                      header=TRUE, sep=',', quote='"')
noncomplaint <- read.csv("~/UR/Social Media/noncomplaint1700.csv",
                         header=TRUE, sep=',', quote='"')

#####Combine two dataset############
#set 1 as negative, 0 as non-negative
complaint$sentiment <- 1
noncomplaint$sentiment <- 0
tweets <- rbind(complaint, noncomplaint)
#encode with UTF-8
tweettext <- iconv(tweets$tweet,"WINDOWS-1252","UTF-8") 

#create corpus
docs <- Corpus(VectorSource(tweettext))


#deine preprocessing function and tokenization fucntion using text2vec
it_train = itoken(tweettext,preprocessor = tolower,
                  tokenizer = word_tokenizer,
                  progressbar = F)
stop_words = c("i", "me", "my", "myself", "we", "our", "ours", 
               "ourselves", "you", "your", "yours")  
#build vocabulary with create_vocabulary() function
vocab = create_vocabulary(it_train, stopwords = stop_words) 
# prune vocabulary
pruned_vocab = prune_vocabulary(vocab,  
                                #delete terms with frequency less than 10
                                term_count_min = 10,   
                                doc_proportion_max = 0.5,  
                                doc_proportion_min = 0.001) 


mystop <- c("dont","can","just", "cant", "get", "got",'just',
            'airline','flight','air')
#construct document-term matrix
dtm.control = list(dictionary=pruned_vocab$term, tolower=T, 
                   removePunctuation=T, removeNumbers=T,
                   stopwords=c(mystop,stopwords('english'), stopwords('spanish'),
                               stopwords('portuguese')), 
                   stemming=T, weighting=weightTfIdf)
dtm = DocumentTermMatrix(docs, control=dtm.control)

#convert dtm to matrix
X <- as.matrix(dtm)
#factorize sentiment
Y <- as.numeric(tweets$sentiment)

#seperate into train and test
set.seed(1) 
n=length(tweets$sentiment)
n1=round(n*0.8)
n2=n-n1
train=sample(1:n,n1)

####################################
#######Evaluation function##########
####################################
Evaluation <- function(pred, true, class)
{
  
  tp <- sum( pred==class & true==class)
  fp <- sum( pred==class & true!=class)
  tn <- sum( pred!=class & true!=class)
  fn <- sum( pred!=class & true==class)
  precision <- tp/(tp+fp)
  recall <- tp/(tp+fn)
  F1 <- 2/(1/precision + 1/recall)
  accuracy <- mean(pred==true)
  cat("precision = ", precision, "\n")
  cat("recall = ", recall, "\n")
  cat("F1 = ", F1, "\n")
  cat("accuracy = ", accuracy,"\n")
  
}

###########################################
#######   Support Vector Machine   ########

svm.model <- svm(Y[train] ~ ., data = X[train,], kernel='linear')
pred <- predict( svm.model, X[-train,] )
pred.class <- as.numeric( pred>0.98 ) 
table(pred.class, Y[-train])
Evaluation(pred.class, Y[-train], 1)
Evaluation(pred.class, Y[-train], 0)


###########################################
##########   Naive Bayesion   #############
###########################################
nb.model <- naiveBayes(X[train,], Y[train])
pred1 <- predict(nb.model, X[-train,])
table(pred1, Y[-train])
Evaluation(pred1, Y[-train], 1)
Evaluation(pred1, Y[-train], 0)

#load test data
temp <- read.csv("~/UR/Social Media/temp.csv", 
                 header=TRUE, sep=',', quote='"')
test <- iconv(temp$tweet,"WINDOWS-1252","UTF-8") 
#construct corpus
docs.test <- Corpus(VectorSource(temp$tweet))
it_test = itoken(tweettext,preprocessor = tolower,
                  tokenizer = word_tokenizer,
                  progressbar = F)
# create vocabulary
vocab = create_vocabulary(it_test, stopwords = stop_words)
#prune vocabulary
pruned_vocab = prune_vocabulary(vocab,
                                term_count_min = 10,
                                doc_proportion_max = 0.5,
                                doc_proportion_min = 0.001)
# construct dtm
dtm.control = list(dictionary=pruned_vocab$term, 
                   tolower=T, removePunctuation=T, removeNumbers=T,
                   stopwords=c(stopwords('english'), stopwords('spanish'),
                               stopwords('portuguese')),
                   stemming=T, weighting=weightTfIdf)
dtm.test <- DocumentTermMatrix(docs.test, dtm.control)
# convert dtm to matrix
dtm.test.matrix <- as.matrix(dtm.test)

#prediction using SVM (because of the higher accuracy)
pred <- predict( svm.model, dtm.test.matrix )
pred.class <- as.numeric( pred>0.99 ) 
#pred.test <- predict(svm.model, dtm.test.matrix)
temp$sentiment = pred.class
nrow(temp[temp$sentiment==1,])

#write the output into csv
df <- temp[temp$sentiment==0,]
write.csv(df,file = "~/UR/Social Media/Tongxin_Guo.csv",row.names = F)

