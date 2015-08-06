# Wifi Locationing/Internet of Things

## OVERVIEW

#The UJIIndoorLoc database covers three buildings of Universitat Jaume I with 4 or more floors 
#and almost 110.000m2. It can be used for classification, e.g. actual building and floor 
#identification, or regression, e.g. actual longitude and latitude estimation. It was created 
#in 2013 by means of more than 20 different users and 25 Android devices. The database consists of
#19937 training/reference records (trainingData.csv file) and 1111 validation/test records 
#(validationData.csv file). 

#The 529 attributes contain the WiFi fingerprint, the coordinates where it was taken, and other 
#useful information. 

#Each WiFi fingerprint can be characterized by the detected Wireless Access Points (WAPs) and 
#the corresponding Received Signal Strength Intensity (RSSI). The intensity values are represented
#as negative integer values ranging -104dBm (extremely poor signal) to 0dbM. The positive value 
#100 is used to denote when a WAP was not detected. During the database creation, 520 different WAPs were detected. Thus, the WiFi fingerprint is composed by 520 intensity values. 

#Then the coordinates (latitude, longitude, floor) and Building ID are provided as the attributes
#to be predicted. 


## GOALS
# Goals: construct a machine learning predictive model
# for missing values in the validation data (spaceid, relativeposition and userid).

## ENVIRONMENT SET UP 

library(dplyr)
library(class)
library(gmodels)
library(ggplot2)

setwd("~/Documents/INDOOR_LOCALIZATION_PROJ/UJIndoorLoc")

tdata <- read.csv("trainingData.csv", header=TRUE, sep = ",")

dim(tdata)

td <- select(tdata, LONGITUDE:TIMESTAMP)
colnames(td) <-tolower(colnames(td))
write.csv(td, file = "trainingDataSmp.csv")

dim(td)
colnames(td) <-tolower(colnames(td))
head(td, 5)
tail(td, 5)
str(td)
summary(td)

## DATA VISUALIZATION
### Scatterplot showing people locations in the university complex

plot(x = td$longitude, y=td$latitude, main = "Space distribution",
     xlab = "Longitude", ylab ="Latitude")



plot(x = td$userid, y= td$spaceid, type = "p", col = "black", main = "User distribution",
     xlab = "User")
points(td$relativeposition, type ="p", xlab=" ", ylab ="Relative Pos", col ="red")


# Userid, Spaceid, RelativePosition
g1 <- ggplot(data=td, aes(x=td$userid, y=td$spaceid))
g1 <- g1 + geom_point(color = td$userid, fill="blue")
g1 <- g1 + labs(title="Location of each user", x="User Id", y="Space Id")
#g1 <- g1 + geom_point(x=td$relativeposition)
g1 <- g1 + scale_x_discrete(breaks=c(1:18),
                            labels=c("1","2","3","4","5","6","7","8",
                             "9","10","11","12", "13", "14", "15", "16","17", "18"))
g1

plot( x=td$userid, y=td$longitud)

tabla <- data.frame(tabla)
tabla <- cbind(td$userid,td$spaceid)
colnames(tabla) <- c("userid","spaceid")
row.names(tabla) <- NULL

tabla$userid <- as.factor(tabla$userid)
tabla$spaceid <- as.factor(tabla$spaceid)

op <- par(mfrow = c(3, 1))

hist(td$userid, col = "lightblue", border = "black", main= "User Location Pattern",
     xlab="User", ylab="Locations visited")
hist(td$spaceid, col = "blue", border = "black", main = "Locations where users concentrate",
     xlab="SpaceId Locations")
hist(td$relativeposition, col = "darkblue", border = "black", main = "Relative Position Values",
     xlab = "Relative position")


## ATTRIBUTE SELECTION USING WEKA
### CFsSubsetEval algorithm was used to select highly predictive attributes for SPACEID, RELATIVEPOSITION
### and USERID
### Evaluates the worth of a subset of attributes by considering 
### the individual predictive ability of each feature along with the degree of redundancy between
### them.
### A training subset with better predictors was created for each attribute.


attSpaceId <- c(9, 14, 15, 21, 35, 37, 38, 126, 147, 155, 157,168, 178, 179, 180, 195, 197, 214, 231,
                286, 335, 338, 367, 373, 439, 443, 460, 463, 467, 483, 508, 518,525, 527) #525 is target feature

attRelPos <- c(6,8,33,39,40,41,42,43,44,47,48,51,52,63,64,82,84,87,88,89,90,91,101,102,103,104,105,106,107,108,113,114,119,120,123,124,125,126,127,128,129,130,136,137,138,139,
               140,141,143,147,154,155,156,161,162,170,171,172,173,176,177,191,192,224,225,229,249,253,
               258,259,260,261,262,263,268,274,278,282,284,288,295,313,314,315,316,317,318,323,329,
               334,338,340,344,351,369,370,371,372,374,375,380,386,390,394,396,
               400,405,418,428,434,452,456,481,526,529) # 526 is target feature

attUserId <- c(2,18,51,55,100,108,118,135,193,194,195,196,201,202,206,208,209,219,220,224,251,265,
               269,271,273,280,281,286,291,298,302,306,308,336,342,343,359,363,364,373,384,385,388,
               398,404,409,413,415,418,421,432,434,437,446,448,459,460,461,462,463,466,468,469,
               472,476,490,492,493,498,499,503,506,508,513,514,521,523,525,526,527,528) # 527 is target feature


### Creating Subsets

tdSpaceId <- tdata[attSpaceId] #Subset of SpaceID Attribute selection (33 attributes)
tdRelPos <- tdata[attRelPos]   #Subset of RelativePosition Attribute selection (115 attributes)
tdUserId <- tdata[attUserId]   #Subset of UserId Attribute selection (80 attributes)

str(tdSpaceId)
str(tdRelPos)
str(tdUserId)

write.csv(tdSpaceId, file = "tdSpaceId.csv")
write.csv(tdRelPos, file = "tdRelPos.csv")
write.csv(tdUserId, file = "tdUserId.csv")


## Preparing test subsets

## Test dataset for SPACEID

tstdata <- read.csv("validationData.csv", header=TRUE, sep = ",")
tstSpaceId <- tstdata[attSpaceId] # Test Subset of SpaceID Attribute selection (33 attributes)
tstSpaceId$SPACEID <- factor(tstSpaceId$SPACEID)

## K-nearest neighbor 

### Setting K equal to the square root of the number of training examples (sqrt(19937)=141.19)
K0 <- 141

td$spaceid <- factor(td$spaceid)
head(td$spaceid)

### Preparing data for use with KNN

## Min-Max normalization
### X_new = (X -min(X)) / max(X)-min(x)

normalize <- function(x) {
  return((x-min(x))/(max(x)-min(x)))
}

### No normalization is required for tdSpaceId because numeric values go from -101 to 100 
### for all the numeric attributes

summary(tdSpaceId)
max(tdSpaceId)
min(tdSpaceId)

### Factor target feature

tdSpaceId$SPACEID <- factor(tdSpaceId$SPACEID) #123 levels: 123 different locations
str(tdSpaceId)

## Training a model on the data

### We use kNN() function in the class package. It provides a classic implementation
### of the kNN algorithm. For each instance in the test data, the function will identify
### the k-nearest neighbors using Euclidean distance, whe k is a user-specified number (141).
### The test instance is classified by taking a "vote" among the k-nearest neighbors. 
### A tie is broken at random.
### knn(train,test,class,k)

SpaceIdTrain <- tdSpaceId[1:18937,]
SpaceIdTest <- tdSpaceId[18938:19937,]
row.names(SpaceIdTest) <- NULL 

write.csv(SpaceIdTrain, file = "SpaceIdTrain.csv")
write.csv(SpaceIdTest, file = "SpaceIdTest.csv")

### Create target feature training vector and prediction vector

SpaceId_train_labels <- tdSpaceId[1:18937,33]
SpaceId_test_labels <- tdSpaceId[18938:19937,33]

SpaceIdPred <- knn(train=SpaceIdTrain, test=SpaceIdTest, cl=SpaceId_train_labels, k=141)

SpaceIdPred[1:10] # Vector with predicted values for SpaceId in Test set

## Evaluate model performance

ctSpaceId <- CrossTable(x=SpaceId_test_labels, y=SpaceIdPred, prop.chisq=FALSE)

spaceIdEvTbl <- cbind(SpaceId_test_labels,SpaceIdPred)
colnames(spaceIdEvTbl) <- c("real", "predicted")
cbind(diff = 0, spaceIdEvTbl)
EvTb <- data.frame(spaceIdEvTbl)
EvTb$diff <- EvTb$real - EvTb$predicted
EvTb[1:10,]
summary(EvTb)

TP <- EvTb$diff == 0
table(TP) # 700 incorrectly predicted / 300 correctly predicted


## Create a validation dataset

vdata <- read.csv("validationData.csv", header=TRUE, sep = ",")
vd <- select(vdata, LONGITUDE:TIMESTAMP)
colnames(vd) <-tolower(colnames(td))
vd$spaceid <- factor(vd$spaceid)
write.csv(vd, file = "validationDataSmp.csv")

# Working on Userid prediction using linear model

summary(td)
hist(td$userid)
cor(td[c("longitude", "latitude", "floor", "buildingid", "userid", "phoneid", "timestamp")])
userIdModel <- lm(userid ~ longitude + buildingid + timestamp, data=td)

userIdpred <- predict(userIdModel, vd)
userIdModel
summary(userIdModel)

#Correlation for SpaceId on simplified set

hist(td$spaceid)
cor(td[c("longitude", "latitude", "floor", "buildingid", "spaceid", "phoneid", "timestamp")])

#Correlation for Relative Position on simplified set

hist(td$)

