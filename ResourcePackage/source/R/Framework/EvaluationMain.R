###############################################################################
#                                                                             # 
#      GECCO 2019 Industrial Challenge - Main Evaluation                      #
#                                                                             #  
#      Run this file to test and evaluate your Detector                       #
#                                                                             #
###############################################################################

###############################################################################
### initialize workspace ######################################################
rm(list=ls());
set.seed(2);

baseDir <- getwd()
dataDir  <- paste(baseDir, "Data", sep="/")
submissionDir <- paste(baseDir, "Detectors", sep="/")
librariesDir  <- paste(baseDir, "Lib", sep="/")

setwd(librariesDir)
source("f1score.R")


###############################################################################
### read training data  #######################################################
#timeSeriesData <- data.frame(X1=(runif(n = 100)*100), X2=(runif(n = 100)*100), X3=(runif(n = 100)*100), EVENT=(runif(n = 100)+0.03)>=1, Prediction=NA)
setwd(dataDir)
trainingData <- readRDS(file = "waterDataTraining.RDS")

###############################################################################
### execute and evaluate all detectors ########################################
setwd(submissionDir)
allDetectors <- dir(pattern = "*.R")

completeResult <- NULL

for (submission in allDetectors){ # submission <- allDetectors[6]
  ## Load detector
  source(submission)
  
  ## Run detector
  predictionResult <- rep(NA, nrow(trainingData)) # empty result array
  for (rowIndex in 1:nrow(trainingData)){
    predictionResult[rowIndex] <- detect(dataset = trainingData[rowIndex, -8])
  }
  
  ## Evaluate prediction using F1 score
  result <- calculateScore(observations = trainingData$Event, predictions = predictionResult)
  cat("\nEvaluation finished:\n")
  cat(result$SCORE)
}










