
detect <- function(dataset){
  ## If you are using a model, load it like this, it has to be saved under this EXACT name:
  ## load("model.Rdata")
    
  ## Predict event as random guess with 50% probability
  probability <- runif(1)
  event <- probability > 0.5
  
  ## return prediction
  return(event)
}