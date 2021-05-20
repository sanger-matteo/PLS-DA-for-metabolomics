# if (!requireNamespace("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")

# BiocManager::install("mixOmics")


split_data <- function(dfX, dfY, nn){
  #' Perform Multilevel PLS-DA
  
  outDF <- multilevel(X,  Y = NULL, design,   ncomp = nn, keepX = NULL, keepY = NULL,   
                      method = c("splsda"),   mode = c("regression"),  
                      max.iter = 500,  tol = 1e-06, near.zero.var = FALSE)
  
  return(outDF)
}