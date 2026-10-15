library(readxl)
library(dplyr)
rawData <- read_excel("Financial_Dataset.xlsx", sheet=1)
rawData <- rawData %>% mutate(Quarter = as.factor(Quarter), Credit_Risk_Score = as.factor(Credit_Risk_Score))
print(c("All Quarters", sort(unique(as.character(rawData$Quarter)))))
print(c("All", sort(unique(as.character(rawData$Credit_Risk_Score)))))

