suppressMessages(library(readxl))
suppressMessages(library(caret))
suppressMessages(library(dplyr))
suppressMessages(library(rpart))

rawData <- read_excel("Financial_Dataset.xlsx", sheet = 1)
rawData <- rawData %>%
  mutate(
    Quarter = as.factor(Quarter),
    Credit_Risk_Score = as.factor(Credit_Risk_Score),
    Stock_Trend = as.factor(Stock_Trend)
  )

set.seed(42)
creditRiskModel <- train(
  Credit_Risk_Score ~ Debt_to_Equity + Current_Ratio + EPS + Revenue + Credit_Score,
  data = rawData,
  method = "rpart",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = data.frame(cp = 0.001),
  control = rpart.control(cp = 0.001, minsplit = 20, xval = 10),
  na.action = na.omit
)

creditCases <- data.frame(
  Debt_to_Equity = c(0.5, 2.5, 4.5),
  Current_Ratio = c(1.2, 2.0, 0.8),
  EPS = c(5, 0, -5),
  Revenue = c(100000, 300000, 700000),
  Credit_Score = c(650, 500, 300)
)

print(predict(creditRiskModel, creditCases, type = "prob"))
