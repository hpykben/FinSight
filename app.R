library(shiny)
library(shinyjs)
library(ggplot2)
library(dplyr)
library(readxl)
library(caret)
library(rpart)
library(DT)

# Load dataset
rawData <- read_excel("Financial_Dataset.xlsx", sheet = 1)
rawData <- rawData %>%
  mutate(
    Quarter = as.factor(Quarter),
    Credit_Risk_Score = as.factor(Credit_Risk_Score),
    Stock_Trend = as.factor(Stock_Trend)
  )

quarterChoices <- c("All Quarters", sort(unique(as.character(rawData$Quarter))))
creditRiskChoices <- c("All", sort(unique(as.character(rawData$Credit_Risk_Score))))

# ── Train Credit Risk Prediction Model ────────────────────────────────────────
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

# ── Train Stock Trend Prediction Model ────────────────────────────────────────
stockTrendModel <- train(
  Stock_Trend ~ EPS + Stock_Volatility + Market_Cap,
  data = rawData,
  method = "rpart",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = data.frame(cp = 0.001),
  control = rpart.control(cp = 0.001, minsplit = 20, xval = 10),
  na.action = na.omit
)

# ── SVG icon helper ───────────────────────────────────────────────────────────
svgIcon <- function(name) {
  icons <- list(
    dashboard = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M3 3h8v8H3zm0 10h8v8H3zm10-10h8v8h-8zm0 10h8v8h-8z"/></svg>',
    shield    = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M11.488 21.754c.294.157.663.156.957-.001 8.012-4.304 8.581-12.713 8.574-15.104a.988.988 0 0 0-.596-.903l-8.05-3.566a1.005 1.005 0 0 0-.813.001L3.566 5.747a.99.99 0 0 0-.592.892c-.034 2.379.445 10.806 8.514 15.115zM8.674 10.293l2.293 2.293 4.293-4.293 1.414 1.414-5.707 5.707-3.707-3.707 1.414-1.414z"></path></svg>',
    stats     = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M20 12a2 2 0 0 0-.703.133l-2.398-1.963c.059-.214.101-.436.101-.67C17 8.114 15.886 7 14.5 7S12 8.114 12 9.5c0 .396.1.765.262 1.097l-2.909 3.438A2.06 2.06 0 0 0 9 14c-.179 0-.348.03-.512.074l-2.563-2.563C5.97 11.348 6 11.179 6 11c0-1.108-.892-2-2-2s-2 .892-2 2 .892 2 2 2c.179 0 .348-.03.512-.074l2.563 2.563A1.906 1.906 0 0 0 7 16c0 1.108.892 2 2 2s2-.892 2-2c0-.237-.048-.46-.123-.671l2.913-3.442c.227.066.462.113.71.113a2.48 2.48 0 0 0 1.133-.281l2.399 1.963A2.077 2.077 0 0 0 18 14c0 1.108.892 2 2 2s2-.892 2-2-.892-2-2-2z"></path></svg>',
    financial = '<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="currentColor"><path d="M320-414v-306h120v306l-60-56-60 56Zm200 60v-526h120v406L520-354ZM120-216v-344h120v224L120-216Zm0 98 258-258 142 122 224-224h-64v-80h200v200h-80v-64L524-146 382-268 232-118H120Z"/></svg>',
    sliders   = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M3 17v2h6v-2H3zM3 5v2h10V5H3zm10 16v-2h8v-2h-8v-2h-2v6h2zM7 9v2H3v2h4v2h2V9H7zm14 4v-2H11v2h10zm-6-4h2V7h4V5h-4V3h-2v6z"/></svg>',
    home      = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z"/></svg>',
    moon      = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M12 3a9 9 0 1 0 9 9c0-.46-.04-.92-.1-1.36a5.389 5.389 0 0 1-4.4 2.26 5.403 5.403 0 0 1-3.14-9.8c-.44-.06-.9-.1-1.36-.1z"/></svg>',
    sun       = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M6.76 4.84l-1.8-1.79-1.41 1.41 1.79 1.79zM4 10.5H1v2h3zm9-9.95h-2V3.5h2zm7.45 3.91l-1.41-1.41-1.79 1.79 1.41 1.41zm-3.21 13.7l1.79 1.8 1.41-1.41-1.8-1.79zM20 10.5v2h3v-2zm-8-5c-3.31 0-6 2.69-6 6s2.69 6 6 6 6-2.69 6-6-2.69-6-6-6zm-1 16.95h2V19.5h-2zm-7.45-3.91l1.41 1.41 1.79-1.8-1.41-1.41z"/></svg>',
    chevron   = '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><path d="M15.41 7.41L14 6l-6 6 6 6 1.41-1.41L10.83 12z"/></svg>'
    )
  HTML(icons[[name]])
}
# ── SVG icons for Dashboard ───────────────────────────────────────────
dashboardIcon <- function(name) {
  icons <- list(
    buildings = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" style="fill: rgba(122, 30, 16, 1);transform: ;msFilter:;"><path d="M19 2H9c-1.103 0-2 .897-2 2v6H5c-1.103 0-2 .897-2 2v9a1 1 0 0 0 1 1h16a1 1 0 0 0 1-1V4c0-1.103-.897-2-2-2zM5 12h6v8H5v-8zm14 8h-6v-8c0-1.103-.897-2-2-2H9V4h10v16z"></path><path d="M11 6h2v2h-2zm4 0h2v2h-2zm0 4.031h2V12h-2zM15 14h2v2h-2zm-8 .001h2v2H7z"></path></svg>',
    trending = '<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#7a1e10"><path d="M200-120q-33 0-56.5-23.5T120-200v-640h80v640h640v80H200Zm40-120v-360h160v360H240Zm200 0v-560h160v560H440Zm200 0v-200h160v200H640Z"/></svg>',
    info      = '<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#7a1e10"><path d="M160-640h640v-80H160v80Zm-80-80q0-33 23.5-56.5T160-800h640q33 0 56.5 23.5T880-720v240H160v240h164v80H160q-33 0-56.5-23.5T80-240v-480ZM598-80 428-250l56-56 114 112 226-226 56 58L598-80ZM160-720v480-180 113-413Z"/></svg>',
    profit      = '<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#7a1e10"><path d="M441-120v-86q-53-12-91.5-46T293-348l74-30q15 48 44.5 73t77.5 25q41 0 69.5-18.5T587-356q0-35-22-55.5T463-458q-86-27-118-64.5T313-614q0-65 42-101t86-41v-84h80v84q50 8 82.5 36.5T651-650l-74 32q-12-32-34-48t-60-16q-44 0-67 19.5T393-614q0 33 30 52t104 40q69 20 104.5 63.5T667-358q0 71-42 108t-104 46v84h-80Z"/></svg>'
  )
  HTML(icons[[name]])
}
# ── SVG icons for Financial ───────────────────────────────────────────
financialIcon <- function(name) {
  icons <- list(
    money = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" style="fill: rgba(122, 30, 16, 1);transform: ;msFilter:;"><path d="M21 4H3a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1h18a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1zm-1 11a3 3 0 0 0-3 3H7a3 3 0 0 0-3-3V9a3 3 0 0 0 3-3h10a3 3 0 0 0 3 3v6z"></path><path d="M12 8c-2.206 0-4 1.794-4 4s1.794 4 4 4 4-1.794 4-4-1.794-4-4-4zm0 6c-1.103 0-2-.897-2-2s.897-2 2-2 2 .897 2 2-.897 2-2 2z"></path></svg>',
    analyse      = '<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#7a1e10"><path d="M300-520q-58 0-99-41t-41-99q0-58 41-99t99-41q58 0 99 41t41 99q0 58-41 99t-99 41Zm0-80q25 0 42.5-17.5T360-660q0-25-17.5-42.5T300-720q-25 0-42.5 17.5T240-660q0 25 17.5 42.5T300-600Zm360 440q-58 0-99-41t-41-99q0-58 41-99t99-41q58 0 99 41t41 99q0 58-41 99t-99 41Zm42.5-97.5Q720-275 720-300t-17.5-42.5Q685-360 660-360t-42.5 17.5Q600-325 600-300t17.5 42.5Q635-240 660-240t42.5-17.5ZM216-160l-56-56 584-584 56 56-584 584Z"/></svg>',
    linechart = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" style="fill: rgba(122, 30, 16, 1);transform: ;msFilter:;"><path d="M12 2C6.486 2 2 6.486 2 12s4.486 10 10 10 10-4.486 10-10S17.514 2 12 2zm0 18c-4.411 0-8-3.589-8-8s3.589-8 8-8 8 3.589 8 8-3.589 8-8 8z"></path><path d="M12 5.166V12h6.834A6.817 6.817 0 0 0 12 5.166z"></path></svg>'
  )
  HTML(icons[[name]])
}
# ── SVG icons for Risk Analysis ───────────────────────────────────────────
riskIcon <- function(name) {
  icons <- list(
    shield = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" style="fill: rgba(122, 30, 16, 1);transform: ;msFilter:;"><path d="m20.48 6.105-8-4a1 1 0 0 0-.895 0l-8 4a1.002 1.002 0 0 0-.547.795c-.011.107-.961 10.767 8.589 15.014a.99.99 0 0 0 .812 0c9.55-4.247 8.6-14.906 8.589-15.014a1.001 1.001 0 0 0-.548-.795zm-8.447 13.792C5.265 16.625 4.944 9.642 4.999 7.635l7.034-3.517 7.029 3.515c.038 1.989-.328 9.018-7.029 12.264z"></path><path d="M14.293 8.293 12 10.586 9.707 8.293 8.293 9.707 10.586 12l-2.293 2.293 1.414 1.414L12 13.414l2.293 2.293 1.414-1.414L13.414 12l2.293-2.293z"></path></svg>',
    chart = '<svg xmlns="http://www.w3.org/2000/svg" height="24" width="24" viewBox="0 0 640 640"><!--!Font Awesome Free v7.2.0 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2026 Fonticons, Inc.--><path fill="#7a1e10" d="M519.8 126.4C536.6 120.8 545.6 102.7 540 85.9C534.4 69.1 516.3 60.1 499.6 65.6L386.6 103.3C372.7 79.8 347 64 317.7 64C273.5 64 237.7 99.8 237.7 144C237.7 147 237.9 149.9 238.2 152.8L115.6 193.6C98.8 199.2 89.8 217.3 95.4 234.1C101 250.9 119.1 259.9 135.9 254.3L271.4 209.1C275.9 212.3 280.7 215 285.8 217.3L285.8 544C285.8 561.7 300.1 576 317.8 576L509.8 576C527.5 576 541.8 561.7 541.8 544C541.8 526.3 527.5 512 509.8 512L349.8 512L349.8 217.3C370.8 208.1 387 190.3 394 168.3L519.9 126.3zM437.3 352L509.7 227.8L582.1 352L437.2 352zM509.7 448C572.6 448 624.9 414 635.7 369.1C638.3 358.1 634.7 346.8 629 337L533.8 173.8C528.8 165.2 519.6 160 509.7 160C499.8 160 490.6 165.3 485.6 173.8L390.4 337.1C384.7 346.9 381.1 358.2 383.7 369.2C394.5 414 446.8 448.1 509.7 448.1zM126.8 355.8L199.2 480L54.3 480L126.7 355.8zM.9 497.1C11.7 542 64 576 126.8 576C189.6 576 242 542 252.8 497.1C255.4 486.1 251.8 474.8 246.1 465L150.9 301.8C145.9 293.2 136.7 288 126.8 288C116.9 288 107.7 293.3 102.7 301.8L7.6 465.1C1.9 474.9-1.7 486.2 .9 497.2z"/></svg>',
    error = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" style="fill: rgba(122, 30, 16, 1);transform: ;msFilter:;"><path d="M11.001 10h2v5h-2zM11 16h2v2h-2z"></path><path d="M13.768 4.2C13.42 3.545 12.742 3.138 12 3.138s-1.42.407-1.768 1.063L2.894 18.064a1.986 1.986 0 0 0 .054 1.968A1.984 1.984 0 0 0 4.661 21h14.678c.708 0 1.349-.362 1.714-.968a1.989 1.989 0 0 0 .054-1.968L13.768 4.2zM4.661 19 12 5.137 19.344 19H4.661z"></path></svg>'
    )
  HTML(icons[[name]])
}
# ── SVG icons for Risk Analysis ───────────────────────────────────────────
stockIcon <- function(name) {
  icons <- list(
    pulse = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" style="fill: rgba(122, 30, 16, 1);transform: ;msFilter:;"><path d="M16.97 4.757a.999.999 0 0 0-1.918-.073l-3.186 9.554-2.952-6.644a1.002 1.002 0 0 0-1.843.034L5.323 12H2v2h3.323c.823 0 1.552-.494 1.856-1.257l.869-2.172 3.037 6.835c.162.363.521.594.915.594l.048-.001a.998.998 0 0 0 .9-.683l2.914-8.742.979 3.911A1.995 1.995 0 0 0 18.781 14H22v-2h-3.22l-1.81-7.243z"></path></svg>',
    uptrend = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" style="fill: rgba(122, 30, 16, 1);transform: ;msFilter:;"><path d="m10 10.414 4 4 5.707-5.707L22 11V5h-6l2.293 2.293L14 11.586l-4-4-7.707 7.707 1.414 1.414z"></path></svg>'
    )
  HTML(icons[[name]])
}

# ── SVG icons for About page cards ───────────────────────────────────────────
aboutCardIcon <- function(type) {
  svgs <- list(
    shield = '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><path d="M11.488 21.754c.294.157.663.156.957-.001 8.012-4.304 8.581-12.713 8.574-15.104a.988.988 0 0 0-.596-.903l-8.05-3.566a1.005 1.005 0 0 0-.813.001L3.566 5.747a.99.99 0 0 0-.592.892c-.034 2.379.445 10.806 8.514 15.115zM8.674 10.293l2.293 2.293 4.293-4.293 1.414 1.414-5.707 5.707-3.707-3.707 1.414-1.414z"/></svg>',
    chart  = '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><path d="M3.5 18.49l6-6.01 4 4L22 6.92l-1.41-1.41-7.09 7.97-4-4L2 16.99z"/></svg>',
    data   = '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 4.02 2 6.5v11C2 19.98 6.48 22 12 22s10-2.02 10-4.5v-11C22 4.02 17.52 2 12 2zm0 2c4.42 0 8 1.57 8 3.5S16.42 11 12 11 4 9.43 4 7.5 7.58 4 12 4zm0 16c-4.42 0-8-1.57-8-3.5v-2.06c1.74 1.28 4.71 2.06 8 2.06s6.26-.78 8-2.06V16.5c0 1.93-3.58 3.5-8 3.5zm0-5c-4.42 0-8-1.57-8-3.5v-2.06C5.74 10.72 8.71 11.5 12 11.5s6.26-.78 8-2.06V12.5c0 1.93-3.58 3.5-8 3.5z"/></svg>',
    link   = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M19 19H5V5h7V3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7h-2v7zM14 3v2h3.59l-9.83 9.83 1.41 1.41L19 6.41V10h2V3h-7z"/></svg>'
  )
  HTML(svgs[[type]])
}

# ── Helper tag builders ───────────────────────────────────────────────────────
navItem <- function(page, iconName, label, active = FALSE) {
  tags$li(class = "nav-item",
    tags$a(href = "#", `data-page` = page,
           class = if (active) "active" else "",
      tags$span(class = "nav-icon", svgIcon(iconName)),
      tags$span(class = "nav-label", label)
    )
  )
}

# ── About Page Content ────────────────────────────────────────────────────────
aboutPageContent <- function() {
  tags$div(class = "about-section",

    # Intro card
    tags$div(class = "about-intro",
      tags$h2("Financial Intelligence Platform"),
      tags$p(
        "FinSight is a sophisticated data analytics dashboard designed to transform raw financial data ",
        "into actionable strategic intelligence. It provides a multi-dimensional view of corporate health, ",
        "combining historical performance metrics with forward-looking risk assessments and market trends."
      )
    ),

    # Problems FinSight Addresses
    tags$h2(class = "section-title", "Problems FinSight Addresses"),
    tags$div(class = "about-grid",

      # Card 1 – Risk Mitigation
      tags$div(class = "about-card",
        tags$div(class = "about-icon", aboutCardIcon("shield")),
        tags$h3("Risk Mitigation"),
        tags$p(
          "Quantifies ", tags$strong("Credit & Liquidity Risk"),
          " by analyzing Debt-to-Equity and Current Ratios, allowing stakeholders to identify ",
          "potential defaults before they occur."
        )
      ),

      # Card 2 – Anomaly Detection
      tags$div(class = "about-card",
        tags$div(class = "about-icon", aboutCardIcon("chart")),
        tags$h3("Anomaly Detection"),
        tags$p(
          "Identify unusual ", tags$strong("Financial Patterns"),
          " including audit issues, fraud indicators, market shocks, and policy changes."
        )
      ),

      # Card 3 – Financial Analysis
      tags$div(class = "about-card",
        tags$div(class = "about-icon", aboutCardIcon("chart")),
        tags$h3("Financial Analysis"),
        tags$p(
          "Analyze comprehensive ", tags$strong("Accounting Metrics"),
          " including assets, liabilities, equity, cash flow, and profitability ratios."
        )
      )
    ),

    # Dataset card
    tags$div(class = "dataset-card",
      tags$div(class = "dataset-header",
        aboutCardIcon("data"),
        tags$span("Dataset Overview")
      ),
      tags$h3(class = "dataset-title", "Financial Forecasting Dataset (2015\u20132024)"),
      tags$p(class = "dataset-desc",
        "This dataset contains structured and semi-structured financial data for multiple companies ",
        "over a 10-year period. It combines corporate financial indicators, stock market metrics, ",
        "and credit risk scores, reflecting the diverse and complex nature of real-world financial analysis."
      ),
      tags$div(class = "dataset-meta",
        tags$span(tags$strong("Time Period:"), " 2015\u20132024 (10 years)"),
        tags$span(class = "meta-divider", "|"),
        tags$span(tags$strong("Update Frequency:"), " Quarterly")
      ),
      tags$a(
        href   = "https://www.kaggle.com/datasets/programmer3/financial-forecasting-dataset",
        target = "_blank",
        class  = "dataset-btn",
        aboutCardIcon("link"),
        " View Dataset"
      )
    ),
    # Our Team
    tags$h2(class = "team-section-title", HTML("Our<br>Team")),
    tags$div(class = "team-grid",

      # Jay Lumanta
      tags$div(class = "team-card",
        tags$div(class = "team-hex-wrapper",
          tags$div(class = "team-hex-shadow"),
          tags$div(class = "team-hex-photo jay", onclick = "openTeamModal('jay')",
            tags$img(src = "jay.png", alt = "Jay Lumanta"),
            tags$div(class = "hex-overlay",
              tags$div(class = "hex-overlay-content",
                tags$div(class = "overlay-name", "Jay Lumanta"),
                tags$div(class = "overlay-role", "UI Designer"),
                tags$div(class = "overlay-desc", "Responsible for the visual design and user experience of FinSight.")
              )
            )
          ),
          tags$div(class = "team-hex-badge jay",
            HTML('<svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20"><path d="M9.25 22L8.85 18.8Q8.625 18.7 8.388 18.562Q8.15 18.425 7.95 18.275L4.975 19.525L2.225 14.975L4.8 13.1Q4.775 12.975 4.775 12.862V12.137Q4.775 12.025 4.8 11.9L2.225 10.025L4.975 5.475L7.95 6.725Q8.15 6.575 8.388 6.438Q8.625 6.3 8.85 6.2L9.25 3H14.75L15.15 6.2Q15.375 6.3 15.612 6.438Q15.85 6.575 16.05 6.725L19.025 5.475L21.775 10.025L19.2 11.9Q19.225 12.025 19.225 12.137V12.862Q19.225 12.975 19.175 13.1L21.75 14.975L19 19.525L16.05 18.275Q15.85 18.425 15.612 18.562Q15.375 18.7 15.15 18.8L14.75 22ZM12 15.5Q13.458 15.5 14.479 14.479Q15.5 13.458 15.5 12Q15.5 10.542 14.479 9.521Q13.458 8.5 12 8.5Q10.525 8.5 9.512 9.521Q8.5 10.542 8.5 12Q8.5 13.458 9.512 14.479Q10.525 15.5 12 15.5Z"/></svg>')
          )
        ),
        tags$div(class = "team-name", "Jay Lumanta"),
        tags$div(class = "team-role", "UI Designer")
      ),

      # Mariel Rossini Go
      tags$div(class = "team-card",
        tags$div(class = "team-hex-wrapper",
          tags$div(class = "team-hex-shadow"),
          tags$div(class = "team-hex-photo mariel", onclick = "openTeamModal('mariel')",
          tags$img(src = "rossini.png", alt = "Mariel Rossini Go"),
            tags$div(class = "hex-overlay",
              tags$div(class = "hex-overlay-content",
                tags$div(class = "overlay-name", "Mariel Rossini Go"),
                tags$div(class = "overlay-role", "Data Analyst"),
                tags$div(class = "overlay-desc", "Handles data processing, analysis, and insight generation for the platform.")
              )
            )
          ),
          tags$div(class = "team-hex-badge mariel",
            HTML('<svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20"><path d="M19 3H5C3.9 3 3 3.9 3 5V19C3 20.1 3.9 21 5 21H19C20.1 21 21 20.1 21 19V5C21 3.9 20.1 3 19 3M9 17H7V10H9V17M13 17H11V7H13V17M17 17H15V13H17V17Z"/></svg>')
          )
        ),
        tags$div(class = "team-name", "Mariel Rossini Go"),
        tags$div(class = "team-role", "Data Analyst")
      ),

      # Ralph Benedict Moreno
      tags$div(class = "team-card",
        tags$div(class = "team-hex-wrapper",
          tags$div(class = "team-hex-shadow"),
          tags$div(class = "team-hex-photo ralph", onclick = "openTeamModal('ralph')",
          tags$img(src = "ralph.png", alt = "Ralph Benedict Moreno"),
            tags$div(class = "hex-overlay",
              tags$div(class = "hex-overlay-content",
                tags$div(class = "overlay-name", "Ralph Benedict Moreno"),
                tags$div(class = "overlay-role", "Programmer"),
                tags$div(class = "overlay-desc", "Builds and maintains the backend logic and Shiny application architecture.")
              )
            )
          ),
          tags$div(class = "team-hex-badge ralph",
            HTML('<svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20"><path d="M8 3a2 2 0 0 0-2 2v4a2 2 0 0 1-2 2H3v2h1a2 2 0 0 1 2 2v4a2 2 0 0 0 2 2h2v-2H8v-5a2 2 0 0 0-2-2 2 2 0 0 0 2-2V5h2V3m6 0a2 2 0 0 1 2 2v4a2 2 0 0 0 2 2h1v2h-1a2 2 0 0 0-2 2v4a2 2 0 0 1-2 2h-2v-2h2v-5a2 2 0 0 1 2-2 2 2 0 0 1-2-2V5h-2V3z"/></svg>')
          )
        ),
        tags$div(class = "team-name", "Ralph Benedict Moreno"),
        tags$div(class = "team-role", "Programmer")
      )
    )
  )
}

# ── UI ────────────────────────────────────────────────────────────────────────
ui <- fluidPage(

  useShinyjs(),

  tags$head(
  tags$link(rel = "stylesheet",
            href = "https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap"),
    tags$style(HTML("

      /* ── Reset & base ─────────────────────────────── */
      * {
        font-family: 'Poppins', sans-serif;
        margin: 0; padding: 0; box-sizing: border-box;
      }

      :root {
        --body-color:          #FCF8F2;
        --sidebar-color:       #fff;
        --primary-color:       #7A1E10;
        --primary-color-light: #f6f5ff;
        --toggle-color:        #ddd;
        --text-muted:          #A0785A;
        --tran-03: all 0.3s ease;
        --tran-05: all 0.5s ease;
      }

      body.dark-mode {
        --body-color:          #18191A;
        --sidebar-color:       #252624;
        --primary-color:       #7A1E10;
        --primary-color-light: #3a3b3c;
        --toggle-color:        #ddd;
      }

      body {
        height: 100vh;
        background: var(--body-color);
        transition: var(--tran-05);
        overflow-x: hidden;
      }

      /* ── Sidebar ──────────────────────────────────── */
      #sidebar {
        position: fixed;
        left: 0; top: 0; bottom: 0;
        height: 100%;
        width: 250px;
        padding: 10px 14px;
        background: var(--sidebar-color);
        transition: var(--tran-05);
        z-index: 100;
        box-shadow: 0px 4px 15px 0px rgba(0, 0, 0, 0.3);
        display: flex;
        flex-direction: column;
        overflow: visible;
      }

      #sidebar.close { width: 88px; }

      #sidebar.close .brand-text,
      #sidebar.close .nav-label,
      #sidebar.close .mode-label {
        overflow: hidden;
        width: 0;
        opacity: 0;
        pointer-events: none;
      }

      /* ── Header / brand ───────────────────────────── */
      .sidebar-header {
        position: relative;
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 6px 0 0 0;
        min-height: 60px;
      }

      .brand-logo {
        min-width: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
      }

      .brand-logo img {
        width: 40px; height: 40px;
        border-radius: 8px;
        object-fit: contain;
      }

      .brand-text {
        display: flex;
        flex-direction: column;
        overflow: hidden;
        transition: var(--tran-03);
        opacity: 1;
        white-space: nowrap;
      }

      #sidebar.close .brand-text { opacity: 0; width: 0; }

      .brand-name {
        font-size: 18px;
        font-weight: 700;
        color: #d1beaf;
        line-height: 1;
        margin-bottom: 6px;
      }

      .brand-name .highlight { color: #378ADD; }

      .brand-tagline {
        font-size: 8px;
        font-weight: 400;
        color: #575757;
        letter-spacing: 0.5px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 100%;
      }

      body.dark-mode .brand-tagline { color: #fff; }

      /* toggle chevron */
      .sidebar-toggle {
        position: absolute;
        top: 50%; right: -25px;
        transform: translateY(-50%);
        width: 25px; height: 25px;
        background: var(--primary-color);
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        color: #ccc;
        cursor: pointer;
        transition: var(--tran-03);
        z-index: 101;
      }

      #sidebar.close .sidebar-toggle { transform: translateY(-50%) rotate(180deg); }

      /* ── Menu ─────────────────────────────────────── */
      .menu-section {
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        margin-top: 20px;
        overflow: hidden;
      }

      .nav-list { list-style: none; margin-top: 15px; }

      .nav-item { height: 50px; margin-top: 8px; }

      .nav-item a {
        height: 100%;
        width: 100%;
        display: flex;
        align-items: center;
        gap: 15px;
        text-decoration: none;
        border-radius: 6px;
        transition: background 0.25s ease;
        cursor: pointer;
        overflow: hidden;
      }

      .nav-item a:hover,
      .nav-item a.active { background: #FDF0DC; }

      .nav-icon {
        min-width: 60px;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
        color: var(--text-muted);
        transition: color 0.2s ease;
      }

      .nav-icon svg { display: block; }

      .nav-item a:hover .nav-icon,
      .nav-item a.active .nav-icon { color: var(--primary-color); }

      .nav-label {
        font-size: 16px;
        font-weight: 500;
        color: var(--text-muted);
        white-space: nowrap;
        opacity: 1;
        transition: opacity 0.3s ease, color 0.2s ease;
      }

      .nav-item a:hover .nav-label,
      .nav-item a.active .nav-label {
        color: var(--primary-color);
        font-weight: bold;
      }

      #sidebar.close .nav-label { opacity: 0; }

      /* ── Bottom area ──────────────────────────────── */
      .bottom-section { padding-bottom: 8px; }

      .mode-row {
        position: relative;
        height: 50px;
        margin-top: 8px;
        border-radius: 6px;
        background: var(--primary-color-light);
        display: flex;
        align-items: center;
        cursor: pointer;
        overflow: hidden;
      }

      .mode-icons {
        min-width: 60px;
        display: flex; align-items: center; justify-content: center;
        position: relative;
        height: 100%;
        color: var(--text-muted);
      }

      .mode-icons .icon-sun,
      .mode-icons .icon-moon {
        position: absolute;
        transition: opacity 0.3s ease;
        display: flex;
      }

      .mode-icons .icon-sun  { opacity: 0; }
      .mode-icons .icon-moon { opacity: 1; }

      body.dark-mode .mode-icons .icon-sun  { opacity: 1; }
      body.dark-mode .mode-icons .icon-moon { opacity: 0; }

      .mode-label {
        font-size: 16px;
        font-weight: 500;
        color: var(--text-muted);
        white-space: nowrap;
        opacity: 1;
        transition: opacity 0.3s ease;
        flex: 1;
      }

      #sidebar.close .mode-label { opacity: 0; }

      .toggle-switch-wrap {
        min-width: 60px;
        display: flex; align-items: center; justify-content: center;
      }

      .toggle-track {
        width: 44px; height: 22px;
        border-radius: 25px;
        background: var(--toggle-color);
        position: relative;
        transition: background 0.3s ease;
      }

      .toggle-thumb {
        position: absolute;
        width: 15px; height: 15px;
        border-radius: 50%;
        top: 50%; left: 4px;
        transform: translateY(-50%);
        background: var(--sidebar-color);
        transition: left 0.3s ease;
        box-shadow: 0 1px 4px rgba(0,0,0,0.2);
      }

      body.dark-mode .toggle-thumb { left: 24px; }

      /* ── Main Content ─────────────────────────────── */
      #main-content {
  position: fixed;
  left: 250px;
  top: 0;
  right: 0;
  bottom: 0;
  background: var(--body-color);
  transition: var(--tran-05);
  overflow-y: auto;
  overflow-x: hidden;
  height: 100vh;
}

      #sidebar.close ~ #main-content { left: 88px; }

      /* ── Pages ────────────────────────────────────── */
      .page {
  visibility: hidden;
  opacity: 0;
  transform: translateY(12px);
  transition: opacity 0.35s ease, transform 0.35s ease;
  padding: 0;
  min-height: auto;
  position: absolute;
  left: -9999px;
  width: 100%;
  right: auto;
}

      .page.active {
        visibility: visible;
        position: relative;
        left: auto;
      }

      .page.visible {
        opacity: 1;
        transform: translateY(0);
      }

      .page-header {
        padding: 0 40px 40px 40px;
        border-bottom: none !important;
      }

      .page-header h1 {
        margin-top: -15.6px;
        font-size: 28.8px;
        font-weight: 700;
        color: #7A1E10;
      }
      .fin-color { color: #d1beaf; }
        
      body.dark-mode .page-header h1 { color: #fff; }
      
      /* ── Dashboard Page ───────────────────────────────── */
      #page-dashboard .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
        margin-top: 30px;
      }
      #page-financials .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        margin-top: 30px;
      }
      #page-risk .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        margin-top: 30px;
      }
      #page-stocks .stats-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 20px;
        margin-top: 30px;
      }

      .stat-card {
        background: var(--sidebar-color);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: var(--tran-03);
      }
      .stat-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
      }
      .stat-card h4{
          font-size: 14px;
          font-weight: 500;
          color: #8A6040;
          margin-bottom: 8px;
      }

      .stat-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
      }

      .stat-card .stat-icon {
        width: 44px;
        height: 44px;
        border-radius: 10px;
        background: #FDF0DC;
        display: flex;
        align-items: center;
        justify-content: center;
      }

      .stat-card .stat-icon i { font-size: 22px; color: #7A1E10; }

      .stat-card .stat-value {
        font-size: 26px;
        font-weight: 700;
        color: #1a1a1a;
      }

      body.dark-mode .stat-card .stat-value { color: #fff; }

      .stat-card .stat-label {
        font-size: 13px;
        color: #888;
        margin-top: 4px;
      }

      body.dark .stat-card { background: var(--sidebar-color); }


      /* ── About Page ───────────────────────────────── */
      .about-section {
        padding: 0 40px 40px 40px;
        margin-top: 20px;
      }

      .about-intro {
        background: var(--sidebar-color);
        border-left: 5px solid #7A1E10;
        border-radius: 12px;
        padding: 26px 28px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
      }

      .about-intro h2 {
        font-size: 22px;
        font-weight: 600;
        color: #7A1E10;
        margin-bottom: 10px;
      }

      .about-intro p,
      .about-card p {
        color: #555;
        font-size: 14px;
        line-height: 1.7;
      }

      body.dark-mode .about-intro p,
      body.dark-mode .about-card p { color: #ccc; }

      .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #1a1a1a;
        margin: 28px 0 8px;
      }

      body.dark-mode .section-title { color: #fff; }

      .about-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        margin-top: 8px;
      }

      .about-card {
        background: var(--sidebar-color);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
      }

      .about-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
      }

      .about-icon {
        width: 44px;
        height: 44px;
        border-radius: 10px;
        background: #FDF0DC;
        color: #7A1E10;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 16px;
      }

      .about-card h3 {
        color: #1a1a1a;
        font-size: 16px;
        margin-bottom: 8px;
      }

      body.dark-mode .about-card h3 { color: #fff; }
      body.dark-mode .about-card    { background: var(--sidebar-color); }

      /* ── Dataset card ─────────────────────────────── */
      .dataset-card {
        background: var(--sidebar-color);
        border-radius: 12px;
        padding: 26px 28px;
        margin-top: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
      }

      .dataset-header {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 15px;
        font-weight: 600;
        color: #7A1E10;
        margin-bottom: 14px;
      }

      .dataset-title {
        font-size: 15px;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 10px;
      }

      body.dark-mode .dataset-title { color: #fff; }

      .dataset-desc {
        font-size: 13.5px;
        color: #555;
        line-height: 1.7;
        margin-bottom: 14px;
      }

      body.dark-mode .dataset-desc { color: #ccc; }

      .dataset-meta {
        font-size: 13px;
        color: #444;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 8px;
      }

      body.dark-mode .dataset-meta { color: #bbb; }

      .meta-divider { color: #aaa; }

      .dataset-btn {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: #7A1E10;
        color: #fff;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 600;
        text-decoration: none;
        transition: background 0.2s ease;
      }

      .dataset-btn:hover { background: #9b2516; color: #fff; }

      /* ── Our Team Section ─────────────────────────────── */
      .team-section-title {
        font-size: 40px;
        font-weight: 800;
        color: #1a1a1a;
        margin: 36px 0 55px 60px;
        line-height: 1.0;
        letter-spacing: -1px;
        text-transform: uppercase;
      }

      body.dark-mode .team-section-title { color: #fff; }

      .team-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 32px;
        margin-top: 8px;
        justify-items: center;
      }

      .team-card {
        display: flex;
        flex-direction: column;
        align-items: center;
        position: relative;
        text-align: center;
      }

      .team-hex-wrapper {
        position: relative;
        width: 228px;
        height: 216px;
        margin-bottom: 18px;
      }

      .team-hex-photo {
        width: 208px;
        height: 196px;
        clip-path: polygon(25% 0%, 75% 0%, 100% 50%, 75% 100%, 25% 100%, 0% 50%);
        overflow: hidden;
        background: #e0e0e0;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        cursor: pointer;
      }
      .team-hex-photo img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
      }
      .hex-overlay {
        position: absolute;
        inset: 0;
        display: flex;
        align-items: flex-end;
        opacity: 0;
        transform: translateY(20px);
        transition: opacity 0.3s ease, transform 0.3s ease;
        pointer-events: none;
        z-index: 10;
      }

      .team-hex-photo.open .hex-overlay {
        opacity: 1;
        transform: translateY(0);
        pointer-events: auto;
      }

      .hex-overlay-content {
        width: 100%;
        padding: 10px 10px 14px;
        background: rgba(0, 0, 0, 0.72);
        text-align: center;
      }

      .overlay-name {
        font-size: 11px;
        font-weight: 700;
        color: #fff;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        line-height: 1.3;
        margin-bottom: 2px;
      }

      .overlay-role {
        font-size: 10px;
        font-weight: 500;
        color: rgba(255,255,255,0.75);
        margin-bottom: 5px;
      }

      .overlay-desc {
        font-size: 9px;
        color: rgba(255,255,255,0.65);
        line-height: 1.5;
      }
      /* ── Team Modal ───────────────────────────────────── */
      .team-modal-backdrop {
        display: none;
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,0.45);
        z-index: 999;
        align-items: center;
        justify-content: center;
        backdrop-filter: blur(3px);
      }

      .team-modal-backdrop.open {
        display: flex;
      }

      .team-modal {
        background: var(--sidebar-color);
        border-radius: 24px;
        padding: 0;
        width: 410px;
        text-align: center;
        box-shadow: 0 24px 80px rgba(0,0,0,0.25);
        position: relative;
        animation: modalPop 0.25s ease;
        overflow: hidden;
      }

      @keyframes modalPop {
        from { transform: scale(0.85); opacity: 0; }
        to   { transform: scale(1);    opacity: 1; }
      }

      .team-modal-close {
        position: absolute;
        top: 14px; right: 16px;
        font-size: 20px;
        color: #fff;
        cursor: pointer;
        line-height: 1;
        background: rgba(255,255,255,0.2);
        border: none;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10;
        transition: background 0.2s;
      }

      .team-modal-close:hover { background: rgba(255,255,255,0.35); }

      .team-modal-header {
        background: linear-gradient(135deg, #7A1E10, #b5341e);
        padding: 32px 28px 48px;
        position: relative;
        display: flex;
        flex-direction: column;
        align-items: center;
      }

      .team-modal-avatar {
        width: 90px;
        height: 90px;
        border-radius: 50%;
        overflow: hidden;
        border: 3px solid rgba(255,255,255,0.6);
        background: rgba(255,255,255,0.2);
        margin-bottom: 12px;
      }

      .team-modal-avatar img {
        width: 100%; height: 100%;
        object-fit: cover;
        display: block;
      }

      .team-modal-avatar.no-photo {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 36px;
        font-weight: 700;
        color: #fff;
      }

      .team-modal-name {
        font-size: 18px;
        font-weight: 700;
        color: #fff;
        margin-bottom: 4px;
      }

      .team-modal-role {
        display: inline-block;
        font-size: 11px;
        font-weight: 600;
        color: rgba(255,255,255,0.85);
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 20px;
        padding: 3px 14px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
      }

      .team-modal-body {
        padding: 24px 28px 28px;
        background: var(--sidebar-color);
        border-radius: 20px;
        margin-top: -18px;
        position: relative;
      }
      body.dark-mode .team-modal-body { color: #fff; }

      .team-modal-info-row {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        text-align: left;
        padding: 0px 0;
        border-bottom: 1px solid rgba(0,0,0,0.06);
      }

      body.dark-mode .team-modal-info-row {
        border-bottom-color: rgba(255,255,255,0.07);
      }

      .team-modal-info-row:last-child { border-bottom: none; }

      .info-icon {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        background: #FDF0DC;
        color: #7A1E10;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        margin-top: 1px;
      }

      body.dark-mode .info-icon { background: rgba(122,30,16,0.2); }

      .info-text {
        display: flex;
        flex-direction: column;
      }

      .info-label {
        font-size: 10px;
        font-weight: 600;
        color: #7A1E10;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 1px;
      }

      .info-value {
        font-size: 12.5px;
        color: #444;
        line-height: 1.5;
        font-weight: 500;
      }

      body.dark-mode .info-value { color: #ccc; }

      .team-modal-role {
        font-size: 13px;
        font-weight: 700;
        color: #7A1E10;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }

      .team-hex-shadow {
        position: absolute;
        top: 12px; left: 12px;
        width: 208px;
        height: 196px;
        clip-path: polygon(25% 0%, 75% 0%, 100% 50%, 75% 100%, 25% 100%, 0% 50%);
        background: rgba(0,0,0,0.07);
        z-index: -1;
      }

      .team-hex-badge {
        position: absolute;
        bottom: 0px;
        left: -2px;
        width: 64px;
        height: 60px;
        clip-path: polygon(25% 0%, 75% 0%, 100% 50%, 75% 100%, 25% 100%, 0% 50%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #fff;
        transform: rotate(90deg);  
      }

      .team-hex-badge.jay    { background: linear-gradient(135deg, #b5541c, #e07b3a); }
      .team-hex-badge.mariel { background: linear-gradient(135deg, #6a0dad, #c850c0); }
      .team-hex-badge.ralph  { background: linear-gradient(135deg, #0d47a1, #1976d2); }

      .team-hex-badge svg { width: 28px; height: 28px; transform: rotate(-90deg);}

      .team-name {
        font-size: 15px;
        font-weight: 700;
        color: #1a1a1a;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
      }

      body.dark-mode .team-name { color: #fff; }

      .team-role {
        font-size: 13px;
        color: #777;
        font-weight: 400;
      }

      body.dark-mode .team-role { color: #bbb; }

      .chart-grid {
        margin-bottom: 0px;
      }

      .chart-card canvas {
          max-height: 300px;
      }

      #page-simulation .chart-card canvas {
          max-height: none !important;
      }

      .chart-card {
          display: flex;
          flex-direction: column;
      }

      .chart-card h3 {
          font-size: 16px;
          margin-bottom: 15px;
          color: #7A1E10;
          font-weight: bold;
      }

      body.dark .chart-card {
          background: #252624 !important;
      }

      body.dark .chart-card h3 {
          color: #fff;
      }

      @media (max-width: 768px) {
        .team-grid { grid-template-columns: 1fr; }
        .about-grid { grid-template-columns: 1fr; }
      }
    "))
  ),

  # ── Sidebar ─────────────────────────────────────────────────────────────────
  tags$nav(id = "sidebar",

    tags$div(class = "sidebar-header",
      tags$div(class = "brand-logo",
        tags$img(src = "Logo.png", alt = "FinSight logo")
      ),
      tags$div(class = "brand-text",
        tags$div(class = "brand-name",
          HTML('Fin<span class="highlight">Sight</span>')
        ),
        tags$span(class = "brand-tagline", "FINANCIAL INTELLIGENCE PLATFORM")
      ),
      tags$div(class = "sidebar-toggle", id = "sidebarToggle",
        svgIcon("chevron")
      )
    ),

    tags$div(class = "menu-section",

      tags$ul(class = "nav-list",
        navItem("dashboard", "dashboard", "Dashboard", active = TRUE),
        navItem("financials", "financial",  "Financial"),
        navItem("risk",       "shield",    "Risk Analysis"),
        navItem("stocks",     "stats",     "Stock Insights"),
        navItem("simulation", "sliders",   "Simulation")
      ),

      tags$div(class = "bottom-section",
        tags$ul(class = "nav-list",
          navItem("about", "home", "About")
        ),
        tags$div(class = "mode-row", id = "darkModeToggle",
          tags$div(class = "mode-icons",
            tags$span(class = "icon-moon", svgIcon("moon")),
            tags$span(class = "icon-sun",  svgIcon("sun"))
          ),
          tags$span(class = "mode-label", "Dark Mode"),
          tags$div(class = "toggle-switch-wrap",
            tags$div(class = "toggle-track",
              tags$div(class = "toggle-thumb")
            )
          )
        )
      )
    )
  ),

  # ── Main Content ────────────────────────────────────────────────────────────
  tags$div(id = "main-content",

    # Dashboard
    tags$div(class = "page active", id = "page-dashboard",
      tags$div(class = "page-header",
        tags$h1("Overview"),
        tags$div(class="stats-grid",
          tags$div(class="stat-card", style = "border-top: 4px solid #C0392B;",
            tags$div(class="stat-card-header",
              tags$h4("Total Companies"),
              tags$div(class="stat-icon", dashboardIcon("buildings"))
            ),
            tags$div(class="stat-value", textOutput("dash_total_companies")),
            tags$div(class="stat-label", "Unique entities in dataset")
          ),
          tags$div(class="stat-card", style = "border-top: 4px solid #E67E22;",
            tags$div(class="stat-card-header",
              tags$h4("Average Revenue"),
              tags$div(class="stat-icon", dashboardIcon("trending"))
            ),
            tags$div(class="stat-value", textOutput("dash_avg_revenue")),
            tags$div(class="stat-label", "Average across all quarters")
          ),
          tags$div(class="stat-card", style = "border-top: 4px solid #D4A012;",
            tags$div(class="stat-card-header",
              tags$h4("Average Credit Score"),
              tags$div(class="stat-icon", dashboardIcon("info"))
            ),
            tags$div(class="stat-value", textOutput("dash_avg_credit_score")),
            tags$div(class="stat-label", "Risk assessment metric")
          ),
          tags$div(class="stat-card", style = "border-top: 4px solid #D4A012;",
            tags$div(class="stat-card-header",
              tags$h4("Average Net Profit"),
              tags$div(class="stat-icon", dashboardIcon("profit"))
            ),
            tags$div(class="stat-value", textOutput("dash_avg_net_profit")),
            tags$div(class="stat-label", "Profitability metric")
          )
        ),
        tags$div(style="margin-top: 16px; max-width: 360px;",
        selectInput("dashboardQuarter", "Quarter Filter", 
          choices = quarterChoices, selected = "All Quarters")
        ),
        # First chart row
        tags$div(class="chart-grid", 
          style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 16px;",
          tags$div(class="chart-card", 
            style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);",
            tags$h3("Revenue by Quarter"),
            plotOutput("revenueQuarterPlot", height = "300px")
          ),
          tags$div(class="chart-card", 
            style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);",
            tags$h3("Quarterly Profit Trend"),
            plotOutput("profitTrendPlot", height = "300px")
          )
        ),

        # Credit Risk filter — between the rows
        tags$div(style="margin-top: 24px; max-width: 360px;",
          selectInput("dashboardCreditRiskFilter", "Credit Risk Filter",
            choices = creditRiskChoices, selected = "All")
        ),
        # Second chart row
        tags$div(class="chart-grid", 
          style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 16px;",
          tags$div(class="chart-card", 
            style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);",
            tags$h3("Credit Risk Distribution"),
            plotOutput("creditRiskDistPlot", height = "300px")
          ),
          tags$div(class="chart-card", 
            style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);",
            tags$h3("Stock Trend Distribution"),
            plotOutput("stockTrendDistPlot", height = "300px")
          )
        )
      )
    ),

    # Financial
    tags$div(class = "page", id = "page-financials",
      tags$div(class = "page-header",
        tags$h1("Financial"),
        tags$div(class="stats-grid", style="margin-top: 20px;",
          tags$div(class="stat-card", style="border-top: 4px solid #2E9E60;",
            tags$div(class="stat-card-header",
              tags$h4("Average Net Profit"),
              tags$div(class="stat-icon", financialIcon("money"))
            ),
            tags$div(class="stat-value", textOutput("fin_avg_net_profit")),
            tags$div(class="stat-label","Average across all")
          ),
          tags$div(class="stat-card", style="border-top: 4px solid #D4A012;",
            tags$div(class="stat-card-header",
              tags$h4("Current Ratio"),
              tags$div(class="stat-icon", financialIcon("analyse"))
            ),
            tags$div(class="stat-value", textOutput("fin_avg_current_ratio")),
            tags$div(class="stat-label","Liquidity measure")
          ),
          tags$div(class="stat-card", style="border-top: 4px solid #7A1E10;",
            tags$div(class="stat-card-header",
              tags$h4("Average EPS"),
              tags$div(class="stat-icon", financialIcon("linechart"))
            ),
            tags$div(class="stat-value", textOutput("fin_avg_eps")),
            tags$div(class="stat-label","Earnings per share")
          )
        ),
        # Top row: Revenue & Profit Trend | Average Market Cap
        tags$div(style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 20px;",
          tags$div(class="chart-card", style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);",
            tags$h3("Revenue & Profit Trend"),
            plotOutput("revProfitTrendPlot", height = "300px")
          ),
          tags$div(class="chart-card", style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);",
            tags$h3("Average Market Cap"),
            plotOutput("avgMarketCapPlot", height = "300px")
          )
        ),

        # Bottom row: Revenue vs Net Profit full width
        tags$div(style="margin-top: 20px;",
          tags$div(class="chart-card", style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);",
            tags$h3("Revenue vs Net Profit"),
            plotOutput("revProfitPlot", height = "300px")
          )
        ),
        tags$div(style="margin-top:18px;",
        tags$div(style = "display:flex; gap:12px; align-items:center; margin-bottom:12px;",
          selectInput("topMetric", "Metric", choices = c("EPS", "Revenue"), selected = "EPS", width = "160px"),
          radioButtons("topBottom", "Select", choices = c("Top", "Bottom"), selected = "Top", inline = TRUE),
          numericInput("topN", "N", value = 10, min = 1, max = 100, width = "120px")
        ),
        tags$div(class="chart-card", style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);",
          tags$h3("Top / Bottom Companies"),
          DT::dataTableOutput("topBottomTable")
        )
      )
      )
    ),

    # Risk Analysis
    tags$div(class = "page", id = "page-risk",
      tags$div(class = "page-header",
        tags$h1("Risk Analysis"),
        tags$div(class="stats-grid",style="margin-top: 20px;",
          tags$div(class="stat-card", style="border-top: 4px solid #C0392B;",
            tags$div(class="stat-card-header",
              tags$h4("Average Credit Score"),
              tags$div(class="stat-icon", riskIcon("shield"))
            ),
            tags$div(class="stat-value", textOutput("risk_avg_credit_score")),
            tags$div(class="stat-label","Out of 850")
          ),
          tags$div(class="stat-card", style="border-top: 4px solid #378ADD;",
            tags$div(class="stat-card-header",
              tags$h4("Debt-to-Equity"),
              tags$div(class="stat-icon", riskIcon("chart"))
            ),
            tags$div(class="stat-value", textOutput("risk_avg_debt_equity")),
            tags$div(class="stat-label","Average ratio")
          ),
          tags$div(class="stat-card", style="border-top: 4px solid #E67E22;",
            tags$div(class="stat-card-header",
              tags$h4("High-Risk Companies"),
              tags$div(class="stat-icon", riskIcon("error"))
            ),
            tags$div(class="stat-value", textOutput("risk_high_risk_companies")),
            tags$div(class="stat-label","Companies")
          )
        ),
       # Top row: Debt-to-Equity | Credit Risk Score
      tags$div(style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 20px;",
        tags$div(class="chart-card", style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);",
          tags$h3("Debt-to-Equity Ratio Distribution"),
          plotOutput("debtEquityHist", height = "300px")
        ),
        tags$div(class="chart-card", style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);",
          tags$h3("Credit Risk Score"),
          plotOutput("creditRiskScorePlot", height = "300px")
        )
      ),

      # Bottom row: Current Ratio vs Credit Score full width
      tags$div(style="margin-top: 20px;",
        tags$div(class="chart-card", style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);",
          tags$h3("Current Ratio vs Credit Score"),
          plotOutput("currentVSCreditPlot", height = "300px")
        )
      ),
        tags$div(style="margin-top:18px; display:flex; gap:12px; align-items:center;",
          selectInput("risk_table_quarter", "Quarter", choices = c("All", levels(rawData$Quarter)), selected = "All", width = "180px"),
          numericInput("risk_current_ratio_threshold", "Max Current Ratio", value = 1.5, min = 0, max = 10, step = 0.1, width = "160px")
        ),
        tags$div(style="margin-top:12px;",
          tags$div(class="chart-card", style="background: white; padding: 12px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);",
            tags$h3("High-Risk Companies (filterable)"),
            DT::dataTableOutput("highRiskTable")
          )
        )
      )
    ),

    # Stock Insights
    tags$div(class = "page", id = "page-stocks",
      tags$div(class = "page-header",
        tags$h1("Stock & Volatility Analysis"),
        tags$div(class="stats-grid",style="margin-top: 20px;",
          tags$div(class="stat-card", style="border-top: 4px solid #7A1E10;",
            tags$div(class="stat-card-header",
              tags$h4("Average Volatility"),
              tags$div(class="stat-icon", stockIcon("pulse"))
            ),
            tags$div(class="stat-value", textOutput("stock_avg_volatility")),
            tags$div(class="stat-label","Stock volatility")
          ),
          tags$div(class="stat-card", style="border-top: 4px solid #378ADD;",
            tags$div(class="stat-card-header",
              tags$h4("Uptrend Companies"),
              tags$div(class="stat-icon", stockIcon("uptrend"))
            ),
            tags$div(class="stat-value", textOutput("stock_uptrend_companies")),
            tags$div(class="stat-label","Companies")
          )
        ),
        # Top row: 3 columns
        tags$div(style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-top: 20px;",
          tags$div(class="chart-card", style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);",
            tags$h3("Stock Volatility Distribution"),
            plotOutput("stockVolatilityPlot", height = "300px")
          ),
          tags$div(class="chart-card", style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);",
            tags$h3("Average EPS by Stock Trend"),
            plotOutput("avgEpsByTrendPlot", height = "300px")
          ),
          tags$div(class="chart-card", style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);",
            tags$h3("Proportion of Stock Trends per Quarter"),
            plotOutput("stockTrendQuarterProportionPlot", height = "300px")
          )
        ),

        # Bottom row: Market Cap vs Stock Volatility full width
        tags$div(style="margin-top: 20px;",
          tags$div(class="chart-card", style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);",
            tags$h3("Market Cap vs Stock Volatility"),
            plotOutput("marketCapVolatilityPlot", height = "300px")
          )
        )
      )
    ),

    # Simulation
    tags$div(class = "page", id = "page-simulation",
      tags$div(class = "page-header",
        tags$h1("Simulation and Scenario Analysis"),
        tags$div(style="margin-top: 30px;",
          # ── Credit Risk Prediction Tab ────────────────────────────────────────
          tags$div(id = "sim-credit-risk-tab", style="border: 1px solid #ddd; border-radius: 12px; padding: 30px; background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.08);",
            tags$h2("A. Credit Risk Prediction"),
            tags$p("Adjust the financial metrics below and predict whether a company falls into Low, Medium, or High credit risk."),
            
            tags$div(style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 20px; margin: 30px 0;",
              tags$div(
                tags$label("Debt-to-Equity Ratio", style="font-weight: 600;"),
                sliderInput("sim_debt_equity", NULL, 
                  min = 0, max = 5, value = 2, step = 0.1)
              ),
              tags$div(
                tags$label("Current Ratio", style="font-weight: 600;"),
                sliderInput("sim_current_ratio", NULL, 
                  min = 0.5, max = 5, value = 2, step = 0.1)
              ),
              tags$div(
                tags$label("EPS (Earnings Per Share)", style="font-weight: 600;"),
                sliderInput("sim_eps", NULL, 
                  min = -10, max = 30, value = 10, step = 0.5)
              ),
              tags$div(
                tags$label("Revenue", style="font-weight: 600;"),
                numericInput("sim_revenue", NULL,
                  value = median(rawData$Revenue, na.rm = TRUE), min = 0, max = max(rawData$Revenue, na.rm = TRUE) * 1.2, step = 1000, width = "100%")
              ),
              tags$div(
                tags$label("Credit Score", style="font-weight: 600;"),
                numericInput("sim_credit_score", NULL,
                  value = round(mean(rawData$Credit_Score, na.rm = TRUE), 1), min = 0, max = 850, step = 1, width = "100%")
              )
            ),
            
            actionButton("predict_credit_btn", "Predict Credit Risk", 
              style="background-color: #7A1E10; color: white; padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 14px;"),
            tags$span(" "),
            actionButton("add_credit_scenario", "Add Scenario", 
              style="background-color: #378ADD; color: white; padding: 10px 16px; border: none; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 13px; margin-left: 8px;"),
            
            tags$div(style="margin-top: 30px; display: grid; grid-template-columns: 1fr 1fr; gap: 20px;",
              tags$div(style="background: #f9f9f9; padding: 20px; border-radius: 12px;",
                tags$h3("Prediction Result", style="margin-top: 0;"),
                tags$div(style="font-size: 28px; font-weight: 700; color: #7A1E10; margin: 15px 0;",
                  textOutput("credit_risk_prediction")
                ),
                tags$div(style="font-size: 14px; color: #666;",
                  "Predicted Credit Risk Category"
                )
              ),
              tags$div(style="background: #f9f9f9; padding: 20px; border-radius: 12px;",
                tags$h3("Confidence", style="margin-top: 0;"),
                tags$div(style="font-size: 28px; font-weight: 700; color: #378ADD; margin: 15px 0;",
                  textOutput("credit_risk_confidence")
                ),
                tags$div(style="font-size: 14px; color: #666;",
                  "Model Confidence Level"
                )
              )
            ),
            
            tags$div(style="margin-top: 30px; padding: 20px; background: #f9f9f9; border-radius: 12px;",
              tags$h3("Variable Importance", style="margin-top: 0;"),
              plotOutput("credit_risk_importance", height = "250px")
            ),
            tags$div(style="margin-top: 20px;",
              tags$h3("What-if Scenarios", style="margin-top: 0;"),
              tableOutput("whatif_table")
            )
          ),
          
          # ── Stock Trend Simulation Tab ───────────────────────────────────────
          tags$div(id = "sim-stock-trend-tab", style="border: 1px solid #ddd; border-radius: 12px; padding: 30px; background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-top: 30px;",
            tags$h2("B. Stock Trend Simulation"),
            tags$p("Adjust the market metrics below and simulate the stock trend probability."),
            
            tags$div(style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin: 30px 0;",
              tags$div(
                tags$label("EPS (Earnings Per Share)", style="font-weight: 600;"),
                sliderInput("sim_stock_eps", NULL, 
                  min = -10, max = 30, value = 10, step = 0.5)
              ),
              tags$div(
                tags$label("Stock Volatility", style="font-weight: 600;"),
                sliderInput("sim_stock_volatility", NULL, 
                  min = 0, max = 100, value = 40, step = 1)
              ),
              tags$div(
                tags$label("Market Cap (M)", style="font-weight: 600;"),
                sliderInput("sim_market_cap", NULL, 
                  min = 10, max = 1000, value = 400, step = 10)
              )
            ),
            
            actionButton("predict_stock_btn", "Simulate Stock Trend", 
              style="background-color: #7A1E10; color: white; padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 14px;"),
            
            tags$div(style="margin-top: 30px;",
              tags$h3("Trend Probabilities", style="margin-bottom: 20px;"),
              tags$div(style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;",
                tags$div(style="background: #f0f8ff; padding: 20px; border-radius: 12px; border-left: 4px solid #378ADD;",
                  tags$div(style="font-weight: 600; color: #378ADD; margin-bottom: 10px;", "↑ Uptrend"),
                  tags$div(style="font-size: 24px; font-weight: 700; color: #378ADD;",
                    textOutput("stock_prob_up")
                  ),
                  tags$div(style="font-size: 12px; color: #666; margin-top: 5px;", "Probability")
                ),
                tags$div(style="background: #fff8f0; padding: 20px; border-radius: 12px; border-left: 4px solid #D4A012;",
                  tags$div(style="font-weight: 600; color: #D4A012; margin-bottom: 10px;", "→ Stable"),
                  tags$div(style="font-size: 24px; font-weight: 700; color: #D4A012;",
                    textOutput("stock_prob_stable")
                  ),
                  tags$div(style="font-size: 12px; color: #666; margin-top: 5px;", "Probability")
                ),
                tags$div(style="background: #fff0f0; padding: 20px; border-radius: 12px; border-left: 4px solid #C0392B;",
                  tags$div(style="font-weight: 600; color: #C0392B; margin-bottom: 10px;", "↓ Downtrend"),
                  tags$div(style="font-size: 24px; font-weight: 700; color: #C0392B;",
                    textOutput("stock_prob_down")
                  ),
                  tags$div(style="font-size: 12px; color: #666; margin-top: 5px;", "Probability")
                )
              )
            ),
            
            tags$div(style="margin-top: 30px; padding: 20px; background: #f9f9f9; border-radius: 12px;",
              tags$h3("Variable Importance", style="margin-top: 0;"),
              plotOutput("stock_trend_importance", height = "250px")
            )
          )
        )
      )
    ),

    # About ──────────────────────────────────────────────────────────────────
    tags$div(class = "page", id = "page-about",
      tags$div(class = "page-header",
        tags$h1("About", class = "brand-name",
          HTML('<span class="fin-color">Fin</span><span class="highlight">Sight</span>'))
      ),
      aboutPageContent()
    )
  ),

  # ── Team Modal ──────────────────────────────────────────────────────────────
  tags$div(class = "team-modal-backdrop", id = "teamModal",
    tags$div(class = "team-modal",
      tags$button(class = "team-modal-close", onclick = "closeTeamModal()", HTML("&times;")),
      tags$div(class = "team-modal-header",
        tags$div(class = "team-modal-avatar", id = "modalAvatar"),
        tags$div(class = "team-modal-name",   id = "modalName"),
        tags$div(class = "team-modal-role",   id = "modalRole")
      ),
      tags$div(class = "team-modal-body",     id = "modalDesc")
    )
  ),
 # ── JavaScript ──────────────────────────────────────────────────────────────
  tags$script(HTML("
    (function() {
      var sidebar = document.getElementById('sidebar');

      if (localStorage.getItem('sidebarClosed') === 'true') {
        sidebar.classList.add('close');
      }

      if (localStorage.getItem('darkMode') === 'true') {
        document.body.classList.add('dark-mode');
        document.querySelector('.mode-label').textContent = 'Light Mode';
      }

      var firstPage = document.querySelector('.page.active');
      if (firstPage) {
        setTimeout(function() { firstPage.classList.add('visible'); }, 50);
        function triggerResize(){
          for(var i=0;i<3;i++){
            (function(delay){
              setTimeout(function(){
                try{ window.dispatchEvent(new Event('resize')); }catch(e){}
                try{ if(window.jQuery) window.jQuery(window).trigger('resize'); }catch(e){}
                try{ if(window.Shiny && Shiny.onInputChange) Shiny.onInputChange('__page_resize', Date.now()); }catch(e){}
              }, delay);
            })(i*120);
          }
        }
        setTimeout(function() { triggerResize(); }, 100);
      }

      document.getElementById('sidebarToggle').addEventListener('click', function() {
        sidebar.classList.toggle('close');
        localStorage.setItem('sidebarClosed', sidebar.classList.contains('close'));
      });

      document.getElementById('darkModeToggle').addEventListener('click', function() {
        document.body.classList.toggle('dark-mode');
        var isDark = document.body.classList.contains('dark-mode');
        document.querySelector('.mode-label').textContent = isDark ? 'Light Mode' : 'Dark Mode';
        localStorage.setItem('darkMode', isDark);
      });

      document.querySelectorAll('.nav-item a[data-page]').forEach(function(a) {
        a.addEventListener('click', function(e) {
          e.preventDefault();
          var targetPage = this.getAttribute('data-page');

          document.querySelectorAll('.nav-item a').forEach(function(link) {
            link.classList.remove('active');
          });
          this.classList.add('active');

          document.querySelectorAll('.page').forEach(function(page) {
            page.classList.remove('active', 'visible');
          });

          var target = document.getElementById('page-' + targetPage);
          if (target) {
            target.classList.add('active');
            setTimeout(function() { target.classList.add('visible'); }, 20);
            setTimeout(function() { triggerResize(); }, 50);
          }
        });
      });

      // ── Team Modal ───────────────────────────────────────────────────────
      var teamMembers = {
        jay: {
          name:    'Jay Lumanta',
          year:    '2nd Year Student',
          college: 'College of Information and Computing (CIC)',
          course:  'BS Computer Science Major in Data Science',
          subject: 'CS 226 Data Analytics - Stat Using R, AY 2025-2026',
          role:    'UI Designer',
          img:     'jay.png'
        },
        mariel: {
          name:    'Mariel Rossini Go',
          year:    '2nd Year Student',
          college: 'College of Information and Computing (CIC)',
          course:  'BS Computer Science Major in Data Science',
          subject: 'CS 226 Data Analytics - Stat Using R, AY 2025-2026',
          role:    'Data Analyst',
          img:     'rossini.png'
        },
        ralph: {
          name:    'Ralph Benedict Moreno',
          year:    '2nd Year Student',
          college: 'College of Information and Computing (CIC)',
          course:  'BS Computer Science Major in Data Science',
          subject: 'CS 226 Data Analytics - Stat Using R, AY 2025-2026',
          role:    'Programmer',
          img:     null
        }
      };

      window.openTeamModal = function(member) {
        var m = teamMembers[member];
        if (!m) return;

        var avatar = document.getElementById('modalAvatar');
        avatar.innerHTML = '';
        avatar.className = 'team-modal-avatar';
        if (m.img) {
          var img = document.createElement('img');
          img.src = m.img; img.alt = m.name;
          avatar.appendChild(img);
        } else {
          avatar.classList.add('no-photo');
          avatar.style.background = member === 'mariel' ? '#9b38c2' : '#1565c0';
          avatar.textContent = m.name.charAt(0);
        }

        document.getElementById('modalName').textContent = m.name;
        document.getElementById('modalRole').textContent = m.role;
        document.getElementById('modalDesc').innerHTML =
          '<strong>' + m.year + '</strong><br>' +
          m.college + '<br>' +
          m.course + '<br>' +
          '<em>' + m.subject + '</em>';

        document.getElementById('teamModal').classList.add('open');
      };

      window.closeTeamModal = function() {
        document.getElementById('teamModal').classList.remove('open');
      };

      setTimeout(function() {
        var modal = document.getElementById('teamModal');
        if (modal) {
          modal.addEventListener('click', function(e) {
            if (e.target === this) closeTeamModal();
          });
        }
      }, 100);

    })();
  "))
)

# ── Server ────────────────────────────────────────────────────────────────────
server <- function(input, output, session) {

  filteredDashboard <- reactive({
    rawData
  })

  dashboardQuarterData <- reactive({
    if (is.null(input$dashboardQuarter) || input$dashboardQuarter == "All Quarters") {
      rawData
    } else {
      filter(rawData, Quarter == input$dashboardQuarter)
    }
  })

  dashboardCreditRiskData <- reactive({
    if (is.null(input$dashboardCreditRiskFilter) || input$dashboardCreditRiskFilter == "All") {
      rawData
    } else {
      filter(rawData, Credit_Risk_Score == input$dashboardCreditRiskFilter)
    }
  })

  output$dash_total_companies <- renderText({
    formatC(n_distinct(filteredDashboard()$Company_ID), format = "d", big.mark = ",")
  })
  output$dash_avg_revenue <- renderText({
    paste0("$", formatC(mean(filteredDashboard()$Revenue, na.rm = TRUE), format = "f", digits = 2, big.mark = ","))
  })
  output$dash_avg_credit_score <- renderText({
    formatC(mean(filteredDashboard()$Credit_Score, na.rm = TRUE), format = "f", digits = 1)
  })
  output$dash_avg_net_profit <- renderText({
    paste0("$", formatC(mean(filteredDashboard()$Net_Profit, na.rm = TRUE), format = "f", digits = 2, big.mark = ","))
  })

  output$fin_avg_net_profit <- renderText({
    paste0("$", formatC(mean(filteredDashboard()$Net_Profit, na.rm = TRUE), format = "f", digits = 2, big.mark = ","))
  })
  output$fin_avg_current_ratio <- renderText({
    formatC(mean(filteredDashboard()$Current_Ratio, na.rm = TRUE), format = "f", digits = 2)
  })
  output$fin_avg_eps <- renderText({
    formatC(mean(filteredDashboard()$EPS, na.rm = TRUE), format = "f", digits = 2)
  })

  output$risk_avg_credit_score <- renderText({
    formatC(mean(rawData$Credit_Score, na.rm = TRUE), format = "f", digits = 1)
  })
  output$risk_avg_debt_equity <- renderText({
    formatC(mean(rawData$Debt_to_Equity, na.rm = TRUE), format = "f", digits = 2)
  })
  output$risk_high_risk_companies <- renderText({
    sum(rawData$Credit_Risk_Score == "High", na.rm = TRUE)
  })

  output$stock_avg_volatility <- renderText({
    formatC(mean(rawData$Stock_Volatility, na.rm = TRUE), format = "f", digits = 2)
  })
  output$stock_uptrend_companies <- renderText({
    sum(filteredDashboard()$Stock_Trend == "Up", na.rm = TRUE)
  })

  output$revenueQuarterPlot <- renderPlot({
    data <- dashboardQuarterData() %>%
      group_by(Quarter) %>%
      summarise(Revenue = sum(Revenue, na.rm = TRUE), .groups = "drop")

    ggplot(data, aes(x = Quarter, y = Revenue, fill = Quarter)) +
      geom_col() +
      labs(x = "Quarter", y = "Revenue", subtitle = "Revenue by quarter") +
      theme_minimal() +
      theme(legend.position = "none")
  })

  output$creditRiskDistPlot <- renderPlot({
    data <- dashboardCreditRiskData() %>%
      count(Credit_Risk_Score, name = "Count")

    ggplot(data, aes(x = 2, y = Count, fill = Credit_Risk_Score)) +
      geom_col(width = 1, color = "white") +
      coord_polar(theta = "y") +
      xlim(0.5, 2.5) +
      labs(fill = "Credit Risk", subtitle = "Distribution by risk category") +
      theme_void()
  })

  output$stockTrendDistPlot <- renderPlot({
    data <- dashboardCreditRiskData() %>%
      count(Stock_Trend, name = "Count")

    ggplot(data, aes(x = 2, y = Count, fill = Stock_Trend)) +
      geom_col(width = 1, color = "white") +
      coord_polar(theta = "y") +
      xlim(0.5, 2.5) +
      labs(fill = "Stock Trend", subtitle = "Trend proportions") +
      theme_void()
  })

  output$profitTrendPlot <- renderPlot({
    data <- dashboardQuarterData() %>%
      group_by(Quarter) %>%
      summarise(Net_Profit = mean(Net_Profit, na.rm = TRUE), .groups = "drop")

    ggplot(data, aes(x = Quarter, y = Net_Profit, group = 1)) +
      geom_line(color = "#7A1E10", linewidth = 1.5) +
      geom_point(color = "#7A1E10", size = 3) +
      labs(x = "Quarter", y = "Average Net Profit", subtitle = "Profit trend by quarter") +
      theme_minimal()
  })

  output$revProfitPlot <- renderPlot({
    # Scatter: Revenue vs Net Profit colored by Stock Trend
    cat("DEBUG: start revProfitPlot\n")
    flush.console()
    ggplot(rawData, aes(x = Revenue, y = Net_Profit, fill = Stock_Trend)) +
      geom_point(shape = 21, color = "black", alpha = 0.8, size = 1.8, stroke = 0.6) +
      labs(x = "Revenue", y = "Net Profit", fill = "Stock Trend", subtitle = "Revenue vs Net Profit (colored by stock trend)") +
      theme_minimal()
  })

  output$revProfitTrendPlot <- renderPlot({
    # Boxplot comparing Revenue and Net Profit distributions
    cat("DEBUG: start revProfitTrendPlot\n")
    flush.console()
    plotData <- data.frame(
      Measure = rep(c("Revenue", "Net Profit"), each = nrow(rawData)),
      Value = c(rawData$Revenue, rawData$Net_Profit)
    )
    ggplot(plotData, aes(x = Measure, y = Value, fill = Measure)) +
      geom_boxplot() +
      labs(x = "Measure", y = "Value", subtitle = "Distribution: Revenue vs Net Profit") +
      theme_minimal() +
      theme(legend.position = "none")
  })

  output$avgMarketCapPlot <- renderPlot({
    cat("DEBUG: start avgMarketCapPlot\n")
    flush.console()
    data <- rawData %>%
      group_by(Quarter) %>%
      summarise(Market_Cap = mean(Market_Cap, na.rm = TRUE), .groups = "drop")

    ggplot(data, aes(x = Quarter, y = Market_Cap, fill = Quarter)) +
      geom_col() +
      labs(x = "Quarter", y = "Average Market Cap", subtitle = "Average market capitalization") +
      theme_minimal() +
      theme(legend.position = "none")
  })

  output$debtEquityHist <- renderPlot({
    cat("DEBUG: start debtEquityHist\n")
    flush.console()
    ggplot(rawData, aes(x = Debt_to_Equity, fill = Credit_Risk_Score)) +
      geom_histogram(position = "identity", alpha = 0.45, color = "white", bins = 25) +
      labs(x = "Debt-to-Equity", y = "Count", fill = "Credit Risk", subtitle = "Debt-to-Equity distribution by credit risk") +
      theme_minimal()
  })

  output$currentVSCreditPlot <- renderPlot({
    cat("DEBUG: start currentVSCreditPlot\n")
    flush.console()
    ggplot(rawData, aes(x = Current_Ratio, y = Credit_Score)) +
      geom_point(shape = 21, fill = "#378ADD", color = "black", alpha = 0.8, size = 1.8, stroke = 0.6) +
      geom_smooth(method = "lm", se = FALSE, color = "#D4A012") +
      labs(x = "Current Ratio", y = "Credit Score", subtitle = "Liquidity vs credit quality") +
      theme_minimal()
  })

  output$creditRiskScorePlot <- renderPlot({
    cat("DEBUG: start creditRiskScorePlot\n")
    flush.console()
    data <- rawData %>%
      group_by(Quarter, Credit_Risk_Score) %>%
      summarise(Count = n(), .groups = "drop")
    ggplot(data, aes(x = Quarter, y = Count, fill = Credit_Risk_Score)) +
      geom_col(position = position_dodge(width = 0.8)) +
      labs(x = "Quarter", y = "Count", fill = "Credit Risk", subtitle = "Credit risk breakdown per quarter") +
      theme_minimal()
  })

  output$stockVolatilityPlot <- renderPlot({
    cat("DEBUG: start stockVolatilityPlot\n")
    flush.console()
    ggplot(rawData, aes(x = Stock_Volatility, fill = Stock_Trend)) +
      geom_density(alpha = 0.45) +
      labs(x = "Stock Volatility", y = "Density", fill = "Stock Trend", subtitle = "Volatility density by stock trend") +
      theme_minimal()
  })

  output$marketCapVolatilityPlot <- renderPlot({
    cat("DEBUG: start marketCapVolatilityPlot\n")
    flush.console()
    plotData <- rawData %>%
      mutate(Revenue_Bin = cut(Revenue,
                               breaks = quantile(Revenue, probs = seq(0, 1, length.out = 6), na.rm = TRUE),
                               include.lowest = TRUE,
                               labels = c("Very Low", "Low", "Medium", "High", "Very High")))

    ggplot(plotData, aes(x = Market_Cap, y = Stock_Volatility, fill = Revenue_Bin)) +
      geom_point(shape = 21, color = "black", alpha = 0.9, size = 1.8, stroke = 0.6) +
      scale_fill_manual(values = c("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"), name = "Revenue Range") +
      labs(x = "Market Cap", y = "Stock Volatility", subtitle = "Market cap vs volatility (Revenue categories)") +
      theme_minimal()
  })

  output$stockTrendProportionPlot <- renderPlot({
    cat("DEBUG: start stockTrendProportionPlot\n")
    flush.console()
    data <- rawData %>%
      count(Quarter, Stock_Trend, name = "Count")
    ggplot(data, aes(x = Quarter, y = Count, fill = Stock_Trend)) +
      geom_col() +
      labs(x = "Quarter", y = "Count", fill = "Stock Trend", subtitle = "Stock trend breakdown per quarter") +
      theme_minimal()
  })

  # Average EPS by stock trend
  output$avgEpsByTrendPlot <- renderPlot({
    cat("DEBUG: start avgEpsByTrendPlot\n")
    flush.console()
    data <- rawData %>%
      group_by(Stock_Trend) %>%
      summarise(Average_EPS = mean(EPS, na.rm = TRUE), .groups = "drop")
    ggplot(data, aes(x = Stock_Trend, y = Average_EPS, fill = Stock_Trend)) +
      geom_col() +
      labs(x = "Stock Trend", y = "Average EPS", subtitle = "Average EPS by stock trend") +
      theme_minimal() +
      theme(legend.position = "none")
  })

  output$stockTrendQuarterProportionPlot <- renderPlot({
    cat("DEBUG: start stockTrendQuarterProportionPlot\n")
    flush.console()
    data <- rawData %>%
      count(Quarter, Stock_Trend, name = "Count")
    ggplot(data, aes(x = Quarter, y = Count, fill = Stock_Trend)) +
      geom_col() +
      labs(x = "Quarter", y = "Count", fill = "Stock Trend", subtitle = "Proportion of stock trends per quarter") +
      theme_minimal()
  })

  # Top/Bottom N companies table
  output$topBottomTable <- DT::renderDataTable({
    req(input$topN)
    metricCol <- ifelse(is.null(input$topMetric) || input$topMetric == "EPS", "EPS", "Revenue")
    df <- rawData[, c("Company_ID", "Quarter", "EPS", "Revenue")]
    if (input$topBottom == "Top") {
      ord <- order(-df[[metricCol]], na.last = TRUE)
    } else {
      ord <- order(df[[metricCol]], na.last = TRUE)
    }
    df <- df[ord, ]
    DT::datatable(head(df, input$topN), options = list(pageLength = input$topN, searching = TRUE), rownames = FALSE)
  })

  # High-risk companies filterable table
  output$highRiskTable <- DT::renderDataTable({
    selQ <- ifelse(is.null(input$risk_table_quarter), "All", input$risk_table_quarter)
    df <- rawData[rawData$Credit_Risk_Score == "High", ]
    if (!is.null(selQ) && selQ != "All") df <- df[df$Quarter == selQ, ]
    if (!is.null(input$risk_current_ratio_threshold)) df <- df[!is.na(df$Current_Ratio) & df$Current_Ratio <= input$risk_current_ratio_threshold, ]
    DT::datatable(df %>% select(Company_ID, Quarter, Current_Ratio, Debt_to_Equity, Credit_Score, Credit_Risk_Score), filter = 'top', options = list(pageLength = 10), rownames = FALSE)
  })

  # ── Simulation: Credit Risk prediction handler ───────────────────────────
  observeEvent(input$predict_credit_btn, {
    req(creditRiskModel)
    newdata <- data.frame(
      Debt_to_Equity = as.numeric(input$sim_debt_equity),
      Current_Ratio = as.numeric(input$sim_current_ratio),
      EPS = as.numeric(input$sim_eps),
      Revenue = as.numeric(input$sim_revenue),
      Credit_Score = as.numeric(input$sim_credit_score)
    )
    probs <- predict(creditRiskModel, newdata, type = "prob")
    pred <- names(probs)[which.max(probs)]
    conf <- round(max(probs) * 100, 1)

    output$credit_risk_prediction <- renderText({ pred })
    output$credit_risk_confidence <- renderText({ paste0(conf, "%") })
  })

  output$credit_risk_importance <- renderPlot({
    cat("DEBUG: start credit_risk_importance\n")
    flush.console()
    imp <- varImp(creditRiskModel)$importance
    imp$Feature <- rownames(imp)
    ggplot(imp, aes(x = reorder(Feature, Overall), y = Overall)) +
      geom_col(fill = "#7A1E10") +
      coord_flip() +
      labs(x = "", y = "Importance") +
      theme_minimal()
  })

  # Ensure non-dashboard plots are not suspended when hidden so they render after page switch
  try({
    outputOptions(output, 'revProfitPlot', suspendWhenHidden = FALSE)
    outputOptions(output, 'revProfitTrendPlot', suspendWhenHidden = FALSE)
    outputOptions(output, 'avgMarketCapPlot', suspendWhenHidden = FALSE)
    outputOptions(output, 'debtEquityHist', suspendWhenHidden = FALSE)
    outputOptions(output, 'currentVSCreditPlot', suspendWhenHidden = FALSE)
    outputOptions(output, 'creditRiskScorePlot', suspendWhenHidden = FALSE)
    outputOptions(output, 'stockVolatilityPlot', suspendWhenHidden = FALSE)
    outputOptions(output, 'marketCapVolatilityPlot', suspendWhenHidden = FALSE)
    outputOptions(output, 'stockTrendProportionPlot', suspendWhenHidden = FALSE)
    outputOptions(output, 'credit_risk_importance', suspendWhenHidden = FALSE)
    outputOptions(output, 'stock_trend_importance', suspendWhenHidden = FALSE)
  }, silent = TRUE)

  # what-if scenarios storage for credit risk
  creditScenarios <- reactiveVal(data.frame(
    Debt_to_Equity = numeric(), Current_Ratio = numeric(), EPS = numeric(), Revenue = numeric(), Credit_Score = numeric(), Predicted = character(), Confidence = character(), stringsAsFactors = FALSE
  ))

  observeEvent(input$add_credit_scenario, {
    req(creditRiskModel)
    newrow <- data.frame(
      Debt_to_Equity = as.numeric(input$sim_debt_equity),
      Current_Ratio = as.numeric(input$sim_current_ratio),
      EPS = as.numeric(input$sim_eps),
      Revenue = as.numeric(input$sim_revenue),
      Credit_Score = as.numeric(input$sim_credit_score)
    )
    probs <- predict(creditRiskModel, newrow, type = "prob")
    pred <- names(probs)[which.max(probs)]
    conf <- paste0(round(max(probs) * 100, 1), "%")
    df <- creditScenarios()
    df <- rbind(df, cbind(newrow, Predicted = pred, Confidence = conf))
    creditScenarios(df)
  })

  output$whatif_table <- renderTable({
    creditScenarios()
  }, sanitize.text.function = function(x) x)

  # ── Simulation: Stock Trend prediction handler ───────────────────────────
  observeEvent(input$predict_stock_btn, {
    req(stockTrendModel)
    newdata <- data.frame(
      EPS = as.numeric(input$sim_stock_eps),
      Stock_Volatility = as.numeric(input$sim_stock_volatility),
      Market_Cap = as.numeric(input$sim_market_cap)
    )
    probs <- predict(stockTrendModel, newdata, type = "prob")

    up <- if ("Up" %in% colnames(probs)) probs[, "Up"] else 0
    stable <- if ("Stable" %in% colnames(probs)) probs[, "Stable"] else 0
    down <- if ("Down" %in% colnames(probs)) probs[, "Down"] else 0

    output$stock_prob_up <- renderText({ paste0(round(up * 100, 1), "%") })
    output$stock_prob_stable <- renderText({ paste0(round(stable * 100, 1), "%") })
    output$stock_prob_down <- renderText({ paste0(round(down * 100, 1), "%") })
  })

  output$stock_trend_importance <- renderPlot({
    imp <- varImp(stockTrendModel)$importance
    imp$Feature <- rownames(imp)
    ggplot(imp, aes(x = reorder(Feature, Overall), y = Overall)) +
      geom_col(fill = "#378ADD") +
      coord_flip() +
      labs(x = "", y = "Importance") +
      theme_minimal()
  })
}

shinyApp(ui = ui, server = server)
