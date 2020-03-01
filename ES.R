head(elec)

which(elec$elec==23230)

ES <- function() {
  #Reading hourly time series 
  library(readxl)
  library(writexl)
  #Sys.setenv(JAVA_HOME='C:\\Program Files\\Java\\jre1.8.0_241')
  library(xlsx)
  library(xts)
  library(tsbox)
  library(forecast)
  library(ggplot2)
  library(lubridate)
  
  #Reading datafile
  elec_dataset <- read_excel("C:/Users/JYW/Desktop/Github/repository/elec/elec_dataset_0019_v1.xlsx")
  
  #Data split index
  elec_dataset$date = sprintf("%0016s",elec_dataset$date)
  
  elec_dataset$year <- substr(elec_dataset$date,7,10)
  elec_dataset$month <- sprintf("%02d",as.numeric(substr(elec_dataset$date,1,2)))
  elec_dataset$day <- substr(elec_dataset$date,4,5)
  elec_dataset$time = as.numeric(paste0(elec_dataset$year, elec_dataset$month, elec_dataset$day))
  
  start = which(20100101<=elec_dataset$time)[1]
  end = which(20171228<=elec_dataset$time)[1]
  
  #generating ts variable
  time_all <- seq(from = ymd_h('2000-01-01 00'), 
                  to = ymd_h('2018-12-31 23'), by = 'hour')
  time_oos <- seq(from = ymd_h('2017-12-29 00'), 
                  to = ymd_h('2018-12-31 23'), by = 'hour')
  
  elec <- xts(elec_dataset[,2], order.by = time_all)
  
  fcg_ets <- list()
  
  for (x in 1:52) {
    y=24*7*(x-1)
    z_start = start + y   #2010.01.01 + y 
    z_end = end + y    #2017.12.28 + y ###왜 157728로 설정되어있는걸까??
    
    insample <- elec[z_start:z_end, ]   #2010.01.01 ~ 2017.12.28(목)
    insample_ts <- ts_ts(insample)
    fit_ets <- ets(insample_ts, model="ZZZ")
    fc_ets <- fit_ets %>% forecast(h=192)
    fcmean_ets <- fc_ets$mean
    
    time_fcos <- window(time_oos, start=1+y, end=192+y)
    fcg_ets[[x]] <- xts(fcmean_ets, order.by = time_fcos)
  }
  df_ets <- as.data.frame(fcg_ets)
  write.xlsx2(df_ets, file="C:/Users/JYW/Desktop/Github/repository/elec/ets.xlsx", sheetName="Sheet_ets")
}

ES()
