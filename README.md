# ST_558_Project_2

Purpose: The purpose of this repo was to contain and track all of the work that we did for our second project in ST 558. The goal of this project was to analyze data about the popularity of online news articles and create predictive models based on this data. 

R packages used: 
tidyverse
knitr
GGally
corrplot
qwraps2
vtable
psych
ggplot2
cowplot
caret
gbm
randomForest
tree
class
bst
reshape
reshape2
corrr
ggcorrplot
FactoMineR
factoextra
data.table

links to reports:
Lifestyle:
Entertainment:
Business:
Social.Media:
Tech:
World:

Code to render reports:
#### creates a list of all 6 desired params from online
data_channel_is <- c("Lifestyle", "Entertainment", "Business", "Social.Media", "Tech", "World")

#### creates the output file name 
output_file <- paste0(data_channel_is, ".md")

####create a list for each channel with just the channel name parameter
params = lapply(data_channel_is, FUN = function(x){list(DataChannel = x)})

####put into a data frame
reports <- tibble(output_file, params)

#### renders with params to all based on rows in reports
apply(reports, MARGIN=1, FUN = function(x){
#### change first path to your Rmd  and output_dir to whatever folder you want it to output to   
rmarkdown::render('C:/Documents/Github/ST_558_Project_2/_Rmd/ST_558_project_2', output_dir = "./automations_test2_html", output_file = x[[1]], params = x[[2]]
    )
  }
)