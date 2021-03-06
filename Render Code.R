# Authors: Alex Prevatte and Chennade Brown
# Date: 10/31/21

# Create variable that contains each article topic 
library(tidyverse)
Topic <- list("Lifestyle", "Business", "Entertainment", "Social Media",
              "Tech", "World")

# Create list for output files
output_file <- paste0("Reports/", Topic, ".md")
params <- lapply(Topic, FUN = function(x){list(topic = x)})
reports <- tibble(output_file, params)

# Use apply function to render through each article
library(rmarkdown)
apply(reports, MARGIN = 1, 
      FUN = function(x){
        render("Project 2 Rmd.Rmd", output_file = x[[1]], params = x[[2]])
      })



