
## Project 2 

This repo documents our group’s work on Project 2. Project 2 involved
analyzing the [online news popularity data
set](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity),
creating predictive models to predict the number of shares for each
channel, and automating Markdown reports for each type of article.

### Packages Required

The following packages were used to retrieve and analyze the data:  
\* `tidyverse`: Functions used to manipulate and reshape data.  
\* `dplyr`: Functions used to manipulate data in R.  
\* `GGally`: Functions used to create correlograms.  
\* `caret`: Functions that streamline the model training process for
regression and classification problems.  
\* `randomForest`: Has the function randomForest() which is used to
create and analyze random forests.  
\* `doParallel`: Functions used to allow parallel computing in R.   
\* `rmarkdown`: Package helps you create dynamic analysis documents that
combine code, rendered output, and text.

### Reports

A report was created for each data channel: Lifestyle, Business,
Entertainment, Social Media, Tech, and World articles.

-   The analysis for [Lifestyle articles is available
    here](https://github.com/ChennadeBrown/Project-2/blob/main/Reports/Lifestyle.md).
-   The analysis for [Business articles is available
    here](https://github.com/ChennadeBrown/Project-2/blob/main/Reports/Business.md).
-   The analysis for [Entertainment articles is available
    here](https://github.com/ChennadeBrown/Project-2/blob/main/Reports/Entertainment.md).
-   The analysis for [Social Media articles is available
    here](https://github.com/ChennadeBrown/Project-2/blob/main/Reports/Social%20Media.md).
-   The analysis for [Tech articles is available
    here](https://github.com/ChennadeBrown/Project-2/blob/main/Reports/Tech.md).
-   The analysis for [World articles is available
    here](https://github.com/ChennadeBrown/Project-2/blob/main/Reports/World.md).

### Code

The code to create the analysis from a single .Rmd file is below:

``` r
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
```


