library(tidyverse)
library(tibble)
setwd('/home/ks2823/my_CRM/microbial_load_predictor')
devtools::install() 

library("MLP")
print("Loading parallel library")

library(parallel)



num_cores <- detectCores() - 1


# Function to rename columns based on the best available taxonomic information
rename_columns <- function(colnames) {
  sapply(colnames, function(colname) {
    # Split the column name by dots
    parts <- strsplit(colname, '\\.')[[1]]
    
    # Length of taxonomy levels
    num_parts <- length(parts)
    
    # Genus level exists
    if (num_parts >= 6 && !is.na(parts[6]) && parts[6] != "NA" && parts[6] != "") {
      return(paste0('g_', parts[6]))
    } 
    # Fallback to family level
    else if (num_parts >= 5 && !is.na(parts[5]) && parts[5] != "NA" && parts[5] != "") {
      return(paste0('uc_f_', parts[5]))
    } 
    # Fallback to order level
    else if (num_parts >= 4 && !is.na(parts[4]) && parts[4] != "NA" && parts[4] != "") {
      return(paste0('uc_o_', parts[4]))
    } 
    # Fallback to class level
    else if (num_parts >= 3 && !is.na(parts[3]) && parts[3] != "NA" && parts[3] != "") {
      return(paste0('uc_c_', parts[3]))
    } 
    else {
      return(paste0('fallback_', colname))
    }
  })
}

# Define the path to the directory containing the data files and the output CSV file
data_dir <- '/home/ks2823/Microbiomap/data_subsampled'
output_file <- '/home/ks2823/Microbiomap/Data/MB_100_subsampled_results/microbiomap_load.csv'

# Read the existing output file if it exists
if (file.exists(output_file)) {
  existing_results <- read.csv(output_file, stringsAsFactors = FALSE)
  processed_projects <- existing_results$project_i
} else {
  existing_results <- data.frame(project_i = character(), mean_load = numeric(), var_load = numeric(), stringsAsFactors = FALSE)
  processed_projects <- character()
}

# Function to append a new row to the output CSV file
append_to_csv <- function(df, file) {
  write.table(df, file, sep = ",", col.names = !file.exists(file), row.names = FALSE, append = TRUE)
}

# List of projects
projects <- list.files(data_dir)

# Loop through each project and its corresponding files
for (project in projects) {
  print("#######################################")
  for (i in 0:9) {
    
    project_i <- paste0(project, '_', i)
    print(paste("Processing", project_i))
    # Skip if the project iteration is already processed
    if (project_i %in% processed_projects) {
      next
    }
    
    file_path <- file.path(data_dir, project, paste0(project, '_', i, '.csv'))
    
    if (file.exists(file_path)) {
      # Read the CSV file
      input <- read.csv(file_path, header=TRUE, row.names=1, check.names=FALSE)

      if (nrow(input) < 100) {
        next
      }

      # Apply the renaming function to column names
      new_colnames <- rename_columns(colnames(input))
      
      # Set new column names to the data frame
      colnames(input) <- new_colnames

      results <- mclapply(0:9, function(j) {
        # Select 100 random samples without replacement
        subsampled_input <- input[sample(nrow(input), 100, replace=FALSE), ]
        
        # Get load
        load <- MLP(subsampled_input, "rdp_train_set_16", "load")
        
        # Calculate the mean and variance
        list(mean = mean(load$load), var = var(load$load))
      }, mc.cores = num_cores)

      # Extract mean and variance from results
      mean_load_list <- sapply(results, function(x) x$mean)
      var_load_list <- sapply(results, function(x) x$var)

      mean_load <- mean(mean_load_list)
      var_load <- mean(var_load_list)
      
      # Create a new data frame to append
      new_result <- data.frame(project_i = project_i, mean_load = mean_load, var_load = var_load)
      
      # Append the new result to the output file
      append_to_csv(new_result, output_file)
      
      # Optionally, update the processed_projects vector
      processed_projects <- c(processed_projects, project_i)
    }
  }
}