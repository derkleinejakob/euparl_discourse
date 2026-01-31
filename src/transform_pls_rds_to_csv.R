install.packages("data.table")

library(data.table)

# define data paths -> NOTE: define custom input paths

path_speech<- "C:/Users/dennis/Downloads/Corpora_PLS_EP/Corpus_speeches_EP.RDS"
path_law <- "C:/Users/dennis/Downloads/Corpora_PLS_EP/Corpus_laws_EP.RDS"
path_bills <- "C:/Users/dennis/Downloads/Corpora_PLS_EP/Corpus_bills_EP.RDS"

# define helper functions 

save_as_csv_with_metadata <- function(data, csv_path) {
    stopifnot(is.data.frame(data))
    fwrite(
        data,
        file = csv_path,
        quote = TRUE,
        na = "",
        row.names = FALSE,
        dateTimeAs = "write.csv"
    )
    meta_path <- sub("\\.csv$", "_colclasses.txt", csv_path)
    column_classes <- sapply(data, class)
    dput(column_classes, file = meta_path)
    
    message("Saved CSV: ", csv_path)
    message("Saved column metadata: ", meta_path)
}

reload_with_metadata <- function(csv_path) {
    meta_path <- sub("\\.csv$", "_colclasses.txt", csv_path)
    col_classes <- dget(meta_path)
data <- fread(csv_path, colClasses = col_classes)
return(data)
}


# transform data (speeches, bills and laws) -> NOTE: define custom output paths

myspeech <- readRDS(path_speech)
save_as_csv_with_metadata(myspeech, "D:/DataLiteracy/PLS_data_transformed/speech_output.csv")

mybills <- readRDS(path_bills)
save_as_csv_with_metadata(mybills, "D:/DataLiteracy/PLS_data_transformed/bills_output.csv")

mylaw <- readRDS(path_law)
save_as_csv_with_metadata(mylaw, "D:/DataLiteracy/PLS_data_transformed/law_output.csv")


# (opt) revert transformation and check for equality

restored_speech_meta <- reload_with_metadata("D:/DataLiteracy/PLS_data_transformed/speech_output.csv")
all.equal(myspeech, restored_speech_meta)