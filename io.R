load.data <- function(p.path = NULL, p.header = FALSE, p.dec = ".", p.sep = "\t", p.blank.lines.skip = FALSE, 
                      p.stringsAsFactors = FALSE, p.comment.char = "", p.initial.rows = 100, 
                      p.total.nrows = NULL, p.column.names = NULL, p.id = FALSE)
{    
    # Read the first N rows of the table.
    tableHead <- read.table(
        p.path, 
        header = p.header, 
        dec = p.dec, 
        sep = p.sep, 
        blank.lines.skip = p.blank.lines.skip, 
        stringsAsFactors = p.stringsAsFactors,
        comment.char = p.comment.char, 
        nrows = p.initial.rows);
    
    # Set the classes of features.
    classes <- sapply(tableHead, class)
    
    # Read the table and assign the classes.
    table.complete <- read.table(
        p.path, 
        header = p.header, 
        dec = p.dec, 
        sep = p.sep, 
        blank.lines.skip = p.blank.lines.skip,
        stringsAsFactors = p.stringsAsFactors, 
        comment.char = p.comment.char, 
        colClasses = classes, 
        nrows = p.total.nrows);
    
    # Define the column names in the table.
    if (!is.null(p.column.names))
    {
        colnames(table.complete) <- p.column.names
    }

    # Add a unique identifier column.
    if (p.id == TRUE)
    {
        id <- 1:nrow(table.complete)
        table.complete <- cbind(id, table.complete)        
    }
    
    # Remove unused data objects.
    rm(tableHead)
    
    return(table.complete)
}