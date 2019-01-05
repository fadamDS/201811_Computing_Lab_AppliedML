classDistribution <- function(dataset.name = NULL, table = NULL, class = ncol(table))
{    
    print(paste("Class Distribution:", dataset.name, sep = " "));
    print(prop.table(table(table[,class])))
    cat("\n");
}