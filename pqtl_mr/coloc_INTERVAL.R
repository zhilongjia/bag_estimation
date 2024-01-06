library(coloc)
library(dplyr)
library(readxl)

df_interval <- read.csv('pqtl_data/pqtls_interval_bag.csv')
pheno <- 'bag'
genes <- unique(df_interval$id)  


i <- 1
res_path <- 'coloc_results/coloc_results_INTERVAL_bag.csv'
if (file.exists(res_path)) {
  coloc_results <- read.csv(res_path)
} else {
  coloc_results <- data.frame(snp = character(),
                              pvalues.df1 = numeric(),
                              MAF.df1 = numeric(),
                              N.df1 = numeric(),
                              V.df1 = numeric(),
                              z.df1 = numeric(),
                              r.df1 = numeric(),
                              lABF.df1 = numeric(),
                              pvalues.df2 = numeric(),
                              MAF.df2 = numeric(),
                              N.df2 = numeric(),
                              V.df2 = numeric(),
                              z.df2 = numeric(),
                              r.df2 = numeric(),
                              lABF.df2 = numeric(),
                              internal.sum.lABF = numeric(),
                              SNP.PP.H4 = numeric(),
                              gene = numeric())
}

gwas_path <-sprintf('gwas_data/bag.txt.gz')
gwas <- read.table(gwas_path, header = T)
i <- 1

for (i in 1:length(genes)) {
  protein <- genes[i]
  if (protein %in% coloc_results$gene) {
    print(paste('skipping', protein))
    next
  }
  print(paste('coloc on', pheno, 'with', protein,  sep = ' '))
  
  # pqtl_path <- sprintf('interval_pqtl_merged/%s.csv', protein)
  
  # pqtls <- read.csv(pqtl_path)
  pqtls <- subset.data.frame(df_interval, id == protein)

  input <- merge(pqtls, gwas, by = 'SNP', 
                     all = FALSE, suffixes=c("_pqtl", "_gwas"))
  input <- input[!duplicated(input$SNP), ]
  input$P_pqtl <- ifelse(input$P_pqtl == 0, 1e-200, input$P_pqtl)
  result <- coloc.abf(dataset1=list(pvalues = input$P_gwas, type = "quant",
                                    N = input$NMISS,
                                    snp = input$SNP),
                      
                      dataset2=list(pvalues=input$P_pqtl, type="quant",
                                    N = 3301, snp = input$SNP),
                      MAF = input$MAF)
  
  coloc_result <- result$results %>% filter(SNP.PP.H4 >= 0.75)
  if (nrow(coloc_result) > 0) {
    coloc_result$gene <- genes[i]
    coloc_results <- rbind.data.frame(coloc_results, coloc_result)
  }
  
}

write.csv(coloc_results, 'coloc_results/coloc_results_INTERVAL_bag.csv', row.names = F)

