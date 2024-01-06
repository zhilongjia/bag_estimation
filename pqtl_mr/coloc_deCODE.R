library(coloc)
library(dplyr)
library(readxl)

df_deCODE <- read.csv('data/pqtls_deCODE_bag.csv')
pheno <- 'bag'
genes <- df_deCODE$ProteinName

res_path <- 'coloc_results/coloc_results_deCODE_bag.csv'
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

i <- 1
gwas_path <- sprintf('gwas_data/bag.txt.gz')
gwas <- read.table(gwas_path, header = T)

for (i in 1:length(genes)) {
  protein <- genes[i]
  print(paste('coloc on', pheno, 'with', protein,  sep = ' '))
  if (protein %in% coloc_results$gene) {
    print(paste('skipping', protein))
    next
  }
  pqtl_path <- sprintf('deCODE_pqtl_data_significant/%s', protein)
  pqtls <- read.csv(pqtl_path, sep = '\t')
  
  input <- merge(pqtls, gwas, by.x = 'rsids', by.y = 'SNP', 
                     all = FALSE, suffixes=c("_pqtl","_gwas"))
  
  input <- input[!duplicated(input$rsids), ]
  input$Pval <- ifelse(input$Pval == 0, 1e-200, input$Pval)

  result <- coloc.abf(dataset1=list(pvalues = input$P, type = "quant",
                                    N = input$NMISS,
                                    snp = input$rsids),
                      
                      dataset2=list(pvalues=input$Pval, type="quant",
                                    N = input$N, snp = input$rsids),
                      MAF = input$MAF)

  coloc_result <- result$results %>% filter(SNP.PP.H4 >= 0.75)
  if (nrow(coloc_result) > 0) {
    coloc_result$gene <- genes[i]
    coloc_results <- rbind.data.frame(coloc_results, coloc_result)
  }
  
}

# drug_info <- read_excel('data/druggable_genome.xlsx', sheet = 'Data')
# coloc_results <- merge.data.frame(drug_info, coloc_results, by.x = 'ensembl_gene_id', by.y = 'gene')

write.csv(coloc_results, 'coloc_results/coloc_results_deCODE_bag.csv', row.names = F)

