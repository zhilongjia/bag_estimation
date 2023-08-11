library(coloc)
library(dplyr)
library(readxl)


phenos <- c('bag3', 'bag3', 'bag3','bagm3')
genes <- c('2855_49_MAPK3_ERK_1', 
           '6207_10_PSAP_prosaposin', 
           '6609_22_CNP_CN37',
           '5939_42_TNFSF12_TWEAK')

case_pop <- 5577 / 37990

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

i <- 1
for (i in 1:length(genes)) {
  protein <- genes[i]
  print(paste('coloc on', phenos[i], 'with', protein,  sep = ' '))
  
  pqtl_path <- sprintf('deCODE_pqtl_data_significant/%s.txt.gz', protein)
  pqtls <- read.csv(pqtl_path, sep = '\t')
  
  gwas_path <-sprintf('gwas_data/gwas_res_%s_healthy.txt', phenos[i])
  gwas <- read.table(gwas_path, header = T)
  
  input <- merge(pqtls, gwas, by.x = 'rsids', by.y = 'SNP', 
                     all = FALSE, suffixes=c("_pqtl","_gwas"))
  
  input <- input[!duplicated(input$rsids), ]
  input$Pval <- ifelse(input$Pval == 0, 1e-200, input$Pval)

  result <- coloc.abf(dataset1=list(pvalues = input$P, type = "cc",
                                    s = case_pop, N = input$NMISS,
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

write.csv(coloc_results, 'coloc_results/coloc_results_deCODE.csv', row.names = F)

