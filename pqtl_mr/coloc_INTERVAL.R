library(coloc)
library(dplyr)
library(readxl)


phenos <- c('bag3', 'bagm3')
genes <- c('C1RL.9348.1.3.csv', 'MPL.3473.78.2.csv')

case_pop <- 5577 / 37990
i <- 1

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
  
  pqtl_path <- sprintf('interval_pqtl_merged/%s', protein)
  
  pqtls <- read.csv(pqtl_path)
  gwas_path <-sprintf('gwas_data/gwas_res_%s_healthy.txt', phenos[i])

  
  gwas <- read.table(gwas_path, header = T)
  
  input <- merge(pqtls, gwas, by = 'SNP', 
                     all = FALSE, suffixes=c("_pqtl", "_gwas"))
  input <- input[!duplicated(input$SNP), ]
  input$P_pqtl <- ifelse(input$P_pqtl == 0, 1e-200, input$P_pqtl)
  result <- coloc.abf(dataset1=list(pvalues = input$P_gwas, type = "cc",
                                    s = case_pop, N = input$NMISS,
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

# drug_info <- read_excel('data/druggable_genome.xlsx', sheet = 'Data')
# coloc_results <- merge.data.frame(drug_info, coloc_results, by.x = 'ensembl_gene_id', by.y = 'gene')

write.csv(coloc_results, 'coloc_results/coloc_results_INTERVAL.csv', row.names = F)

