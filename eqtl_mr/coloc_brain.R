library(coloc)
library(dplyr)
library(readxl)

gwas <- read.table('gwas_data/gwas_res_bagm3_healthy.txt', header = T)

eqtls <- read.csv('data_250kb_eqtls/druggable_brain_tissue_eqtls.csv')

input_all <- merge(eqtls, gwas, by.x = 'Rsid', by.y = 'SNP', 
                   all = FALSE, suffixes=c("_eqtl","_gwas"))

# input <- input[order(input$Rsid, abs(input$SNP_distance_to_TSS)), ]
# input <- input[!duplicated(input$Rsid), ]

genes <- unique(input_all$ensembl_gene_id)

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

for (i in 1:length(genes)) {
  input <- subset.data.frame(input_all, ensembl_gene_id == genes[i])
  result <- coloc.abf(dataset1=list(pvalues = input$P, type = "cc",
                                    s = case_pop, N = input$NMISS, snp = input$Rsid),
                      dataset2=list(pvalues=input$nominal_pval, type="quant",
                                    N = input$number_of_SNPs_tested, snp = input$Rsid),
                      MAF=input$MAF)
  
  
  coloc_result <- result$results %>% filter(SNP.PP.H4 >= 0.75)
  if (nrow(coloc_result) > 0) {
    coloc_result$gene <- genes[i]
    coloc_results <- rbind.data.frame(coloc_results, coloc_result)
  }
  
}

drug_info <- read_excel('data/druggable_genome.xlsx', sheet = 'Data')
coloc_results <- merge.data.frame(drug_info, coloc_results, by.x = 'ensembl_gene_id', by.y = 'gene')

# coloc_results <- coloc_results[, c(18, 1:17)]
write.csv(coloc_results, 'coloc_results/coloc_brain_bagm3.csv', row.names = F)

