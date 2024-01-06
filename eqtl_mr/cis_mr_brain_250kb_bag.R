library(TwoSampleMR)
library(ieugwasr)

source('ld_clump_local.R')


df <- read.csv('data_250kb_eqtls/druggable_brain_tissue_eqtls.csv')
print(length(unique(df$ensembl_gene_id)))

df_expo <- read_exposure_data('data_250kb_eqtls/druggable_brain_tissue_eqtls.csv',
                              sep = ',',
                              phenotype_col = 'ensembl_gene_id',
                              snp_col = 'Rsid', 
                              beta_col = 'regression_slope',
                              se_col = 'se',
                              pval_col = 'nominal_pval',
                              effect_allele_col = 'ALT',
                              other_allele_col = 'REF',
                              
                              chr_col = 'SNP_chr',
                              pos_col = 'position',
                              id_col = 'ensembl_gene_id'
)

genes <- unique(df_expo$exposure)

res_path <- 'results_250kb_eqtls_bag'

# results path
mr_res_path <- sprintf('%s/mr_res_brain.csv', res_path) 
pleiotropy_res_path <- sprintf('%s/pleiotropy_res_brain.csv', res_path)


if (file.exists(mr_res_path)) {
  res_all <- read.csv(mr_res_path)
  res_pleiotropy <- read.csv(pleiotropy_res_path)
} else {
  res_all <- data.frame(id.exposure = character(),
                        id.outcome = character(),
                        outcome = character(),
                        exposure = character(),
                        method = character(),
                        nsnp = numeric(),
                        b = numeric(),
                        se = numeric(),
                        pval = numeric())
  
  
  res_pleiotropy <- data.frame(id.exposure = character(),
                               id.outcome = character(),
                               outcome = character(),
                               exposure = character(),
                               egger_intercept = numeric(),
                               se = numeric(),
                               pval = numeric())
}


i <- 1

for (i in 1:length(genes)) {
  print(paste(i, 'analysing on', genes[i], sep = ' '))
  
  x <- subset.data.frame(res_all, id.exposure == genes[i])
  if (nrow(x) > 0) {
    print(sprintf('Done. Analying %s on BAG', genes[i]))
    next
  }
  
  # exposure data: e-QTLs for gene
  expo <- subset.data.frame(df_expo, exposure == genes[i])
  # expo <- subset.data.frame(df_expo, exposure == 'ENSG00000186868')
  expo$id.exposure <- genes[i]
  
  # f statistic > 10
  expo$fstat <- (expo$beta.exposure / expo$se.exposure)^2
  expo <- subset.data.frame(expo, fstat >= 10)
  
  
  # LD clump using EUR reference panel
  # expo_clamped <- clump_data(expo, clump_kb = 1000, clump_r2 = 0.1, pop = "EUR")
  expo$rsid <- expo$SNP
  expo$pval <- expo$pval.exposure
  expo_clamped <- ld_clump_local(expo, clump_kb = 1000, 
                                 clump_r2 = 0.1, 
                                 clump_p = 1,
                                 plink_bin = 'reference/plink',
                                 bfile = 'reference/EUR')
  
  
  
  # write.csv(expo_clamped[2, ], 'results_250kb_eqtls_bag/mapt_blood.csv', row.names = F)
  
  # load e-QTLs effect on outcome BAG
  bag3 <- try(read_outcome_data(
    filename = 'gwas_data/bag.txt.gz',
    sep = ' ',
    snps = expo_clamped$SNP,
    snp_col = "SNP",
    beta_col = "BETA",
    se_col = "SE",
    effect_allele_col = "A1",
    other_allele_col = "A2",
    chr_col = 'CHR',
    eaf_col = "MAF",
    pval_col = "P",
    pos_col = 'BP',
  ))
  
  if("try-error" %in% class(bag3)) {
    next
  }
  
  bag3$id.outcome <- 'BAG'
  
  dat <- harmonise_data(
    exposure_dat = expo_clamped, 
    outcome_dat = bag3,
    action = 2
  )
  
  if ((nrow(dat) == 0) || nrow(dat[dat$mr_keep == T, ] ) == 0) {
    # no snp present for MR analysis 
    print('not enough snp can be analysed.')
  } else {
    # MR analysis
    print(paste(genes[i], ':', nrow(dat), 'e-QTLs used for MR analysis', sep = ' '))
    res <- mr(dat)
    res_all <- rbind(res_all, res)
    pleiotropy_test <- mr_pleiotropy_test(dat)
    res_pleiotropy <- rbind(res_pleiotropy, pleiotropy_test)
    write.csv(res_all, 'results_250kb_eqtls_bag/mr_res_brain.csv', row.names = F)
    write.csv(res_pleiotropy, 'results_250kb_eqtls_bag/pleiotropy_res_brain.csv', row.names = F)
  }
}



