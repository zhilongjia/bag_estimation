library(TwoSampleMR)
library(ieugwasr)

source('ld_clump_local.R')

df <- read.csv('pqtl_data/pqtls_interval_bag3.csv')
print(length(unique(df$id)))


df_expo <- read_exposure_data('pqtl_data/pqtls_interval_bag3.csv',
                              sep = ',',
                              phenotype_col = 'id',
                              snp_col = 'SNP',
                              beta_col = 'Effect',
                              se_col = 'StdErr',
                              pval_col = 'P',
                              effect_allele_col = 'Allele1',
                              other_allele_col = 'Allele2',
                              # eaf_col = 'AlleleB_all',
                              chr_col = 'Chromosome',
                              pos_col = 'position',
                              id_col = 'id'
)

genes <- unique(df_expo$exposure)

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

i <- 1

for (i in 1:length(genes)) {
  
  print(paste(i, ': analysing on', genes[i], sep = ' '))
  
  # exposure data: e-QTLs for gene
  expo <- subset.data.frame(df_expo, exposure == genes[i])
  expo$id.exposure <- genes[i]
  
  # f statistic > 10
  expo$fstat <- (expo$beta.exposure / expo$se.exposure)^2
  expo <- subset.data.frame(expo, fstat >= 10)
  
  # LD clump using EUR reference panel
  expo$rsid <- expo$SNP
  expo$pval <- expo$pval.exposure
  expo_clamped <- ld_clump_local(expo, clump_kb = 1000, 
                                 clump_r2 = 0.1, clump_p = 1,
                                 plink_bin = 'reference/plink',
                                 bfile = 'reference/EUR')
  
  # load e-QTLs effect on outcome BAG
  bag3 <- try(bag3 <- read_outcome_data(
    filename = 'gwas_data/gwas_res_bag3_healthy.txt',
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
  }
}

write.csv(res_all, 'results_interval_bag3/mr_res_blood.csv', row.names = F)
write.csv(res_pleiotropy, 'results_interval_bag3/pleiotropy_res_blood.csv', row.names = F)
