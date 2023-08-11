library(TwoSampleMR)
library(ieugwasr)

source('ld_clump_local.R')

pheno_gwas_id <- read.csv('data/pheno_gwas_ids.csv')

# read genes significant for BAG
df_expo <- read_exposure_data('eqtls_bagm3/significant_brain_eqtls.csv',
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
phenos <- pheno_gwas_id$pheno

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

  # outcome data: phenotypes
  j <- 1
  pheno <- 1
  for (j in 1:length(phenos)) {
    sprintf('%d: analying %s on %s', i, genes[i], phenos[j])
    
    if (pheno_gwas_id[j, 3] == 0) {
      # load outcome from open gwas
      # add while(T) in extracting outcome data from open gwas to \
      # avoid server code: 502 or any other network issues.
      try_times <- 0
      
      while (T) {
        print(paste('extract from', pheno_gwas_id[j, 2]))
        pheno <- try(extract_outcome_data(
          snps = expo_clamped$SNP,
          outcomes = pheno_gwas_id[j, 2]
        ))
        if(("try-error" %in% class(pheno)) || is.numeric(pheno)) {
          print('import outcome data from opengwas failed due to
              network connection, try again')
          try_times <- try_times + 1
        } else {
          break
        }
        if (try_times >= 20) {
          print('Tried too many times, maybe there is no network connnection or the open gwas server is currently unreachable. Abort.')
          break
        }
      }

    } else {
      # load outcome from local gwas summary data
      # load IL data
      if (grepl(pattern = 'IL', phenos[j])) {
        try(pheno <- read_outcome_data(
          filename = pheno_gwas_id[j, 2],
          sep = '\t',
          snps = expo_clamped$SNP,
          snp_col = "variant_id",
          beta_col = "beta",
          se_col = "standard_error",
          effect_allele_col = "effect_allele",
          other_allele_col = "other_allele",
          chr_col = 'chromosome',
          eaf_col = "effect_allele_frequency",
          pval_col = "p_value",
          pos_col = 'base_pair_location',
        ))
      }
      # load MEGA-STROKE data
      else if (grepl(pattern = 'stroke', phenos[j])) {
        try(pheno <- read_outcome_data(
          filename = pheno_gwas_id[j, 2],
          sep = ' ',
          snps = expo_clamped$SNP,
          snp_col = "MarkerName",
          beta_col = "Effect",
          se_col = "StdErr",
          effect_allele_col = "Allele1",
          other_allele_col = "Allele2",
          # chr_col = 'chromosome',
          eaf_col = "Freq1",
          pval_col = "P-value",
          # pos_col = 'base_pair_location',
        ))
      }

      if(("try-error" %in% class(pheno))) {
          pheno <- NULL
      }
    }

    # no snp find in both exposure data and outcome data
    if (is.null(pheno)) {
      sprintf('No snp matches for exposure %s and outcome %s. or the opengwas sever is not available yet',
              genes[i], phenos[j])
      next
    }

    # set outcome name
    pheno$id.outcome <- phenos[j]

    # harmonise data
    dat <- harmonise_data(
      exposure_dat = expo_clamped,
      outcome_dat = pheno,
      action = 2
    )

    # mr analysis
    if ((nrow(dat) == 0) || nrow(dat[dat$mr_keep == T, ] ) == 0) {
      # no snp present for MR analysis
      print('not enough snp can be analysed.')
    } else {
      # MR analysis
      print(paste(genes[i], ':', nrow(dat), 'e-QTLs used for MR analysis', sep = ' '))
      res <- mr(dat)
      res_all <- rbind(res_all, res)
      if (nrow(dat[dat$mr_keep == T, ] ) > 3) {
        pleiotropy_test <- mr_pleiotropy_test(dat)
        res_pleiotropy <- rbind(res_pleiotropy, pleiotropy_test)
      }
      
    }
  }
}

write.csv(res_all, 'results_bagm3/phwas_res_brain.csv', row.names = F)
write.csv(res_pleiotropy, 'results_bagm3/pleiotropy_res_brain.csv', row.names = F)

