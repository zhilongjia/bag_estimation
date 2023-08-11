df <- read.csv('data_50kb_eqtls/druggable_brain_tissue_eqtls.csv')

length(unique(df$ensembl_gene_id))

# se = abs(beta / qnrom(p / 2))
df$se <- abs(df$regression_slope / qnorm(df$nominal_pval / 2))

write.csv(df, 'data_50kb_eqtls/druggable_brain_tissue_eqtls.csv', row.names = F)


df <- read.csv('data_50kb_eqtls/druggable_blood_eqtls.csv')

length(unique(df$ensembl_gene_id))

# calculate beta, se by zscore (z):
# beta = z / sqrt(2 * frq *(1 − frq) *(n + z^2))
# se = 1 / sqrt(2 * frq * (1 - frq) * (n + z^2))
# where frq is the effective allele frequency, n is the sample size


df$beta <- df$Zscore / sqrt(2 * df$AlleleB_all * (1 - df$AlleleB_all) * (df$NrSamples + df$Zscore^2))
df$se <- 1 / sqrt((2 * df$AlleleB_all) * (1 - (df$AlleleB_all)) * (df$NrSamples + (df$Zscore^2)))

write.csv(df, 'data_50kb_eqtls/druggable_blood_eqtls.csv', row.names = F)
