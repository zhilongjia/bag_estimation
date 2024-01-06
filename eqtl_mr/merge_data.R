eqtl <- read.table('data/2019-12-11-cis-eQTLsFDR0.05-ProbeLevel-CohortInfoRemoved-BonferroniAdded.txt',
                   header = T)

eqtl <- subset.data.frame(eqtl, eqtl$FDR < 0.05)

snp_info <- read.table('data/2018-07-18_SNP_AF_for_AlleleB_combined_allele_counts_and_MAF_pos_added.txt', header = T)


eqtl2 <- merge.data.frame(eqtl, snp_info, by = 'SNP')

eqtl2 <- eqtl2[order(eqtl2$Gene), ]

write.table(eqtl2, 'data/blood_eqtls.txt', row.names = F, sep = '\t')


