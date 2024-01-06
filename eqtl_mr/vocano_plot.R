library(ggplot2)
library(dplyr)
library(ggrepel)
library(patchwork)
library(readxl)
library(ggthemes)
library(readxl)
# read MR results 
tissue <- 'brain'
pheno <- 'bag'

data_path <- sprintf('results_250kb_eqtls_%s/mr_result_%s_preprocessed.csv', pheno, tissue)

df <- read.csv(data_path)

# read colocalization results
df_coloc <- read.csv(sprintf('coloc_results/coloc_%s_%s.csv', tissue, pheno) )
df_coloc <- subset.data.frame(df_coloc, 
                              select = c('ensembl_gene_id', 'snp', 'SNP.PP.H4'))


# remove results with egger_pval < 0.05
df <- df[df$egger_pval >= 0.05, ]

# use only wald ratio and IVW results
unique(df$method)
df <- subset.data.frame(df, method %in% c('Inverse variance weighted', 'Wald ratio'))


df <- merge.data.frame(df, df_coloc, by = 'ensembl_gene_id', all.x = TRUE)
df[is.na(df$SNP.PP.H4),  ]$SNP.PP.H4 <- 0
df$is_coloc <- ifelse(df$SNP.PP.H4 >= 0.75, 1, 0)

df$is_sig_coloc <- ifelse(df$is_coloc == 1 & df$significant == 1, 1, 0)
table(df$significant)
table(df$is_sig_coloc)
df$pval <- df$pval + 1e-10

# plotting
df$neg_log10p <- -log10(df$pval)

df$group <- case_when(
  df$is_sig_coloc == 1 ~ "Colocalized & Significant",
  df$significant == 1 ~ "Significant",
  TRUE ~ 'None'
)

print(unique(df$group))

if (length(unique(df$group)) == 2) {
  df$group <- factor(df$group, levels = c('Significant', 'None'))
} else {
  df$group <- factor(df$group, levels = c('Colocalized & Significant', 'Significant', 'None'))
}
print(unique(df$group))

# color for each group
if (length(unique(df$group)) == 2) {
  mycol <- c("#317EAB","#d8d8d8")
} else {
  mycol <- c("#EA5D4E","#317EAB","#d8d8d8")
}


xmax <- ceiling(max(abs(df$b)))
ymax <- ceiling(max(abs(df$neg_log10p)))

if (tissue == 'blood' && pheno == 'bag3') {
  ptitle <- 'Blood eQTLs for BAG > 3 years'
  leg_pos <- c(0.8, 0.9)
} else if (tissue == 'blood' && pheno == 'bagm3') {
  ptitle <- 'Blood eQTLs for BAG < -3 years'
  leg_pos <- c(0.8, 0.9)
} else if (tissue == 'brain' && pheno == 'bag3') {
  ptitle <- 'Brain tissue eQTLs for BAG > 3 years'
  leg_pos <- c(0.8, 0.9)
} else if (tissue == 'brain' && pheno == 'bagm3') {
  ptitle <- 'Brain tissue eQTLs for BAG < -3 years'
  leg_pos <- c(0.8, 0.93)
} else if (tissue == 'brain' && pheno == 'bag') {
  ptitle <- 'Brain tissue eQTLs for BAG'
  leg_pos <- c(0.8, 0.93)
} else if (tissue == 'blood' && pheno == 'bag') {
  ptitle <- 'Blood eQTLs for BAG'
  leg_pos <- c(0.8, 0.93)
}


p <- ggplot(data = df, aes(x = b, y = -log10(pval), color = group)) + #建立映射
  geom_point(size = 2.2) + 
  scale_y_continuous(expand = expansion(add = c(0, 0)),
                   limits = c(0, ymax * 1.2), breaks = seq(0, 12, by = 2)) +
  scale_x_continuous(expand = expansion(add = c(0, 0)),
                     limits = c(-2, 2), breaks = seq(-2, 2, by = 1)) +
    # labs()
    xlab('Effect size') + ylab('-log10(p-value)') + ggtitle(ptitle) +
    scale_colour_manual(name = "", values = mycol) + 
    theme_few() + theme(
      plot.title = element_text(size = 16, face = 'bold'),
      axis.title = element_text(size = 16),
      axis.text = element_text(size = 15),
      legend.text = element_text(size = 11),
      legend.background = element_blank(),
      panel.border = element_rect(linewidth = 1),
      legend.position = leg_pos,
    ) + 
    geom_vline(xintercept = 0, linewidth = 1.,
               color = alpha("black", 0.5) ,lty = "dashed") #垂直阈值线

p

# add labels for significant genes
df_evidence <- read_excel('E:/OneDrive/work/BrainAge/source_codes/results_mr_coloc_significant.xlsx',
                          sheet = 'evidence in xQTL')
# genes with at least 2 pieces of evidence
genes_preferred <- subset(df_evidence, evidence_count >= 2)$hgnc_names

if (tissue == 'blood') {
  df_sig <- subset(df, hgnc_names %in% genes_preferred & significant == 1)
  nudge_x <- 0.1
  nudge_y <- 1.0
  seg_size <- 0.3
  force <- 80
} else {
  df_sig <- subset(df, hgnc_names %in% genes_preferred & significant == 1)
  nudge_x <- -0.
  nudge_y <- 0.1
  seg_size <- 0.3
  force <- 100
}

p2 <- p +
  geom_text_repel(data = df_sig,
                  aes(x = b, y = -log10(pval), color = group, label = hgnc_names),
                  force = force, 
                  # color = '#EB4232', 
                  size = 4,
                  fontface = "italic",
                  point.padding = 0.5, 
                  hjust = 0.5,
                  # label.size = 
                  arrow = arrow(length = unit(0.1, "cm"),
                                type = "open", ends = "last"),
                  segment.color="grey",
                  segment.size = seg_size,
                  # nudge_x = -0.1,
                  # nudge_y = 0,
                  nudge_x = nudge_x,
                  nudge_y = nudge_y,
                  max.overlaps = 30)
  
p2 <- p2 + theme(text = element_text(family = 'Arial'))

res_path <- sprintf('E:/OneDrive/work/BrainAge/figs/eqtl_pqtl/eqtls_%s_%s.jpg', pheno, tissue)

ggsave(res_path, p2, width = 7, height = 6, dpi = 300)



