library(ggplot2)
library(dplyr)
library(ggrepel)
library(patchwork)
library(readxl)
library(ggthemes)

# read MR results 
tissue <- 'blood'
pheno <- 'bag'

data_path <- sprintf('results_pqtls_merged/pqtl_%s.xlsx', pheno)

df <- read_excel(data_path)

# read colocalization results
df_coloc <- read.csv(sprintf('coloc_results/coloc_results_bag.csv') )
df_coloc <- subset.data.frame(df_coloc, select = c('Gene', 'snp', 'SNP.PP.H4'))
# drop duplicated genes
df_coloc <- df_coloc[!duplicated(df_coloc$Gene), ]

# remove results with egger_pval < 0.05
df[is.na(df$egger_pval), ]$egger_pval <- 1
df <- df[df$egger_pval >= 0.05, ]
df$significant <- ifelse(df$fdr < 0.05, 1, 0)
# use only wald ratio and IVW results
unique(df$method)
df <- subset.data.frame(df, method %in% c('Inverse variance weighted', 'Wald ratio'))


df <- merge.data.frame(df, df_coloc, by.x = 'hgnc_names', by.y = 'Gene', all.x = TRUE)
df[is.na(df$SNP.PP.H4),  ]$SNP.PP.H4 <- 0
df[is.na(df$snp), ]$snp <- 'None'
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
  # df$is_coloc == 1 ~ 'Colocalized',
  TRUE ~ 'None'
)

print(unique(df$group))

if (length(unique(df$group)) == 2) {
  df$group <- factor(df$group, levels = c('Colocalized & Significant', 'None'))
} else if (length(unique(df$group)) == 3) {
  df$group <- factor(df$group, levels = c('Colocalized & Significant', 'Significant', 'None'))
} else {
  df$group <- factor(df$group, levels = c('Colocalized & Significant', 'Significant', 'Colocalized', 'None'))
}
print(unique(df$group))

# color for each group
if (length(unique(df$group)) == 2) {
  mycol <- c("#EA5D4E","#d8d8d8")
} else {
  mycol <- c("#EA5D4E","#317EAB", "#d8d8d8")
}


xmax <- ceiling(max(abs(df$b)))
ymax <- ceiling(max(abs(df$neg_log10p)))

if (pheno == 'bag3') {
  ptitle <- 'pQTLs for BAG > 3 years'
} else if (pheno == 'bagm3') {
  ptitle <- 'pQTLs for BAG < -3 years'
} else {
  ptitle <- 'pQTLs for BAG'
}
# drop duplicated genes
# df <- df[!duplicated(df$Gene), ]

p <- ggplot(data = df, aes(x = b, y = -log10(pval), shape = Source, color = group)) + #建立映射
  scale_colour_manual(name = "", values = alpha(mycol, 0.7)) + 
  geom_point(size = 2.2) + 
  scale_y_continuous(expand = expansion(add = c(0, 0)),
                   limits = c(0, ymax * 1.5), breaks = seq(0, ymax * 1.5, by = 1)) +
  scale_x_continuous(limits = c(-0.55, 0.55), breaks = seq(-0.5, 0.5, by = 0.25)) +
    # labs()
    xlab('Effect size') + ylab('-log10(p-value)') + ggtitle(ptitle) +
    theme_few() + theme(
      plot.title = element_text(size = 16, face = 'bold'),
      axis.title = element_text(size = 16),
      axis.text = element_text(size = 15),
      legend.title = element_blank(),
      legend.text = element_text(size = 11),
      legend.background = element_blank(),
      legend.spacing.y = unit(0.05, "cm"),
      panel.border = element_rect(linewidth = 1.2),
      legend.position = c(0.8, 0.84),
      # legend.position = 'None'
    ) + guides(color = guide_legend(order = 1),   # 颜色图例排在第一位
               shape = guide_legend(order = 2)) +  # 形状图例排在第二位
    geom_vline(xintercept = 0, linewidth = 1.,
               color = alpha("black", 0.5) ,lty = "dashed") #垂直阈值线

p

# add labels for significant genes
df_evidence <- read_excel('E:/OneDrive/work/BrainAge/source_codes/results_mr_coloc_significant.xlsx',
                          sheet = 'evidence in xQTL')
# genes with at least 2 pieces of evidence
genes_preferred <- subset(df_evidence, evidence_count >= 2)$hgnc_names
genes_preferred <- unique(genes_preferred)
p2 <- p +
  geom_text_repel(data = subset(df, significant == 1),
                  aes(x = b, y = -log10(pval), color = group, label = hgnc_names),
                  force = 80, 
                  # color = '#EB4232', 
                  size = 5,
                  fontface = "italic",
                  point.padding = 0.5, 
                  hjust = 0.5,
                  # label.size = 
                  arrow = arrow(length = unit(0.1, "cm"),
                                type = "open", ends = "last"),
                  segment.color="grey",
                  segment.size = 0.0,
                  nudge_x = 0.,
                  nudge_y = 0.1,
                  max.overlaps = 30)
  
p2 <- p2 + theme(text = element_text(family = 'Arial'))

res_path <- sprintf('E:/OneDrive/work/BrainAge/figs/eqtl_pqtl/pqtl_%s_for_fig1.jpg', pheno)

ggsave(res_path, p2, width = 7, height = 6, dpi = 300)



