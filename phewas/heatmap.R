library(ggplot2)
library(reshape2)
library(RColorBrewer)
library(dplyr)
library(reshape2)
library(pheatmap)

pheno <- 'blood'
bag_type <- 'bag'

data_path <- sprintf('results_merged/heatmap_%s_%s.csv', pheno, bag_type)

annot_path <- sprintf('results_merged/heatmap_%s_%s_annot.csv', pheno, bag_type)

df <- read.csv(data_path, row.names = 'pheno')
df_annot <- read.csv(annot_path, row.names = 'pheno')
df_annot[is.na(df_annot)] <- ''

# pheno <- 'brain'
# bag_type <- 'bag'
# 
# data_path <- sprintf('results_merged/heatmap_%s_%s.csv', pheno, bag_type)
# 
# annot_path <- sprintf('results_merged/heatmap_%s_%s_annot.csv', pheno, bag_type)
# 
# df2 <- read.csv(data_path, row.names = 'pheno')
# df_annot2 <- read.csv(annot_path, row.names = 'pheno')
# df_annot2[is.na(df_annot2)] <- ''
# 
# df <- cbind(df, df2)
# df_annot <- cbind(df_annot, df_annot2)


if (pheno == 'blood') {
  show_row <- F
} else {
  show_row <- T
}

res_path <- sprintf('heatmap/heat_%s_%s.jpg', pheno, bag_type)

pheatmap(df,
         annotation_row = NULL,
         annotation_names_row = T,
         annotation_colors = 'white',
         # color = colorRampPalette(rev(brewer.pal(n = 7, name = "RdBu")))(100),
         color = colorRampPalette(c("royalblue","white","firebrick3"))(100),
         labels_row = rownames(df),
         legend = T,
         show_rownames = show_row, 
         show_colnames = T,
         cluster_rows = F,
         cluster_cols = T,
         angle_col = 90,
         fontsize_number = 15,
         border_color = "#EEEEEE",
         na_col = "white",
         fontsize = 10,
         treeheight_row = 0, 
         treeheight_col = 0,
         legend_labels = "effect dir * -log(p)",
         # color = myColor,
         fontsize_col = 12,
         fontsize_row = 12,
         display_numbers = df_annot,
         # breaks = myBreaks,
         cellwidth = 15,
         cellheight = 15,
         # height = 15,
         # width = 18,
         filename = res_path)
