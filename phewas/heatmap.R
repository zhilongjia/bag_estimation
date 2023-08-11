library(ggplot2)
library(reshape2)
library(RColorBrewer)
library(dplyr)
library(reshape2)
library(pheatmap)

pheno <- 'brain'

data_path <- sprintf('results_merged/heatmap_%s.csv', pheno)

annot_path <- sprintf('results_merged/heatmap_%s_annot.csv', pheno)

df <- read.csv(data_path, row.names = 'pheno')
df_annot <- read.csv(annot_path, row.names = 'pheno')
df_annot[is.na(df_annot)] <- ''

if (pheno == 'blood') {
  show_row <- F
} else {
  show_row <- T
}

res_path <- sprintf('heatmap/heat_%s_2.jpg', pheno)

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
         cluster_cols = F,
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
         filename = res_path)
