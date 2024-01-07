
library(org.Hs.eg.db)
library(ReactomePA)
library(clusterProfiler)
# Gene symbols to EntrezID
symbol2entrezID <- function(gene_symbols) {
    symbol_ENTREZID <- clusterProfiler::bitr(gene_symbols, fromType="SYMBOL", toType="ENTREZID", OrgDb="org.Hs.eg.db" )
    return(symbol_ENTREZID$ENTREZID)
}

# MR gene
MRGene_raw <- readxl::read_xlsx("./data/dgidb.xlsx", sheet= "dgidb_input" )

MRGenes <- unique(MRGene_raw$hgnc_names)

DEG_EntrezID <- symbol2entrezID( MRGenes)

x <- enrichPathway(gene=DEG_EntrezID, pvalueCutoff = 1, qvalueCutoff = 1)
dotplot(x, showCategory=10, label_format=60, font.size=5, color="pvalue")

