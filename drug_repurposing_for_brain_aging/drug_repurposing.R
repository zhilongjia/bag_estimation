
# the DgiDB output
dgidb_raw <- readxl::read_xlsx("./data/dgidb.xlsx", sheet="dgidb_res")

# to compatible with drugbank name
dgidb_raw[which(dgidb_raw$DRUG=="INAMRINONE"),"drug"] <- "amrinone"

################################################################################
# annotating drugs using drugbank
library(dplyr)

con <- DBI::dbConnect(RMariaDB::MariaDB(),
                      dbname="drugbank",
                      host = "localhost",
                      user = "zjia" )

drug <- tbl(con, "drug")%>% collect() 
drug_groups <- tbl(con, "drug_groups") %>% collect() 
drug_groups <- dplyr::group_by(drug_groups, `drugbank-id`) %>% dplyr::summarise(status=paste(group, collapse = ","))
drug_pharmacology <- tbl(con, "drug_pharmacology") %>% collect() 
drug_external_identifiers <- tbl(con, "drug_external_identifiers") %>% collect() 

DrugBankdrugs <-  dplyr::rename(drug, `drugbank-id`=primary_key ) %>% 
    dplyr::mutate(names=tolower(name) ) %>% 
    dplyr::left_join(drug_groups ) %>%
    dplyr::left_join(drug_pharmacology, by=c("drugbank-id"="drugbank_id") ) %>%
    collect() 

# only keep drugs in drugbank.
MRgene_DrugBankdrugs_refined <- dplyr::filter(DrugBankdrugs, names %in% dgidb_raw$drug ) %>% 
    dplyr::select(`drugbank-id`, names, status, description, indication, mechanism_of_action, toxicity) %>% 
    dplyr::mutate(description=gsub("\r+|\n+",".",description), indication=gsub("\r+|\n+",".",indication), 
                  mechanism_of_action=gsub("\r+|\n+",".",mechanism_of_action), toxicity=gsub("\r+|\n+",".",toxicity) )
################################################################################
# drugbankID to ChEMBLID.

drugbID_chEMBLID <- dplyr::filter(drug_external_identifiers, resource=="ChEMBL", parent_key %in% MRgene_DrugBankdrugs_refined$`drugbank-id` ) %>% 
    dplyr::rename(`drugbank-id`=parent_key) %>% dplyr::select(-resource)

# add CHEMBL550495 for Methylene blue
drugbID_chEMBLID <- rbind(drugbID_chEMBLID, c("CHEMBL550495", "DB09241") )

MRgene_DrugBankdrugs_refined <- dplyr::left_join(MRgene_DrugBankdrugs_refined, drugbID_chEMBLID )

################################################################################

dgidb_drugbank <- dplyr::left_join(dgidb_raw, MRgene_DrugBankdrugs_refined, by=c("drug"="names")) 

DBI::dbDisconnect(con)
################################################################################
# indication of ChEMBLID
con1 <- DBI::dbConnect(RMariaDB::MariaDB(),
                      dbname="chEMBL",
                      host = "localhost",
                      user = "zjia" )

drug_indication <- tbl(con1, "drug_indication") %>% collect() 
molecule_dictionary <- tbl(con1, "molecule_dictionary") %>% collect() 

chemblID_ind <- dplyr::left_join(drug_indication, molecule_dictionary) %>% 
    dplyr::filter(chembl_id %in% as.vector(na.omit(MRgene_DrugBankdrugs_refined$identifier)) ) %>% 
    dplyr::mutate(drug=tolower(pref_name) )

gene_drug_indication <- dplyr::select(dgidb_drugbank, gene, drug, `drugbank-id`, interaction_types ) %>% 
    dplyr::left_join(drugbID_chEMBLID) %>% 
    dplyr::left_join(chemblID_ind, by=c("identifier"="chembl_id" ) )

# merge multi-row indication into one row
gene_drug_Merged_indication <- dplyr::select(gene_drug_indication, gene, drug.x, interaction_types,
                                      identifier, max_phase_for_ind, efo_term, max_phase) %>% 
    group_by(gene, drug.x) %>% 
    dplyr::reframe(efo_terms=paste(efo_term, collapse = ","), interaction_types=unique(interaction_types),
                     max_phase_for_ind=paste(max_phase_for_ind, collapse = ","), max_phase=unique(max_phase) )

DBI::dbDisconnect(con1)
################################################################################
################################################################################
# add MR + coloc and beta infor.

MR3_raw <- readxl::read_xlsx("./data/results_mr_coloc_significant.xlsx", sheet= "BAG eqtl" )

MR3_raw <- MR3_raw[,c("hgnc_names","eQTL_type", "b")] %>% tidyr::pivot_wider(names_from =eQTL_type, values_from = b )


coloc3_raw <- readxl::read_xlsx("./data/results_mr_coloc_significant.xlsx", sheet= "BAG eqtl coloc" )


MR_coloc3 <- dplyr::left_join(MR3_raw, coloc3_raw[,c("hgnc_names","SNP.PP.H4")] ) %>% 
    dplyr::rename(blood_eQTL=`Blood eQTL`, Brain_eQTL=`Brain Tissue eQTL`, PPH4_eQTL=SNP.PP.H4)

################################################################################
################################################################################
# MR pQTL results
MR_pQTL_3 <- readxl::read_xlsx("./data/results_mr_coloc_significant.xlsx", sheet="BAG pQTL") %>% 
    dplyr::select(Source, Gene, b)%>% 
    tidyr::pivot_wider(names_from=Source, values_from=b ) %>% 
    dplyr::rename(pQTL_deCODE_b=deCODE, pQTL_INTERVAL_b=INTERVAL)

MR_pQTL_colc <- readxl::read_xlsx("./data/results_mr_coloc_significant.xlsx", sheet="BAG pQTL coloc") %>% 
    dplyr::select(Gene, SNP.PP.H4, Source) %>% 
    tidyr::pivot_wider(names_from=Source, values_from=SNP.PP.H4 )

MR_pQTL_merged <- dplyr::full_join(MR_pQTL_3, MR_pQTL_colc) %>% 
    dplyr::rename(hgnc_names=Gene, pQTL_deCODE_PPH4=deCODE, pQTL_INTERVAL_PPH4=INTERVAL )


MR_coloc_res <- dplyr::full_join(MR_coloc3, MR_pQTL_merged) %>% 
    dplyr::rename(gene=hgnc_names) 

MR_coloc_res_stat <- dplyr::group_by(MR_coloc_res, gene) %>% 
    tidyr::pivot_longer(cols=2:ncol(.), names_to="evidenceFrom", values_to="value" ) %>% 
    dplyr::summarise(evidence_num=sum(!is.na(value)) ) %>% 
    dplyr::arrange(desc(evidence_num), )


MR_coloc_res <- dplyr::left_join(MR_coloc_res, MR_coloc_res_stat) %>% 
    dplyr::arrange(desc(evidence_num), )



# Suppl. data 20
readr::write_tsv(MR_coloc_res, file="./results/MR_coloc_res.tsv", na="")

anti_aging_drugs_raw <-  readxl::read_xlsx("./data/dgidb.xlsx", sheet="antiaging_drugs")
anti_aging_drugs <- anti_aging_drugs_raw$drug

gene_drug_indication_beta <- dplyr::left_join(gene_drug_indication, MR_coloc_res) %>% 
    dplyr::left_join(anti_aging_drugs_raw, by=c("drug.x"="drug") ) %>% 
    dplyr::mutate(agingdrug=ifelse(!is.na(PMIDs), "Y", "" ) )


gene_drug_Mergedindication_beta <- dplyr::left_join(gene_drug_Merged_indication, MR_coloc_res) %>% 
    dplyr::left_join(anti_aging_drugs_raw, by=c("drug.x"="drug") ) %>% 
    dplyr::mutate(agingdrug=ifelse(!is.na(PMIDs), "Y", "" ),
                  beta_dir= dplyr::case_when( blood_eQTL > 0 ~ 1,
                                              blood_eQTL < 0 ~ -1,
                                              Brain_eQTL > 0 ~ 1,
                                              Brain_eQTL < 0 ~ -1) ) %>% 
    dplyr::filter(max_phase >= 0.5 | drug.x == "ro-5458640" ) %>% 
    dplyr::mutate(interaction_types=replace(interaction_types, drug.x == "ro-5458640", "antibody") )


################################################################################
table(gene_drug_Mergedindication_beta$interaction_types )

gene_drug_Mergedindication_beta_direction <- dplyr::mutate(gene_drug_Mergedindication_beta, 
     direction= dplyr::case_when(
         !is.na(interaction_types) & beta_dir >0 & grepl("antagonist|inhibitor|antibody|inverse agonist", interaction_types) ~ "Y",
         !is.na(interaction_types) & beta_dir <0 & grepl("^agonist|partial agonist|activator", interaction_types) ~ "Y",
         !is.na(interaction_types) & beta_dir >0 & grepl("^agonist|partial agonist|activator", interaction_types) ~ "N",
         !is.na(interaction_types) & beta_dir <0 & grepl("antagonist|inhibitor|antibody|inverse agonist", interaction_types) ~ "N",
         is.na(interaction_types) ~ NA
     ) )

# Suppl. data 22
readr::write_tsv(gene_drug_Mergedindication_beta_direction, file="./results/gene_drug_Mergedindication_beta_direction.tsv", na="")

#
save.image("./results/drug_repurposing_for_brain_aging.RData")



