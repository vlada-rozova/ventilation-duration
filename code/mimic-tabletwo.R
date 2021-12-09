library(tidyverse)
library(tableone)

setwd("~/Downloads")
dt <-read.csv("mimic-ft63.csv")

catVars = c("admission_location","insurance","language","ethnicity",
            "marital_status","gender","epinephrine", "vasopressin", 
            "dobutamine","norepinephrine","phenylephrine", "dopamine", "rrt",
            "sinus_rhythm","neuroblocker","congestive_heart_failure", 
            "cerebrovascular_disease", "dementia", "chronic_pulmonary_disease",
            "rheumatic_disease", "mild_liver_disease","diabetes_without_cc",
            "diabetes_with_cc","paraplegia", "renal_disease", "malignant_cancer",
            "severe_liver_disease", "metastatic_solid_tumor", "aids", 
            "over72h", "alive96h")

dt_table <- dt %>% 
        mutate_at(.vars=catVars, as.factor) %>% 
        select(-stay_id, -starttime, -endtime) 

contVars = names(dt_table)[-which(names(dt_table) %in% catVars)] 


## make a continuous variable table
tableone::CreateContTable(vars = contVars, data = dt_table, strata = "over72h") -> contTable
nonNormalVars <- c("hours_in_hosp_before_intubation","glucose_max")
print(contTable, nonnormal = nonNormalVars, test = T) %>%
        write.csv(.,"table2_cont.csv")

## make a categorical value table
tableone::CreateCatTable(vars = catVars, data = dt_table, strata = "over72h") -> catTable
print(catTable, test = T) %>%
        write.csv(.,"table2_cat.csv")
