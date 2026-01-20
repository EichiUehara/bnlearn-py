
.libPaths(c("R_libs", .libPaths()))
library(bnlearn)

set.seed(42)

# Create output dir
if (!dir.exists("tests/data")) {
  dir.create("tests/data", recursive = TRUE)
}

# Load Alarm dataset (simulated patient monitoring)
data(alarm)

# Case 1: Small subset
# Columns: HYP, STKV, CO, LVV, PCWP
nodes_small <- c("HYP", "STKV", "CO", "LVV", "PCWP")
data_small <- alarm[, nodes_small]
data_small <- data_small[1:1000, ]

write.csv(data_small, "tests/data/small_data.csv", row.names = FALSE)

cat("Learning small network...\n")
learned_small <- hc(data_small, score = "bic")
arcs_small <- learned_small$arcs
write.csv(arcs_small, "tests/data/small_arcs_R.csv", row.names = FALSE)
score_small <- score(learned_small, data_small, type = "bic")
# Write with high precision
write(sprintf("%.10f", score_small), "tests/data/small_score_R.txt")

# Case 2: Full Alarm dataset
data_large <- alarm[1:2000, ]
write.csv(data_large, "tests/data/large_data.csv", row.names = FALSE)

cat("Learning large network...\n")
learned_large <- hc(data_large, score = "bic")
arcs_large <- learned_large$arcs
write.csv(arcs_large, "tests/data/large_arcs_R.csv", row.names = FALSE)
score_large <- score(learned_large, data_large, type = "bic")
# Write with high precision
write(sprintf("%.10f", score_large), "tests/data/large_score_R.txt")

cat("Done.\n")
