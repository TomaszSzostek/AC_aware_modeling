################################################################################
# Master Script: Generate All Publication Figures (500 DPI)
################################################################################

cat("\n", strrep("=", 80), "\n")
cat("PUBLICATION FIGURES GENERATION (500 DPI)\n")
cat(strrep("=", 80), "\n\n")

# Check working directory
if (basename(getwd()) != "AC_aware_modeling") {
  if (file.exists("config.yml")) {
    # Already in root
  } else if (file.exists("../config.yml")) {
    setwd("..")
  } else {
    stop("Run from AC_aware_modeling directory")
  }
}

cat(sprintf("Working directory: %s\n\n", getwd()))

# Generate figures
figures_ok <- c()
figures_fail <- c()

# Figure 1
cat(strrep("=", 80), "\n", "Figure 1: Reverse QSAR\n", strrep("=", 80), "\n")
tryCatch({
  source("R_reporting/figure_1.R", echo = FALSE, print.eval = FALSE)
  figures_ok <- c(figures_ok, "Figure 1")
}, error = function(e) {
  cat(sprintf("\nFailed: %s\n\n", e$message))
  figures_fail <- c(figures_fail, "Figure 1")
})

# Figure 2
cat("\n", strrep("=", 80), "\n", "Figure 2: Predictive Models\n", strrep("=", 80), "\n")
tryCatch({
  source("R_reporting/figure_2.R", echo = FALSE, print.eval = FALSE)
  figures_ok <- c(figures_ok, "Figure 2")
}, error = function(e) {
  cat(sprintf("\nFailed: %s\n\n", e$message))
  figures_fail <- c(figures_fail, "Figure 2")
})

# Figure 3
cat("\n", strrep("=", 80), "\n", "Figure 3: CAFE LATE Analysis\n", strrep("=", 80), "\n")
tryCatch({
  source("R_reporting/figure_3.R", echo = FALSE, print.eval = FALSE)
  figures_ok <- c(figures_ok, "Figure 3")
}, error = function(e) {
  cat(sprintf("\nFailed: %s\n\n", e$message))
  figures_fail <- c(figures_fail, "Figure 3")
})

# Summary
cat("\n", strrep("=", 80), "\n", "SUMMARY\n", strrep("=", 80), "\n\n")
cat(sprintf("Generated: %d figure(s)\n", length(figures_ok)))
for (fig in figures_ok) cat(sprintf("  • %s\n", fig))

if (length(figures_fail) > 0) {
  cat(sprintf("\nFailed: %d figure(s)\n", length(figures_fail)))
  for (fig in figures_fail) cat(sprintf("  • %s\n", fig))
}

cat("\nOutput: R_reporting/figures/\n\n")
cat(strrep("=", 80), "\n", "DONE\n", strrep("=", 80), "\n\n")
