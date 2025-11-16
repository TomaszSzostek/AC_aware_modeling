################################################################################
# 00_setup.R
# Setup: Packages, Theme, and Utilities
################################################################################

# Install packages if missing
packages <- c("tidyverse", "ggrepel", "cowplot", "RColorBrewer", "viridis", "ggridges", "patchwork", "magick")

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
    library(pkg, character.only = TRUE)
  } else {
    suppressPackageStartupMessages(library(pkg, character.only = TRUE))
  }
}

# =============================================================================
# DEFAULT THEME
# =============================================================================

# Theme as object (for backward compatibility)
default_theme <- theme(
  panel.border = element_rect(colour = "black", linewidth = 1, fill = NA),
  panel.background = element_blank(),
  plot.title = element_text(hjust = 0.5, face = "plain"),
  axis.ticks.y = element_line(colour = "black"),
  axis.ticks.x = element_line(colour = "black"),
  axis.text.y = element_text(size = 6, face = "plain", colour = "black"),
  axis.text.x = element_text(size = 6, face = "plain", colour = "black"),
  axis.title.x = element_text(size = 6, face = "plain", colour = "black"),
  axis.title.y = element_text(size = 6, face = "plain", colour = "black"),
  legend.key = element_blank(),
  legend.text = element_text(colour = "black"),
  legend.position = 'right',
  legend.box = "vertical",
  legend.title = element_blank(),
  legend.background = element_blank(),
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank()
)

# Theme as function (for custom base_size)
theme_moleculeace <- function(base_size = 6) {
  theme_bw(base_size = base_size) +
  theme(
    panel.border = element_rect(colour = "black", linewidth = 1, fill = NA),
    panel.background = element_blank(),
    plot.title = element_text(hjust = 0.5, face = "plain"),
    axis.ticks.y = element_line(colour = "black"),
    axis.ticks.x = element_line(colour = "black"),
    axis.text.y = element_text(face = "plain", colour = "black"),
    axis.text.x = element_text(face = "plain", colour = "black"),
    axis.title.x = element_text(face = "plain", colour = "black"),
    axis.title.y = element_text(face = "plain", colour = "black"),
    legend.key = element_blank(),
    legend.text = element_text(colour = "black"),
    legend.position = 'right',
    legend.box = "vertical",
    legend.title = element_blank(),
    legend.background = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )
}

# =============================================================================
# COLOR PALETTE
# =============================================================================

# Descriptor colors
descr_cols <- data.frame(
  descr = c('descriptors', 'ecfp1024', 'ecfp2048', 'maccs'),
  cols = c('orange2', 'lightseagreen', 'salmon2', 'olivedrab4')
)

cat("Setup complete: packages, theme, and colors loaded\n\n")

