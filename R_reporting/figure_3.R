################################################################################
# Figure 3: Predictive Model Performance
# Boxplot comparison across algorithms and molecular representations
################################################################################

cat("\n", strrep("=", 80), "\n")
cat("Figure 3: Predictive Model Performance\n")
cat(strrep("=", 80), "\n\n")

# Setup - force reload to get updated colors
rm(list = c("descr_cols", "default_theme"), envir = .GlobalEnv)
source("R_reporting/00_setup.R")

# Load data
backbones <- c("descriptors", "ecfp1024", "ecfp2048", "maccs")

all_runs <- list()
for (bb in backbones) {
  path <- file.path("results/predictive_model", bb, "model_metrics_all_runs.csv")
  if (file.exists(path)) {
    df <- read_csv(path, show_col_types = FALSE)
    df$descriptor <- bb
    all_runs[[bb]] <- df
  }
}

df <- bind_rows(all_runs) %>%
  rename(algorithm = model, run = `repeat`)

# Prepare data
df$descriptor <- factor(df$descriptor, levels = backbones)

algo_order <- df %>%
  group_by(algorithm) %>%
  summarise(mean_cliff = mean(Cliff_RMSE, na.rm = TRUE)) %>%
  arrange(mean_cliff) %>%
  pull(algorithm)

df$algorithm <- factor(df$algorithm, levels = algo_order)

# Get colors from descr_cols
colours <- as.character(descr_cols$cols[match(levels(df$descriptor), descr_cols$descr)])
names(colours) <- levels(df$descriptor)

# Create figure
final_fig <- ggplot(df, aes(x = algorithm, y = Cliff_RMSE)) +
  geom_boxplot(
    aes(fill = descriptor, color = descriptor),
    alpha = 0.3, width = 0.7, position = position_dodge(0.8),
    outlier.shape = NA, linewidth = 0.5
  ) +
  geom_point(
    aes(fill = descriptor, color = descriptor),
    position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.8),
    alpha = 0.6, size = 2, shape = 21, stroke = 0.5
  ) +
  scale_color_manual(values = colours, name = 'Molecular Representation') +
  scale_fill_manual(values = colours, name = 'Molecular Representation') +
  labs(x = NULL, y = bquote(bold("RMSE"[cliff]))) +
  default_theme +
  theme(
    legend.position = 'right',
    legend.title = element_text(size = 11, face = "bold"),
    legend.text = element_text(size = 10),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    axis.text.y = element_text(size = 10),
    axis.title.y = element_text(size = 12, face = "bold")
  )

# Save
dir.create("R_reporting/figures", showWarnings = FALSE, recursive = TRUE)
ggsave("R_reporting/figures/Figure_3.pdf", final_fig, width = 9, height = 4.5, dpi = 500, device = cairo_pdf)
ggsave("R_reporting/figures/Figure_3.png", final_fig, width = 9, height = 4.5, dpi = 500, bg = "white")

cat("\nFigure 3 saved (500 DPI)\n\n")

