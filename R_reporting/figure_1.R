################################################################################
# Figure 1: Reverse QSAR Analysis
# Three-panel visualization of fragment-based activity prediction
################################################################################

cat("\n", strrep("=", 80), "\n")
cat("Figure 1: Reverse QSAR\n")
cat(strrep("=", 80), "\n\n")

# Setup
if (!exists("default_theme")) source("R_reporting/00_setup.R")
suppressPackageStartupMessages(library(cowplot))

# Suppress ggplot2 warnings about NA/non-finite values (normal for classification metrics)
options(warn = -1)

# Load data
metrics_numeric <- read_csv("results/reverse_QSAR/model_metrics_numeric.csv", show_col_types = FALSE)
runs_df <- read_csv("results/reverse_QSAR/model_metrics_all_runs.csv", show_col_types = FALSE)

best_model <- metrics_numeric %>% arrange(desc(AUPRC_active)) %>% slice(1) %>% pull(model)
fragments_df <- read_csv(file.path("results/reverse_QSAR", best_model, "all_fragments_dictionary.csv"), show_col_types = FALSE)

cat(sprintf("Best model: %s\n\n", best_model))

# Panel A: Metrics comparison
metrics_to_plot <- c("AUPRC_active", "F1", "Precision", "MCC")
plot_data <- runs_df %>%
  select(model, all_of(metrics_to_plot)) %>%
  pivot_longer(cols = all_of(metrics_to_plot), names_to = "metric", values_to = "value") %>%
  mutate(metric = factor(metric, levels = metrics_to_plot))

n_models <- length(unique(plot_data$model))
model_colors <- colorRampPalette(c("#FBB4AE", "#B3CDE3", "#CCEBC5", "#DECBE4", "#FED9A6"))(n_models)

panel_a <- ggplot(plot_data, aes(x = model, y = value, fill = model)) +
  geom_boxplot(alpha = 0.7, width = 0.7, outlier.shape = NA, linewidth = 0.6) +
  geom_jitter(width = 0.2, alpha = 0.5, size = 2.5, shape = 21, stroke = 0.6) +
  facet_wrap(~metric, scales = "free_y", ncol = 2) +
  scale_fill_manual(values = model_colors) +
  labs(x = NULL, y = "Value") +
  default_theme +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 16, face = "bold"),
    axis.text.y = element_text(size = 16),
    axis.title.y = element_text(size = 18, face = "bold"),
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 17),
    legend.position = "none"
  )

# Panel B: Confusion matrix
pred_file <- "results/reverse_QSAR/predictions_run0.csv"
preds <- read_csv(pred_file, show_col_types = FALSE)

cm_data <- preds %>%
  count(y_true, y_pred) %>%
  mutate(
    True = factor(y_true, levels = c(0, 1), labels = c("Inactive", "Active")),
    Predicted = factor(y_pred, levels = c(0, 1), labels = c("Inactive", "Active"))
  )

cm_colors <- colorRampPalette(c("#F7FCF5", "#E5F5E0", "#C7E9C0", "#A1D99B", "#74C476", "#41AB5D"))(100)

panel_b <- ggplot(cm_data, aes(x = Predicted, y = True, fill = n)) +
  geom_tile(color = "white", linewidth = 2) +
  geom_text(aes(label = n), size = 10, fontface = "bold", color = "#2B2B2B") +
  scale_fill_gradientn(colors = cm_colors) +
  labs(x = "Predicted", y = "True", fill = "Count") +
  default_theme +
  theme(
    axis.text.x = element_text(size = 16, face = "bold"),
    axis.text.y = element_text(size = 16, face = "bold"),
    axis.title.x = element_text(size = 18, face = "bold"),
    axis.title.y = element_text(size = 18, face = "bold"),
    legend.position = "right",
    legend.title = element_text(size = 16, face = "bold"),
    legend.text = element_text(size = 14),
    legend.key.height = unit(0.7, "cm"),
    legend.key.width = unit(0.35, "cm"),
    plot.margin = margin(5, 30, 5, 5)
  ) +
  coord_fixed()

# Panel C: Fragment importance vs rank
frag_plot_data <- fragments_df %>%
  arrange(desc(importance)) %>%
  mutate(
    rank = row_number(),
    status = case_when(
      was_removed_by_AC == TRUE ~ "CAFE Removed",           # Red: removed by negative AC
      selection_method == "AC_added" ~ "CAFE Added",        # Green: added by positive AC
      TRUE ~ "Other"                                        # Grey: importance-based or not selected
    ),
    status = factor(status, levels = c("CAFE Added", "Other", "CAFE Removed"))
  )

# Cutoff line at total selected fragments (importance + AC-added)
n_selected <- sum(fragments_df$is_selected == TRUE, na.rm = TRUE)
cutoff_rank <- n_selected + 0.5

status_colors <- c(
  "CAFE Added" = "lightskyblue2",      # Green for CAFE-added
  "Other" = "grey91",           # Light blue for other
  "CAFE Removed" = "#FB8072"     # Red for CAFE-removed
)

panel_c <- ggplot(frag_plot_data, aes(x = rank, y = importance, color = status, fill = status)) +
  geom_vline(xintercept = cutoff_rank, linetype = "dashed", color = "#999999", linewidth = 0.8, alpha = 0.6) +
  geom_point(alpha = 0.7, size = 5, shape = 21, stroke = 0.7) +
  scale_color_manual(values = status_colors, name = "Fragment Status") +
  scale_fill_manual(values = status_colors, name = "Fragment Status") +
  labs(x = "Fragment Rank", y = "Importance Score") +
  default_theme +
  theme(
    axis.text.x = element_text(size = 17),
    axis.text.y = element_text(size = 17),
    axis.title.x = element_text(size = 19, face = "bold"),
    axis.title.y = element_text(size = 19, face = "bold"),
    legend.position = c(0.83, 0.78),
    legend.background = element_rect(fill = "white", color = "black", linewidth = 0.6),
    legend.title = element_text(size = 18, face = "bold"),
    legend.text = element_text(size = 16),
    legend.key.size = unit(1, "cm")
  )

# Combine panels
top_row <- plot_grid(panel_a, panel_b, ncol = 2, labels = c("A", "B"), 
                     label_size = 22, label_fontface = "bold", 
                     rel_widths = c(1.4, 1), label_x = c(0, 0.08))

final_fig <- plot_grid(top_row, panel_c, ncol = 1, labels = c("", "C"),
                       label_size = 22, label_fontface = "bold", rel_heights = c(1, 0.85))

# Save
dir.create("R_reporting/figures", showWarnings = FALSE, recursive = TRUE)
ggsave("R_reporting/figures/Figure_1.pdf", final_fig, width = 14, height = 11, dpi = 500, device = cairo_pdf)
ggsave("R_reporting/figures/Figure_1.png", final_fig, width = 14, height = 11, dpi = 500, bg = "white")

cat("\nFigure 1 saved (500 DPI)\n\n")

