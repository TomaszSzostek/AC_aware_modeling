################################################################################
# Figure 4: Split-Aware Predictive Performance
# All models, all backbones, all 10 repeats
################################################################################

cat("\n", strrep("=", 80), "\n")
cat("Figure 4: Split-Aware Predictive Performance\n")
cat(strrep("=", 80), "\n\n")

# Setup - force reload to get updated colors
rm(list = c("descr_cols", "default_theme"), envir = .GlobalEnv)
source("R_reporting/00_setup.R")
suppressPackageStartupMessages(library(cowplot))

# Load data (all repeats)
rev_runs <- read_csv("results/reverse_QSAR/model_metrics_all_runs.csv", show_col_types = FALSE)

predictive_files <- c(
  "results/predictive_model/descriptors/model_metrics_all_runs.csv",
  "results/predictive_model/maccs/model_metrics_all_runs.csv",
  "results/predictive_model/ecfp1024/model_metrics_all_runs.csv",
  "results/predictive_model/ecfp2048/model_metrics_all_runs.csv"
)

pred_runs <- predictive_files %>%
  map_dfr(~ read_csv(.x, show_col_types = FALSE))

# Consistent factor order
model_levels <- sort(unique(c(rev_runs$model, pred_runs$model)))
backbone_levels <- descr_cols$descr

rev_runs <- rev_runs %>%
  mutate(model = factor(model, levels = model_levels))

pred_runs <- pred_runs %>%
  mutate(
    model = factor(model, levels = model_levels),
    backbone = factor(backbone, levels = backbone_levels)
  )

# Pastel, muted backbone colors (molecular representation)
backbone_colors <- setNames(
  c("#F6C38B", "#A5DDE3", "#F4B5A6", "#B7D7A8"),
  backbone_levels
)

legend_title <- "Molecular descriptor"
rqsar_label <- "r-QSAR RMSE (AC-free)"
rqsar_mean_label <- "Mean RMSE (r-QSAR)"

# Prepare data for Panel A
pred_df <- pred_runs %>%
  rename(algorithm = model, descriptor = backbone, run = `repeat`)

rmse_all_runs <- bind_rows(
  pred_df %>% select(algorithm, descriptor, RMSE, run),
  rev_runs %>%
    rename(algorithm = model, run = `repeat`) %>%
    mutate(descriptor = "r-QSAR RMSE (AC-free)") %>%
    select(algorithm, descriptor, RMSE, run)
)

# Order by global RMSE (worst -> best) for Panel A readability
algo_order <- rmse_all_runs %>%
  group_by(algorithm) %>%
  summarise(mean_rmse = mean(RMSE, na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(mean_rmse)) %>%
  pull(algorithm)

pred_df <- pred_df %>%
  mutate(
    algorithm = factor(algorithm, levels = algo_order),
    descriptor = factor(descriptor, levels = backbone_levels)
  )

cliff_df <- pred_df %>%
  filter(!is.na(Cliff_RMSE)) %>%
  filter(Cliff_RMSE > 0)

repr_levels <- c(backbone_levels, rqsar_label)
repr_colors <- c(backbone_colors, setNames("#D81B60", rqsar_label))
legend_breaks <- c(backbone_levels, rqsar_label)
legend_shapes <- c(rep(21, length(backbone_levels)), 23)
legend_levels_full <- c(backbone_levels, rqsar_mean_label)
legend_colors_full <- c(backbone_colors, setNames("#D81B60", rqsar_mean_label))
legend_shapes_full <- c(rep(21, length(backbone_levels)), 21)

cliff_df <- cliff_df %>%
  mutate(repr = factor(descriptor, levels = repr_levels))

rmse_all_runs <- rmse_all_runs %>%
  mutate(
    algorithm = factor(algorithm, levels = algo_order),
    descriptor = factor(descriptor, levels = repr_levels),
    repr = factor(descriptor, levels = repr_levels)
  )

n_repr <- length(repr_levels)
dodge_step <- 0.22
algo_spacing <- 1.35
add_x_pos <- function(df) {
  df %>%
    mutate(
      algo_index = as.numeric(algorithm) * algo_spacing,
      repr_index = as.numeric(repr),
      x_pos = algo_index + (repr_index - (n_repr + 1) / 2) * dodge_step
    )
}

cliff_df <- add_x_pos(cliff_df)
rmse_all_runs <- add_x_pos(rmse_all_runs)

rqsar_rmse <- rev_runs %>%
  rename(algorithm = model) %>%
  group_by(algorithm) %>%
  summarise(rqsar_rmse = mean(RMSE, na.rm = TRUE), .groups = "drop")

# Panel B: prediction-QSAR split (AC-aware test set)
panel_b_data <- pred_df %>%
  filter(!is.na(RMSE)) %>%
  filter(!is.na(Cliff_RMSE)) %>%
  filter(Cliff_RMSE > 0)

pad_limits <- function(x, pad = 0.04) {
  rng <- range(x, na.rm = TRUE)
  delta <- diff(rng)
  if (!is.finite(delta) || delta == 0) delta <- abs(rng[1]) * 0.05 + 0.01
  c(rng[1] - delta * pad, rng[2] + delta * pad)
}

b_x_limits <- pad_limits(panel_b_data$RMSE, pad = 0.03)
b_y_limits <- pad_limits(panel_b_data$Cliff_RMSE, pad = 0.03)

panel_b_summary <- panel_b_data %>%
  group_by(algorithm, descriptor) %>%
  summarise(
    mean_RMSE = mean(RMSE, na.rm = TRUE),
    mean_Cliff_RMSE = mean(Cliff_RMSE, na.rm = TRUE),
    .groups = "drop"
  )

panel_theme <- theme(
  plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
  axis.text.x = element_text(angle = 45, hjust = 1, size = 12, face = "plain"),
  axis.text.y = element_text(size = 12),
  axis.title.x = element_text(size = 14, face = "bold"),
  axis.title.y = element_text(size = 14, face = "bold"),
  strip.background = element_blank(),
  strip.text = element_text(face = "bold", size = 13),
  legend.title = element_text(size = 15, face = "bold", hjust = 0.5),
  legend.text = element_text(size = 13, face = "bold"),
  legend.key.size = unit(0.85, "cm"),
  legend.background = element_rect(fill = "white", color = "black", linewidth = 0.6),
  legend.title.align = 0.5,
  panel.spacing = unit(0.2, "lines")
)

# Panel A plot (RMSEcliff + global RMSE + r-QSAR RMSE)
dodge_width <- 0.8
panel_a <- ggplot() +
  geom_boxplot(
    data = cliff_df,
    aes(x = x_pos, y = Cliff_RMSE, fill = repr, color = repr,
        group = interaction(algorithm, repr)),
    alpha = 0.3, width = 0.16, outlier.shape = NA, linewidth = 0.6
  ) +
  geom_point(
    data = cliff_df,
    aes(x = x_pos, y = Cliff_RMSE, fill = repr, color = repr),
    position = position_jitter(width = 0.05),
    alpha = 0.45, size = 2.0, shape = 21, stroke = 0.5
  ) +
  scale_color_manual(values = repr_colors, name = legend_title) +
  scale_fill_manual(values = repr_colors, name = legend_title) +
  scale_x_continuous(
    breaks = seq_along(algo_order) * algo_spacing,
    labels = algo_order,
    expand = expansion(mult = c(0.02, 0.02))
  ) +
  labs(x = NULL, y = bquote(bold("RMSE"[cliff]) ~ "/ RMSE")) +
  default_theme +
  panel_theme +
  theme(
    plot.title = element_blank(),
    legend.position = "none"
  )

# Panel B plots (RMSEcliff vs RMSE, per engine)
make_engine_plot <- function(engine, show_x_label = FALSE, show_y_label = FALSE, show_legend = FALSE) {
  df_engine <- panel_b_data %>%
    filter(algorithm == engine) %>%
    droplevels()

  rqsar_val <- rqsar_rmse %>%
    filter(algorithm == engine) %>%
    pull(rqsar_rmse)

  x_limits <- b_x_limits
  y_limits <- b_y_limits

  ggplot(df_engine, aes(x = RMSE, y = Cliff_RMSE)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                color = "#8C8C8C", linewidth = 0.7, alpha = 0.7) +
    geom_point(
      data = panel_b_data,
      aes(x = RMSE, y = Cliff_RMSE),
      color = "grey70", fill = "grey80", alpha = 0.35, size = 2.6, shape = 21, stroke = 0.3
    ) +
    { if (length(rqsar_val) == 1 && is.finite(rqsar_val))
        geom_point(
          data = data.frame(x = rqsar_val, y = rqsar_val, descriptor = rqsar_mean_label),
          aes(x = x, y = y, fill = descriptor, color = descriptor),
          size = 3.6, shape = 21, stroke = 0.6, alpha = 0.9,
          show.legend = TRUE
        )
      else NULL
    } +
    geom_point(
      aes(color = descriptor, fill = descriptor),
      alpha = 0.85, size = 3.6, shape = 21, stroke = 0.6
    ) +
    scale_color_manual(values = legend_colors_full, breaks = legend_levels_full, name = legend_title) +
    scale_fill_manual(values = legend_colors_full, breaks = legend_levels_full, name = legend_title) +
    coord_cartesian(xlim = x_limits, ylim = y_limits) +
    labs(
      x = if (show_x_label) "RMSE" else NULL,
      y = if (show_y_label) "RMSEcliff" else NULL,
      title = engine
    ) +
    default_theme +
    panel_theme +
    theme(
      axis.text.x = element_text(angle = 0, hjust = 0.5, size = 13, face = "plain"),
      axis.text.y = element_text(size = 13),
      axis.title = element_text(size = 16, face = "bold"),
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      legend.position = if (show_legend) "bottom" else "none",
      legend.box = "horizontal",
      legend.direction = "horizontal"
    ) +
    guides(
      color = "none",
      fill = guide_legend(
        title.position = "top",
        title.hjust = 0.5,
        nrow = 1,
        byrow = TRUE,
        override.aes = list(shape = 21, size = 4.8, stroke = 0, alpha = 1)
      )
    )
}

ncol_b <- 3
nrow_b <- ceiling(length(algo_order) / ncol_b)
legend_plot <- ggplot(
  tibble(legend_key = factor(legend_levels_full, levels = legend_levels_full)),
  aes(x = 1, y = legend_key, fill = legend_key, color = legend_key, shape = legend_key)
) +
  geom_point(size = 4.8, stroke = 0.7) +
  scale_color_manual(values = legend_colors_full, breaks = legend_levels_full, name = legend_title) +
  scale_fill_manual(values = legend_colors_full, breaks = legend_levels_full, name = legend_title) +
  scale_shape_manual(values = legend_shapes_full, breaks = legend_levels_full, name = legend_title) +
  theme_void() +
  theme(
    legend.position = "bottom",
    legend.box = "horizontal",
    legend.direction = "horizontal",
    legend.title = element_text(size = 15, face = "bold", hjust = 0.5),
    legend.text = element_text(size = 13, face = "bold"),
    legend.key.size = unit(0.85, "cm"),
    legend.background = element_rect(fill = "white", color = "black", linewidth = 0.95),
    legend.spacing.x = unit(0.25, "cm"),
    legend.spacing.y = unit(0.2, "cm"),
    legend.box.margin = margin(4, 8, 4, 8),
    legend.margin = margin(4, 6, 4, 6)
  ) +
  guides(
    color = "none",
    shape = "none",
    fill = guide_legend(
      title.position = "top",
      title.hjust = 0.5,
      nrow = 1,
      byrow = TRUE,
      override.aes = list(shape = legend_shapes_full, size = 4.8, stroke = 0, alpha = 1)
    )
  )

legend_b <- get_legend(legend_plot)
panel_b_list <- lapply(seq_along(algo_order), function(i) {
  engine <- algo_order[i]
  col_idx <- (i - 1) %% ncol_b + 1
  row_idx <- (i - 1) %/% ncol_b + 1
  show_y <- col_idx == 1
  show_x <- row_idx == nrow_b
  make_engine_plot(engine, show_x_label = show_x, show_y_label = show_y, show_legend = FALSE)
})
panel_b <- plot_grid(plotlist = panel_b_list, ncol = ncol_b)

panel_b_full <- plot_grid(panel_b, legend_b, ncol = 1, rel_heights = c(1, 0.12))

# Combine panels (Panel A on top, B bottom)
final_fig <- plot_grid(panel_a, panel_b_full, ncol = 1, labels = c("A", "B"),
                       label_size = 18, label_fontface = "bold", rel_heights = c(0.55, 1.45))

# Save
dir.create("R_reporting/figures", showWarnings = FALSE, recursive = TRUE)
ggsave("R_reporting/figures/Figure_4.pdf", final_fig, width = 14, height = 10.5, dpi = 500, device = cairo_pdf)
ggsave("R_reporting/figures/Figure_4.png", final_fig, width = 14, height = 10.5, dpi = 500, bg = "white")

cat("\nFigure 4 saved (500 DPI)\n\n")
