################################################################################
# Figure 4: CAFE LATE Scoring Analysis
################################################################################

cat("\n", strrep("=", 80), "\n")
cat("Figure 4: CAFE LATE Analysis\n")
cat(strrep("=", 80), "\n\n")

# Setup - force reload to get updated colors
rm(list = c("descr_cols", "default_theme"), envir = .GlobalEnv)
source("R_reporting/00_setup.R")
suppressPackageStartupMessages(library(cowplot))
suppressPackageStartupMessages(library(ggrepel))
suppressPackageStartupMessages(library(scales))

# Load generation results (prefer full post-score table to capture rescues)
preferred_path <- "results/generation/post_score.csv"
fallback_path  <- "results/generation/hits.csv"

hits_path <- if (file.exists(preferred_path)) preferred_path else fallback_path

if (!file.exists(hits_path)) {
  stop("hits.csv not found at ", hits_path)
}

df <- read_csv(hits_path, show_col_types = FALSE)

# Filter invalid rows and compute pre-CAFE QSAR if missing
df <- df %>%
  filter(!is.na(cafe_boost), !is.na(cafe_enrichment), !is.na(qsar)) %>%
  mutate(
    qsar_before_cafe = if ("qsar_before_cafe" %in% colnames(.)) qsar_before_cafe else qsar - cafe_boost
  )

# Validate CAFE LATE columns
if (!("cafe_enrichment" %in% colnames(df)) || !("cafe_boost" %in% colnames(df))) {
  stop("CAFE LATE columns not found in hits.csv")
}

# Filter to molecules with CAFE boost
cafe_molecules <- df %>%
  filter(cafe_boost > 0)

if (nrow(cafe_molecules) == 0) {
  stop("No molecules with CAFE LATE boost found")
}

cat(sprintf("Loaded %d molecules from %s, %d with CAFE LATE boost\n\n", 
            nrow(df), hits_path, nrow(cafe_molecules)))

# =============================================================================
# PANEL A: Overlapping Density Curves (Publication Enhanced)
# =============================================================================

# Categorize by enrichment strength
cafe_molecules <- cafe_molecules %>%
  mutate(
    enrichment_category = cut(
      cafe_enrichment,
      breaks = c(0, 10, 20, 40, Inf),
      labels = c("Weak (0-10)", "Medium (10-20)", "Strong (20-40)", "Very Strong (>40)"),
      include.lowest = TRUE
    )
  )

# Elegant pastel colors - distinct but harmonious
density_colors <- c("bisque2", "#A8DCD1", "plum2", "mediumpurple")

panel_a <- ggplot(cafe_molecules, aes(x = cafe_boost, color = enrichment_category, fill = enrichment_category)) +
  geom_density(alpha = 0.15, linewidth = 1.8) +
  geom_rug(alpha = 0.5, linewidth = 1.5, length = unit(0.04, "npc")) +
  scale_fill_manual(values = density_colors, name = "AC Enrichment Level") +
  scale_color_manual(values = density_colors, name = "AC Enrichment Level") +
  labs(x = "CAFE LATE Boost", y = "Density") +
  default_theme +
  theme(
    legend.position = c(0.78, 0.8),
    legend.background = element_rect(fill = "white", color = "black", linewidth = 1.2),
    legend.title = element_text(size = 26, face = "bold"),
    legend.text = element_text(size = 22, face = "bold"),
    legend.key.size = unit(1.2, "cm"),
    legend.spacing.y = unit(0.3, "cm"),
    axis.text.x = element_text(size = 22),
    axis.text.y = element_text(size = 22),
    axis.title.x = element_text(size = 26, face = "bold"),
    axis.title.y = element_text(size = 26, face = "bold"),
    plot.margin = margin(15, 20, 15, 15)
  ) +
  guides(
    fill = guide_legend(override.aes = list(alpha = 0.6, color = NA, linewidth = 0)),
    color = "none"
  )

# =============================================================================
# PANEL B: Top AC-Added Fragments (REDESIGNED - Publication Perfect)
# =============================================================================

# Load fragment metadata
frag_meta_path <- "R_reporting/figures/fragments/fragment_metadata.csv"
if (!file.exists(frag_meta_path)) {
  stop("Fragment metadata not found. Run prepare_fragment_images.py first.")
}

max_fragments_display <- 18
frag_data <- read_csv(frag_meta_path, show_col_types = FALSE) %>%
  arrange(desc(ac_enrichment)) %>%
  mutate(
    rank = row_number(),
    fragment_label = paste0("F", rank)
  )

cat(sprintf("Loaded %d top AC-added fragments\n", nrow(frag_data)))

# NEW COLOR PALETTE - Blue-Purple gradient (high contrast, beautiful)
color_low <- "#E0F3F8"    # Pale cyan
color_mid <- "#4575B4"    # Deep blue  
color_high <- "#762A83"   # Rich purple

# Load and prepare fragment images FIRST
suppressPackageStartupMessages(library(magick))
suppressPackageStartupMessages(library(grid))

frag_images <- list()
for (i in 1:nrow(frag_data)) {
  img_path <- file.path("R_reporting/figures/fragments", paste0(frag_data$fragment_id[i], ".png"))
  if (file.exists(img_path)) {
    img <- image_read(img_path)
    # Resize to inline display size (larger for better visibility)
    img <- image_resize(img, "250x250")
    img <- image_border(img, "white", "5x5")
    img <- image_border(img, "gray50", "1x1")
    frag_images[[i]] <- img
  }
}

panel_b_palette <- c("#f0f7ec", "#c8e6c9", "#81c784", "#4caf50", "#2e7d32")

panel_b <- ggplot(frag_data, aes(x = reorder(fragment_label, ac_enrichment), y = ac_enrichment)) +
  
  geom_segment(aes(x = fragment_label, xend = fragment_label, y = 0, yend = ac_enrichment),
               color = "gray80", linewidth = 6, alpha = 0.25, 
               position = position_nudge(x = 0.015)) +
  
  geom_segment(aes(x = fragment_label, xend = fragment_label, y = 0, yend = ac_enrichment,
                   color = ac_enrichment),
               linewidth = 4, alpha = 0.95) +
  
  geom_point(aes(size = n_cliff_pairs + 8, fill = ac_enrichment), 
             shape = 21, stroke = 0, color = NA, alpha = 0.15) +
  
  geom_point(aes(size = n_cliff_pairs + 5, fill = ac_enrichment), 
             shape = 21, stroke = 0, color = NA, alpha = 0.25) +
  
  geom_point(aes(size = n_cliff_pairs, fill = ac_enrichment), 
             shape = 21, stroke = 3, color = "white", alpha = 0.95) +
  
  geom_point(aes(size = n_cliff_pairs - 2, fill = ac_enrichment), 
             shape = 21, stroke = 0, color = NA, alpha = 1) +
  
  scale_fill_gradient2(low = color_low, mid = color_mid, high = color_high,
                       midpoint = median(frag_data$ac_enrichment),
                       name = "SALI\nScore",
                       guide = guide_colorbar(
                         barwidth = 2, barheight = 15,
                         title.position = "top",
                         frame.colour = "gray40",
                         ticks.colour = "gray40",
                         frame.linewidth = 1.5
                       )) +
  scale_color_gradient2(low = color_low, mid = color_mid, high = color_high,
                        midpoint = median(frag_data$ac_enrichment),
                        guide = "none") +
  
  scale_size_continuous(range = c(8, 22), guide = "none") +
  
  geom_text(aes(label = paste0("n=", n_cliff_pairs)), 
            vjust = -2.8, size = 10, fontface = "bold",
            color = "gray30", family = "sans") +
  
  labs(x = "", 
       y = "Enrichment Score") +
  
  default_theme +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.title.x = element_blank(),
    axis.text.y = element_text(size = 26, color = "gray20"),
    axis.title.y = element_text(size = 28, face = "bold", margin = margin(r = 15)),
    panel.background = element_rect(fill = "white", color = NA),
    panel.grid.major.y = element_line(color = "gray92", linewidth = 0.6, linetype = "solid"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = c(0.99, 0.70),
    legend.background = element_rect(fill = alpha("white", 0.9), color = "gray40", linewidth = 1.5),
    legend.title = element_text(size = 22, face = "bold", hjust = 0.5),
    legend.text = element_text(size = 20),
    plot.margin = margin(20, 35, 140, 50)
  ) +
  
  scale_y_continuous(expand = expansion(mult = c(0.02, 0.2)),
                    breaks = seq(0, 50, 10))

# Now add fragment structures INLINE at bottom of lollipops
if (length(frag_images) == nrow(frag_data)) {
  cat(sprintf("Adding %d fragment structures inline...\n", length(frag_images)))
  
  temp_frag_paths <- c()
  for (i in 1:length(frag_images)) {
    temp_path <- sprintf("R_reporting/figures/fragments/temp_frag_%d.png", i)
    image_write(frag_images[[i]], temp_path)
    temp_frag_paths <- c(temp_frag_paths, temp_path)
  }
  
  panel_b_final <- ggdraw(panel_b)
  
  x_positions <- seq(0.20, 0.87, length.out = nrow(frag_data))
  y_position <- 0.11
  img_width <- 0.17
  img_height <- 0.22
  
  for (i in 1:length(temp_frag_paths)) {
    panel_b_final <- panel_b_final +
      draw_image(temp_frag_paths[i], 
                 x = x_positions[i] - img_width/2, 
                 y = y_position - img_height/2,
                 width = img_width, 
                 height = img_height)
  }
  
  panel_b <- panel_b_final
  
  cat("Panel B: REDESIGNED with inline structures & improved colors! 🎨\n")
} else {
cat("Warning: could not load all fragment images, using lollipop only\n")
}

# =============================================================================
# PANEL C: Scatter Plot - Boosted vs Non-Boosted with Rescued Molecules
# =============================================================================

# Load REAL threshold from best QSAR model
threshold_file <- "results/predictive_model/best_overall.itxt"
if (file.exists(threshold_file)) {
  threshold_content <- readLines(threshold_file)
  threshold_lines <- grep("Threshold:", threshold_content, value = TRUE)
  if (length(threshold_lines) > 0) {
    # Extract number from line like "Threshold: 0.523"
    activity_threshold <- as.numeric(gsub(".*Threshold:\\s*([0-9.]+).*", "\\1", threshold_lines[1]))
  } else {
    activity_threshold <- 0.5
  }
} else {
  activity_threshold <- 0.5
}

cat(sprintf("Activity threshold from QSAR: %.3f\n", activity_threshold))

# Prepare data for Panel C - use full dataset
panel_c_data <- df %>%
  mutate(
    qsar_before_cafe = if ("qsar_before_cafe" %in% colnames(.)) qsar_before_cafe else qsar - cafe_boost,
    cafe_boost = as.numeric(cafe_boost),
    qsar = as.numeric(qsar),
    qsar_before_cafe = as.numeric(qsar_before_cafe),
    cafe_enrichment = as.numeric(cafe_enrichment)
  ) %>%
  filter(!is.na(qsar), !is.na(cafe_boost), !is.na(cafe_enrichment)) %>%
  mutate(
    rescued = (qsar_before_cafe < activity_threshold) & (qsar >= activity_threshold),
    is_active = qsar >= activity_threshold,
    molecule_class = case_when(
      rescued ~ "Rescued Molecules",
      cafe_boost > 0 ~ "AC Boosted",
      TRUE ~ "Non-Boosted"
    )
  )

n_rescued <- sum(panel_c_data$rescued, na.rm = TRUE)
n_boosted <- sum(panel_c_data$molecule_class == "AC Boosted", na.rm = TRUE)
n_nonboosted <- sum(panel_c_data$molecule_class == "Non-Boosted", na.rm = TRUE)

cat(sprintf("Panel C: Rescued=%d, AC Boosted=%d, Non-Boosted=%d\n\n", n_rescued, n_boosted, n_nonboosted))

# Sample for visualization - larger samples for better cloud effect
set.seed(42)

# Different sample sizes for different classes - REDUCED for better visibility
n_nonboosted_sample <- min(n_nonboosted, 8000)  # Reduced from 15000
n_boosted_sample <- min(n_boosted, 12000)  # Reduced from 20000
n_rescued_sample <- n_rescued  # Show all rescued molecules

panel_c_plot_list <- list()

# Non-Boosted
if (n_nonboosted > 0) {
  nonboosted_data <- panel_c_data %>% filter(molecule_class == "Non-Boosted")
  if (nrow(nonboosted_data) > n_nonboosted_sample) {
    panel_c_plot_list[["Non-Boosted"]] <- nonboosted_data %>% slice_sample(n = n_nonboosted_sample, replace = FALSE)
  } else {
    panel_c_plot_list[["Non-Boosted"]] <- nonboosted_data
  }
}

# AC Boosted
if (n_boosted > 0) {
  boosted_data <- panel_c_data %>% filter(molecule_class == "AC Boosted")
  if (nrow(boosted_data) > n_boosted_sample) {
    panel_c_plot_list[["AC Boosted"]] <- boosted_data %>% slice_sample(n = n_boosted_sample, replace = FALSE)
  } else {
    panel_c_plot_list[["AC Boosted"]] <- boosted_data
  }
}

# Rescued (show all)
if (n_rescued > 0) {
  panel_c_plot_list[["Rescued Molecules"]] <- panel_c_data %>% filter(molecule_class == "Rescued Molecules")
}

panel_c_plot <- bind_rows(panel_c_plot_list)

cat(sprintf("Panel C plotting: Non-Boosted=%d, AC Boosted=%d, Rescued=%d\n\n", 
            sum(panel_c_plot$molecule_class == "Non-Boosted", na.rm = TRUE),
            sum(panel_c_plot$molecule_class == "AC Boosted", na.rm = TRUE),
            sum(panel_c_plot$molecule_class == "Rescued Molecules", na.rm = TRUE)))

# Panel C: X = QSAR Score, Y = AC Enrichment
# Left side (inactive): red glow, Right side (active): green glow
# Squares: color gradient from light gray to intense green based on AC enrichment
# Rescued: pastel orange triangles
set.seed(42)

# Get max enrichment for Y axis scaling and color gradient
max_enrichment <- max(panel_c_plot$cafe_enrichment, na.rm = TRUE)
min_enrichment <- min(panel_c_plot$cafe_enrichment[panel_c_plot$molecule_class != "Rescued Molecules"], na.rm = TRUE)

# Separate squares (non-boosted + boosted) and rescued triangles
squares_data <- filter(panel_c_plot, molecule_class != "Rescued Molecules")
rescued_data <- filter(panel_c_plot, molecule_class == "Rescued Molecules")

# Create scatter plot: X = QSAR Score, Y = AC Enrichment
panel_c <- ggplot() +
  
  # Background glows - red left (inactive), green right (active) - LESS INTENSE
  annotate("rect", 
           xmin = -Inf, xmax = activity_threshold, 
           ymin = -Inf, ymax = Inf,
           fill = scales::alpha("#ffebee", 0.35), color = NA) +  # Red glow - inactive (left) - less intense
  
  annotate("rect", 
           xmin = activity_threshold, xmax = Inf, 
           ymin = -Inf, ymax = Inf,
           fill = scales::alpha("#e8f5e9", 0.35), color = NA) +  # Green glow - active (right) - less intense
  
  # Labels for Active/Inactive halves
  annotate("text", x = activity_threshold / 2, y = max_enrichment * 0.95, 
           label = "Inactive", size = 12, fontface = "bold", color = "#c62828") +
  annotate("text", x = (activity_threshold + 1) / 2, y = max_enrichment * 0.95, 
           label = "Active", size = 12, fontface = "bold", color = "#2e7d32") +
  
  # Vertical threshold line dividing active/inactive
  geom_vline(xintercept = activity_threshold, 
             linetype = "dashed", color = "gray60", linewidth = 1.2, alpha = 0.6) +
  
  # All squares (non-boosted + boosted) with color gradient based on AC enrichment
  geom_point(
    data = squares_data,
    aes(x = qsar, y = cafe_enrichment, color = cafe_enrichment),
    shape = 15,  # Square
    size = 4.5,  # Increased significantly
    alpha = 0.65,
    stroke = 0.3
  ) +
  
  # Rescued molecules (very light pastel blue triangles) - MUCH LARGER
  geom_point(
    data = rescued_data,
    aes(x = qsar, y = cafe_enrichment),
    color = "#b3e5fc",  # Very light pastel blue, much less intense
    shape = 17,  # Triangle
    size = 4.5,  # Increased significantly
    alpha = 0.4,
    stroke = 0.3
  ) +
  
  # Color gradient scale: from very light gray to intense green
  # Smaller vertical colorbar
  scale_color_gradient(
    low = "#e0e0e0",    # Very light gray, almost neutral
    high = "#2e7d32",   # Intense green
    name = "AC Enrichment",
    guide = guide_colorbar(
      title.position = "right",
      title.hjust = 0.5,
      title.vjust = 1,
      barwidth = 5.0,
      barheight = 45.0,
      label.theme = element_text(size = 20),
      title.theme = element_text(size = 28, face = "bold")
    )
  ) +
  
  # Labels
  labs(
    x = "QSAR Score",
    y = NULL  # No Y-axis label, replaced by colorbar
  ) +
  
  # Scales - shift plot slightly to the right
  scale_x_continuous(
    breaks = seq(0, 1, 0.2),
    expand = expansion(mult = c(0.05, 0.08))  # More space on right (for colorbar), more on left to shift right
  ) +
  scale_y_continuous(
    expand = expansion(mult = c(0.02, 0.05))
  ) +
  
  # Legend for Rescued - aligned with plot top edge
  annotate("rect", xmin = 0.85, xmax = 0.995, 
           ymin = max_enrichment * 0.88, ymax = max_enrichment,  # Aligned with top of plot
           fill = "white", color = "gray80", linewidth = 0.3, alpha = 0.7) +
  annotate("point", x = 0.87, y = max_enrichment * 0.94, color = "#b3e5fc", shape = 17, size = 10, alpha = 0.7) +
  annotate("text", x = 0.91, y = max_enrichment * 0.94, label = "Rescued", hjust = 0, vjust = 0.5, size = 11, color = "black", fontface = "bold") +
  
  # Clean theme - minimal, publication-ready
  theme_bw() +
  theme(
    legend.position = "right",  # Smaller vertical colorbar on right side
    legend.justification = "center",
    legend.background = element_blank(),  # Remove frame
    legend.title = element_text(size = 24, face = "bold", color = "black", angle = 90),
    legend.text = element_text(size = 20, color = "black"),
    axis.text.x = element_text(size = 22, color = "black"),
    axis.text.y = element_blank(),  # No Y-axis numeric labels
    axis.ticks.y = element_blank(),  # No Y-axis ticks
    axis.title.x = element_text(size = 26, face = "bold", color = "black"),
    axis.title.y = element_blank(),  # No Y-axis title
    panel.background = element_rect(fill = "white", color = NA),
    panel.grid.major = element_line(color = "gray90", linewidth = 0.5, linetype = "solid"),
    panel.grid.minor = element_line(color = "gray95", linewidth = 0.3, linetype = "dotted"),
    panel.border = element_rect(color = "black", linewidth = 1, fill = NA),
    plot.margin = margin(20, 90, 20, 50)
  )

# =============================================================================
# COMBINE PANELS (A and B on top, C full width bottom)
# =============================================================================

top_row <- plot_grid(
  panel_a, panel_b,
  ncol = 2,
  labels = c("A", "B"),
  label_size = 28,
  label_fontface = "bold",
  rel_widths = c(1, 1)
)

final_fig <- plot_grid(
  top_row, panel_c,
  ncol = 1,
  labels = c("", "C"),
  label_size = 28,
  label_fontface = "bold",
  rel_heights = c(1, 1.1)
)

# =============================================================================
# SAVE (PUBLICATION SIZE - LARGER)
# =============================================================================

dir.create("R_reporting/figures", showWarnings = FALSE, recursive = TRUE)
ggsave("R_reporting/figures/Figure_4.pdf", final_fig, width = 28, height = 20, dpi = 500, device = cairo_pdf)
ggsave("R_reporting/figures/Figure_4.png", final_fig, width = 28, height = 20, dpi = 500, bg = "white")

cat("\nFigure 4 saved (500 DPI, 28×20 inches)\n")
cat(sprintf("   Molecules with CAFE LATE: %d/%d (%.1f%%)\n", 
            nrow(cafe_molecules), nrow(df), nrow(cafe_molecules)/nrow(df)*100))
cat(sprintf("   Rescued molecules: %d\n\n", n_rescued))

