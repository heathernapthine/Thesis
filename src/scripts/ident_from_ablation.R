
#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(readr)
})

`%||%` <- function(x, y) if (!is.null(x)) x else y

# -----------------------------
# Args: Rscript ident_from_ablation_rds.R <results_dir> <out_dir>
# Defaults: results_dir='src/ablationresults', out_dir=results_dir
# -----------------------------
args <- commandArgs(trailingOnly = TRUE)
results_dir <- if (length(args) >= 1) args[1] else "src/ablationresults"
out_dir     <- if (length(args) >= 2) args[2] else results_dir
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# -----------------------------
# Helpers
# -----------------------------
parse_vscale_from_path <- function(path) {
  # 'train_vscale_0p25.rds' -> 0.25 ; 'train_vscale_1.rds' -> 1
  tg <- stringr::str_match(basename(path), "vscale_(.*)\\.rds$")[,2]
  if (is.na(tg)) return(NA_real_)
  as.numeric(gsub("p", ".", tg))
}

safe_get <- function(x, dotted_names) {
  # Try a list of candidate names like c("t_obs","inputs$t_obs","data$t_obs")
  for (nm in dotted_names) {
    parts <- strsplit(nm, "\\$")[[1]]
    val <- x
    ok <- TRUE
    for (p in parts) {
      if (!is.list(val) || is.null(val[[p]])) { ok <- FALSE; break }
      val <- val[[p]]
    }
    if (ok && !is.null(val)) return(val)
  }
  NULL
}

extract_train_fields <- function(obj) {
  pp <- safe_get(obj, c("posterior.parameters", "posterior", "pp"))
  gap <- safe_get(pp, c("gap_mu_star", "gap_mu", "gap_mean", "gap"))
  gap_var <- safe_get(pp, c("gap_sigma2_star", "gap_var", "gap_sigma2"))
  ed <- safe_get(pp, c("expected_d", "D_star", "E_d"))

  t_mat   <- safe_get(obj, c("t_obs", "t", "inputs$t_obs", "inputs$t", "data$t_obs", "data$t", "misc$t_obs"))
  d_mat   <- safe_get(obj, c("d", "inputs$d", "data$d"))
  rho_vec <- safe_get(obj, c("rho", "inputs$rho", "data$rho"))
  tau_vec <- safe_get(obj, c("tau", "inputs$tau", "data$tau"))
  onset_truth <- safe_get(obj, c("onset", "inputs$onset", "data$onset"))
  cond_list <- safe_get(obj, c("cond_list", "inputs$cond_list", "misc$cond_list"))

  list(pp=pp, gap=gap, gap_var=gap_var, ed=ed,
       t=t_mat, d=d_mat, rho=rho_vec, tau=tau_vec,
       onset_truth=onset_truth, cond_list=cond_list)
}

# -----------------------------
# Locate cached RDS files
# -----------------------------
train_files <- list.files(results_dir, pattern="^train_vscale_.*\\.rds$", full.names = TRUE)
test_files  <- list.files(results_dir, pattern="^test_full_vscale_.*\\.rds$", full.names = TRUE)

if (length(train_files) == 0) {
  stop("No 'train_vscale_*.rds' files found in: ", results_dir)
}

train_info <- tibble(file = train_files,
                     var_scale = sapply(train_files, parse_vscale_from_path)) %>%
              arrange(var_scale)

test_info <- tibble(file = test_files,
                    var_scale = sapply(test_files, parse_vscale_from_path)) %>%
             arrange(var_scale)

# -----------------------------
# Load training objects and extract per-(i,m) latents
# -----------------------------
loaded <- list()
for (i in seq_len(nrow(train_info))) {
  f <- train_info$file[i]
  vs <- train_info$var_scale[i]
  obj <- readRDS(f)
  ex  <- extract_train_fields(obj)

  if (is.null(ex$gap)) stop("Couldn't find a delay mean field (e.g., 'gap_mu_star') in: ", f)

  N <- nrow(ex$gap); M <- ncol(ex$gap)

  # Mask: where diagnosis time is observed & positive
  if (!is.null(ex$d) && !is.null(ex$t) && nrow(ex$t) == N && ncol(ex$t) == M) {
    mask <- (ex$d == 1) & !is.na(ex$t)
    onset_implied <- matrix(NA_real_, N, M)
    onset_implied[mask] <- ex$t[mask] - ex$gap[mask]
  } else {
    # If inputs weren't saved in the RDS, we can still compute drift on delays;
    # onset_implied will be unavailable.
    mask <- matrix(TRUE, nrow = N, ncol = M)
    onset_implied <- NULL
  }

  loaded[[as.character(vs)]] <- list(
    gap = ex$gap,
    gap_var = ex$gap_var,
    ed = ex$ed,
    t = ex$t,
    d = ex$d,
    rho = ex$rho,
    tau = ex$tau,
    onset_implied = onset_implied,
    onset_truth = ex$onset_truth,
    mask = mask,
    N = N, M = M,
    cond_list = ex$cond_list %||% paste0("cond_", seq_len(M))
  )
}

# Choose baseline var_scale for drift comparisons
baseline_vs <- if (1 %in% train_info$var_scale) 1 else min(train_info$var_scale)
base <- loaded[[as.character(baseline_vs)]]

M <- base$M
cond_names <- base$cond_list

# -----------------------------
# Compute overall drift and compensation per var_scale
# -----------------------------
overall_rows <- list()
bycond_rows  <- list()

for (vs in train_info$var_scale) {
  cur <- loaded[[as.character(vs)]]
  if (cur$N != base$N || cur$M != base$M) stop("Dimension mismatch vs baseline for var_scale=", vs)

  # Restrict to cells evaluable under baseline mask
  mask <- base$mask
  have_onset <- !is.null(base$onset_implied) && !is.null(cur$onset_implied)

  # Drift relative to baseline
  if (have_onset) {
    d_onset <- cur$onset_implied[mask] - base$onset_implied[mask]
    onset_drift_mae <- mean(abs(d_onset))
  } else {
    d_onset <- NA_real_
    onset_drift_mae <- NA_real_
  }
  d_delay <- cur$gap[mask] - base$gap[mask]
  delay_drift_mae <- mean(abs(d_delay))

  comp_corr <- if (have_onset) suppressWarnings(cor(d_onset, d_delay, use = "complete.obs")) else NA_real_

  # Presence-weighted drift (using baseline E[d] if available)
  if (!is.null(base$ed)) {
    w <- base$ed[mask]; w[!is.finite(w)] <- 0
    w <- w / (sum(w) + 1e-8)
    onset_drift_mae_w <- if (have_onset) sum(w * abs(cur$onset_implied[mask] - base$onset_implied[mask])) else NA_real_
    delay_drift_mae_w <- sum(w * abs(cur$gap[mask] - base$gap[mask]))
  } else {
    onset_drift_mae_w <- NA_real_
    delay_drift_mae_w <- NA_real_
  }

  # Truth-based MAEs if onset truth is available inside the training RDS
  if (have_onset && !is.null(base$onset_truth) &&
      nrow(base$onset_truth) == base$N && ncol(base$onset_truth) == base$M) {
    truth_mask <- mask & !is.na(base$onset_truth)
    onset_mae_vs_truth <- mean(abs(cur$onset_implied[truth_mask] - base$onset_truth[truth_mask]))
    delay_true <- if (!is.null(base$t)) base$t - base$onset_truth else NULL
    delay_mae_vs_truth <- if (!is.null(delay_true)) mean(abs(cur$gap[truth_mask] - delay_true[truth_mask])) else NA_real_
  } else {
    onset_mae_vs_truth <- NA_real_
    delay_mae_vs_truth <- NA_real_
  }

  overall_rows[[length(overall_rows) + 1]] <- tibble(
    var_scale = vs,
    baseline = baseline_vs,
    N = base$N, M = base$M,
    n_eval = sum(mask, na.rm = TRUE),
    onset_drift_mae = onset_drift_mae,
    delay_drift_mae = delay_drift_mae,
    onset_drift_mae_w = onset_drift_mae_w,
    delay_drift_mae_w = delay_drift_mae_w,
    compensation_corr = comp_corr,
    onset_mae_vs_truth = onset_mae_vs_truth,
    delay_mae_vs_truth = delay_mae_vs_truth
  )

  # Per-condition drift & compensation
  for (m in seq_len(M)) {
    mask_m <- mask[, m]
    n_eval_m <- sum(mask_m, na.rm = TRUE)
    if (n_eval_m == 0) next

    onset_drift_m <- if (have_onset) mean(abs(cur$onset_implied[mask_m, m] - base$onset_implied[mask_m, m])) else NA_real_
    delay_drift_m <- mean(abs(cur$gap[mask_m, m] - base$gap[mask_m, m]))
    comp_corr_m <- if (have_onset) suppressWarnings(cor(
      (cur$onset_implied[mask_m, m] - base$onset_implied[mask_m, m]),
      (cur$gap[mask_m, m] - base$gap[mask_m, m]),
      use = "complete.obs"
    )) else NA_real_

    bycond_rows[[length(bycond_rows) + 1]] <- tibble(
      var_scale = vs,
      baseline = baseline_vs,
      condition_index = m,
      condition = if (length(cond_names) >= m) cond_names[m] else paste0("cond_", m),
      n_eval = n_eval_m,
      onset_drift_mae = onset_drift_m,
      delay_drift_mae = delay_drift_m,
      compensation_corr = comp_corr_m
    )
  }
}

overall <- bind_rows(overall_rows) %>% arrange(var_scale)
by_cond <- bind_rows(bycond_rows)   %>% arrange(var_scale, condition_index)

# -----------------------------
# Bring in the 'top' & 'shrink' frames from the test_full caches (if present)
# so you can compare predictive metrics with latent drift.
# -----------------------------
top_list <- list()
shrink_list <- list()
if (nrow(test_info) > 0) {
  for (i in seq_len(nrow(test_info))) {
    obj <- readRDS(test_info$file[i])
    vs <- test_info$var_scale[i]
    if (!is.null(obj$top)) {
      tt <- obj$top
      if (is.null(tt$var_scale)) tt$var_scale <- vs
      top_list[[length(top_list) + 1]] <- tt
    }
    if (!is.null(obj$shrink)) {
      ss <- obj$shrink
      if (is.null(ss$var_scale)) ss$var_scale <- vs
      shrink_list[[length(shrink_list) + 1]] <- ss
    }
  }
}

top_all <- if (length(top_list)) bind_rows(top_list) else NULL
shrink_all <- if (length(shrink_list)) bind_rows(shrink_list) else NULL

joined <- if (!is.null(top_all)) {
  # We only join on var_scale to keep your original top columns intact
  left_join(top_all, overall, by = "var_scale")
} else {
  overall
}

# -----------------------------
# Write outputs
# -----------------------------
out_overall <- file.path(out_dir, "ident_overall_from_rds.csv")
out_bycond  <- file.path(out_dir, "ident_by_condition_from_rds.csv")
out_joined  <- file.path(out_dir, "ident_top_joined_from_rds.csv")
out_shrink  <- file.path(out_dir, "ident_shrink_from_rds.csv")

readr::write_csv(overall, out_overall)
readr::write_csv(by_cond, out_bycond)
if (!is.null(joined)) readr::write_csv(joined, out_joined)
if (!is.null(shrink_all)) readr::write_csv(shrink_all, out_shrink)

cat("\nIdentifiability diagnostics written to:\n")
cat(" - ", out_overall, "\n", sep = "")
cat(" - ", out_bycond,  "\n", sep = "")
if (!is.null(joined)) cat(" - ", out_joined,  "\n", sep = "")
if (!is.null(shrink_all)) cat(" - ", out_shrink,  "\n", sep = "")
cat("\nNotes:\n")
cat(" * 'onset_drift_mae' and 'compensation_corr' use implied onset = t_obs - E[delta];\n")
cat("   they will be NA if 't'/'d' were not saved inside the training RDS.\n")
cat(" * Baseline var_scale is ", baseline_vs, ". Drift metrics compare each vscale to that baseline.\n", sep="")
