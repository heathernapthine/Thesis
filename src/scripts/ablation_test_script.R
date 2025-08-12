set.seed(42)

library(mclust)
library(aricode)
library(pROC)
library(truncnorm)
library(dplyr)
library(tidyr)
library(clue)
library(MCMCpack)
library(purrr)
library(stringr)

source("src/delay_VB.R")                  # VB_gaussian_update(), VB_infer_with_frozen_globals()
source("src/ProMOTe_LTCby_delay.R")      # probability_LTHC_by_T()
source("src/ProMOTe_LTCt_delay.R")       # expected_LTHC_t_after_tau()
source("src/ProMOTe_Predictive_delay.R") # VB_gaussian_predictive_density()
source("src/ProMOTe_utility_delay.R")    # helpers

# -------------------------
# LOAD DATA & TRAIN/TEST
# -------------------------
data_all <- readRDS("data/generated_promote_style_mixed_delays.rds")

n_total <- nrow(data_all$d)
train_prop <- 0.8
train_idx <- sample.int(n_total, size = floor(train_prop * n_total))
test_idx  <- setdiff(seq_len(n_total), train_idx)

subset_data <- function(data, idx) {
  list(
    d = data$d[idx, , drop = FALSE],
    t = data$t[idx, , drop = FALSE],
    rho = data$rho[idx],
    tau = data$tau[idx],
    iota = data$iota[idx],
    onset = data$onset[idx, , drop = FALSE],
    N = length(idx),
    M = data$M,
    sex = if (!is.null(data$sex)) data$sex[idx] else NULL,
    birth_conds = data$birth_conds,
    male_conds = data$male_conds,
    female_conds = data$female_conds,
    cond_list = data$cond_list,
    delay_prior_df  = if (!is.null(data$delay_prior_df))  data$delay_prior_df  else NULL
  )
}

train_data <- subset_data(data_all, train_idx)
test_data  <- subset_data(data_all, test_idx)

# -------------------------
# SHARED CONFIG / PRIORS
# -------------------------
K <- 10
epsilon <- 0.1
M <- train_data$M
cond_list <- train_data$cond_list

stopifnot(!is.null(train_data$delay_prior_df))
df <- train_data$delay_prior_df
df <- df[match(cond_list, df$condition), ]

is_g <- df$delay_dist == "gaussian"
is_u <- df$delay_dist == "uniform"
is_m <- df$delay_dist == "mixture2"

mu0_base     <- numeric(M)
sigma20_base <- numeric(M)

mu0_base[is_g]     <- df$delay_mu[is_g]
sigma20_base[is_g] <- (df$delay_sd[is_g])^2

mu0_base[is_u]     <- (df$unif_a[is_u] + df$unif_b[is_u]) / 2
sigma20_base[is_u] <- (df$unif_b[is_u] - df$unif_a[is_u])^2 / 12

mix_mean <- df$mix_w1*df$mix_mu1 + (1 - df$mix_w1)*df$mix_mu2
mix_var  <- df$mix_w1*(df$mix_sd1^2 + (df$mix_mu1 - mix_mean)^2) +
            (1 - df$mix_w1)*(df$mix_sd2^2 + (df$mix_mu2 - mix_mean)^2)
mu0_base[is_m]     <- mix_mean[is_m]
sigma20_base[is_m] <- mix_var[is_m]

# robust length bins (avoid non-unique breaks)
make_length_bins <- function(x) {
  q <- quantile(x, probs = c(0, .33, .66, 1), na.rm = TRUE)
  if (anyDuplicated(q)) {
    r <- range(x, na.rm = TRUE)
    if (!is.finite(r[1]) || r[1] == r[2]) {
      r <- c(min(x, na.rm = TRUE) - 1e-6, max(x, na.rm = TRUE) + 1e-6)
    }
    q <- seq(r[1], r[2], length.out = 4)
    q[1] <- q[1] - 1e-9
    q[4] <- q[4] + 1e-9
  }
  cut(x, breaks = q, include.lowest = TRUE, labels = c("short","medium","long"))
}

# robust tightness bins for sigma20 (smaller = tighter prior)
make_tight_bins <- function(x) {
  q <- quantile(x, probs = c(0, .33, .66, 1), na.rm = TRUE)
  if (anyDuplicated(q)) {
    r <- range(x, na.rm = TRUE)
    if (!is.finite(r[1]) || r[1] == r[2]) {
      r <- c(min(x, na.rm = TRUE) - 1e-6, max(x, na.rm = TRUE) + 1e-6)
    }
    q <- seq(r[1], r[2], length.out = 4)
    q[1] <- q[1] - 1e-9
    q[4] <- q[4] + 1e-9
  }
  # labels order: tight (small var) -> mid -> loose (large var)
  cut(x, breaks = q, include.lowest = TRUE, labels = c("tight","mid","loose"))
}

delay_family <- df$delay_dist
delay_group  <- if (!is.null(df$delay_group)) as.character(df$delay_group) else as.character(df$delay_dist)
length_bins  <- make_length_bins(mu0_base)
tight_bins   <- make_tight_bins(sigma20_base)

# pick strong/weak subsets
strong_cols <- which(as.character(tight_bins) == "tight")
weak_cols   <- which(as.character(tight_bins) == "loose")
if (length(strong_cols) == 0 || length(weak_cols) == 0) {
  ord <- order(sigma20_base)
  n_take <- max(1, floor(M/3))
  strong_cols <- ord[seq_len(n_take)]
  weak_cols   <- ord[seq.int(from = M - n_take + 1, to = M)]
}

# Prior-strength grid. Smaller scale => stronger prior.
var_scales <- c(0.25, 0.5, 1, 2, 4)

# Weakly-informative global priors (fixed across ablation)
theta_h <- rep(1, K)
a_h     <- matrix(1,   M, K)
b_h     <- matrix(1,   M, K)
u_h     <- matrix(50,  M, K)
v_h     <- matrix(0.3, M, K)
alpha_h <- matrix(5,   M, K)
beta_h  <- matrix(750, M, K)
hyper_h <- list(theta_h, a_h, b_h, u_h, v_h, alpha_h, beta_h)

# -------------------------
# HELPERS
# -------------------------
make_view <- function(d_row, t_row, rho_i, tau_i, M) {
  M_obs_idx  <- which(d_row == 1 & !is.na(t_row) & t_row >= rho_i & t_row <= tau_i)
  M_part_idx <- which(d_row == 1 & !is.na(t_row) & t_row <  rho_i)
  all_idx    <- seq_len(M)
  M_unobs_idx <- setdiff(all_idx, union(M_obs_idx, M_part_idx))
  list(
    M_obs  = M_obs_idx, M_part = M_part_idx, M_unobs = M_unobs_idx,
    d_obs  = rep.int(1L, length(M_obs_idx)),
    t_obs  = if (length(M_obs_idx)) t_row[M_obs_idx] else numeric(0),
    d_part = rep.int(1L, length(M_part_idx))
  )
}

# restrict the view to a subset of columns (strong/weak)
make_view_subset <- function(d_row, t_row, rho_i, tau_i, M, cols) {
  v <- make_view(d_row, t_row, rho_i, tau_i, M)
  v$M_obs  <- intersect(v$M_obs,  cols)
  v$M_part <- intersect(v$M_part, cols)
  v$M_unobs <- setdiff(seq_len(M), union(v$M_obs, v$M_part))
  v$d_obs  <- rep.int(1L, length(v$M_obs))
  v$t_obs  <- if (length(v$M_obs)) t_row[v$M_obs] else numeric(0)
  v$d_part <- rep.int(1L, length(v$M_part))
  v
}

hungarian_align <- function(pred, truth, K) {
  K_used <- max(K, max(pred), max(truth))
  tab <- table(factor(pred, levels = 1:K_used), factor(truth, levels = 1:K_used))
  map <- as.integer(clue::solve_LSAP(tab, maximum = TRUE))
  map[pred]
}

safe_auc <- function(y, p) {
  if (length(unique(y)) > 1) as.numeric(pROC::auc(pROC::roc(y, p, quiet = TRUE))) else NA_real_
}

# -------------------------
# OUTPUT DIR
# -------------------------
directory <- "src/ablationresults"
if (!dir.exists(directory)) dir.create(directory, recursive = TRUE)

# -------------------------
# ONE SETTING (var_scale)
# -------------------------
run_one_setting <- function(var_scale) {
  tag <- sprintf("vscale_%s", str_replace(as.character(var_scale), "\\.", "p"))
  cache_train     <- file.path(directory, sprintf("train_%s.rds", tag))
  cache_test_full <- file.path(directory, sprintf("test_full_%s.rds", tag))

  mu0     <- mu0_base
  sigma20 <- pmax(sigma20_base * var_scale, 1e-8)  # guard against 0

  # --- TRAIN (with cache + field check) ---
  need_train <- TRUE
  if (file.exists(cache_train)) {
    posterior_train <- readRDS(cache_train)
    pp_chk <- posterior_train$posterior.parameters
    if (!is.null(pp_chk$gap_sigma2_star)) need_train <- FALSE
  }
  if (need_train) {
    set.seed(42 + round(100*var_scale))
    N_train <- train_data$N
    init_C <- t(rdirichlet(N_train, rep(1, K))); init_C <- t(init_C)
    init_D <- matrix(runif(N_train * M, 0, 1),  N_train, M)
    init_p <- matrix(runif(N_train * M, 0, 10), N_train, M)
    init_q <- matrix(runif(N_train * M, 1, 2),  N_train, M)
    init_r <- matrix(runif(N_train * M, 0.01, 0.02), N_train, M)

    posterior_train <- VB_gaussian_update(
      t_obs = train_data$t, d = train_data$d, rho = train_data$rho, tau = train_data$tau,
      iota = train_data$iota, hyperparameters = hyper_h,
      initial_Cstar = init_C, initial_Dstar = init_D,
      initial_pstar = init_p, initial_qstar = init_q,
      initial_rstar = init_r, N = N_train, M = M,
      K = K, epsilon = epsilon,
      mu0 = mu0, sigma20 = sigma20,
      sex = train_data$sex,
      birth_conds = train_data$birth_conds, male_conds = train_data$male_conds,
      female_conds = train_data$female_conds, cond_list = cond_list
    )
    saveRDS(posterior_train, cache_train)
  }

  # --- EXTRACT GLOBALS FOR PREDICTION ---
  pp <- posterior_train$posterior.parameters
  theta_post <- if (!is.null(pp$gamma_alpha)) pp$gamma_alpha / sum(pp$gamma_alpha) else rep(1/K, K)
  a_post     <- pp$pi_a
  b_post     <- pp$pi_b
  u_post     <- pp$mu_u
  v_post     <- pp$v_star
  alpha_post <- pp$mu_alpha
  beta_post  <- pp$mu_beta
  hyper_post <- list(theta_post, a_post, b_post, u_post, v_post, alpha_post, beta_post)

  # --- SHRINKAGE (TRAIN SET), PER CONDITION ---
  gap_mu_tr  <- pp$gap_mu_star          # N_train x M
  gap_var_tr <- pp$gap_sigma2_star      # N_train x M
  ed_tr      <- pp$expected_d           # N_train x M

  # guard shapes
  stopifnot(is.matrix(gap_mu_tr), is.matrix(gap_var_tr), is.matrix(ed_tr))
  denom_w <- pmax(colSums(ed_tr, na.rm = TRUE), 1e-8)

  # unweighted
  SR   <- colMeans(gap_var_tr, na.rm = TRUE) / sigma20
  PP   <- (colMeans(gap_mu_tr,  na.rm = TRUE) - mu0) / sqrt(sigma20)

  # presence-weighted
  SR_w <- (colSums(gap_var_tr * ed_tr, na.rm = TRUE) / denom_w) / sigma20
  PP_w <- ((colSums(gap_mu_tr  * ed_tr, na.rm = TRUE) / denom_w) - mu0) / sqrt(sigma20)

  shrink_df <- data.frame(
    tag = tag, var_scale = var_scale,
    condition = cond_list,
    family = delay_family,
    length_bin = as.character(length_bins),
    tight_bin  = as.character(tight_bins),
    SR = SR, SR_w = SR_w,
    PP = PP, PP_w = PP_w,
    stringsAsFactors = FALSE
  )

  # --- CLUSTER RECOVERY (FULL WINDOW, TEST) ---
  N_test <- test_data$N
  phi_mat <- matrix(NA_real_, N_test, K)
  for (i in 1:N_test) {
    vview <- make_view(test_data$d[i, ], test_data$t[i, ], test_data$rho[i], test_data$tau[i], M)
    pred <- VB_gaussian_predictive_density(
      hyperparameters = hyper_post,
      M_obs  = vview$M_obs, M_part = vview$M_part, M_unobs = vview$M_unobs,
      d_obs  = vview$d_obs, t_obs  = vview$t_obs,  d_part  = vview$d_part,
      rho    = test_data$rho[i], tau = test_data$tau[i], M = M,
      mu0 = mu0, sigma20 = sigma20
    )
    phi_mat[i, ] <- pred$phi
  }
  truth_clusters <- data_all$z[test_idx]
  raw_pred <- max.col(phi_mat, ties.method = "first")
  aligned_pred <- hungarian_align(raw_pred, truth_clusters, K)
  acc <- mean(aligned_pred == truth_clusters)
  ari <- mclust::adjustedRandIndex(aligned_pred, truth_clusters)
  nmi <- aricode::NMI(aligned_pred, truth_clusters)

  # --- FORWARD PREDICTION (TEST, ALL CONDITIONS) ---
  set.seed(4242 + round(100*var_scale))
  cut_ages <- runif(N_test, min = 50, max = 90)
  cut_mat  <- matrix(cut_ages, nrow = N_test, ncol = M)

  phi_pre_mat <- matrix(NA_real_, N_test, K)
  pred_list_pre <- vector("list", N_test)
  for (i in 1:N_test) {
    cut_i <- cut_ages[i]
    vview <- make_view(test_data$d[i, ], test_data$t[i, ], test_data$rho[i], cut_i, M)
    pred <- VB_gaussian_predictive_density(
      hyperparameters = hyper_post,
      M_obs  = vview$M_obs, M_part = vview$M_part, M_unobs = vview$M_unobs,
      d_obs  = vview$d_obs, t_obs  = vview$t_obs,  d_part  = vview$d_part,
      rho    = test_data$rho[i], tau = cut_i, M = M,
      mu0 = mu0, sigma20 = sigma20
    )
    phi_pre_mat[i, ] <- pred$phi
    pred_list_pre[[i]] <- pred
  }

  P_after <- matrix(0, N_test, M)
  E_t_after <- matrix(NA_real_, N_test, M)
  for (i in 1:N_test) {
    cut_i <- cut_ages[i]
    T_i   <- test_data$tau[i]
    pred  <- pred_list_pre[[i]]
    P_after[i, ] <- probability_LTHC_by_T(pred, hyper_post, T = T_i, tau = cut_i, M = M,
                                          mu0 = mu0, sigma20 = sigma20)
    E_t_after[i, ] <- expected_LTHC_t_after_tau(pred, hyper_post, tau = cut_i, M = M,
                                                mu0 = mu0, sigma20 = sigma20)
  }

  d_true   <- test_data$d
  t_true   <- test_data$t
  tau_true <- test_data$tau

  is_pos <- (d_true == 1) & (t_true > cut_mat) & (t_true <= tau_true)
  is_neg <- (d_true == 0) | ((d_true == 1) & (t_true <= cut_mat))
  is_unk <- (d_true == 1) & (t_true > tau_true)
  eval_mask <- (is_pos | is_neg) & !is_unk

  y_true <- as.integer(is_pos[eval_mask])
  y_prob <- as.numeric(P_after[eval_mask])
  y_pred <- as.integer(y_prob >= 0.5)

  acc_after   <- mean(y_pred == y_true)
  auroc_after <- safe_auc(y_true, y_prob)

  mae_mask <- is_pos & !is.na(E_t_after)
  mae_vals <- abs(E_t_after[mae_mask] - t_true[mae_mask])
  mae_after <- if (length(mae_vals) > 0) mean(mae_vals) else NA_real_

  # --- DROP-SET SENSITIVITY: STRONG vs WEAK PRIORS ONLY ---
  # Re-run predictive using only strong / only weak columns as evidence.
  pred_list_pre_strong <- vector("list", N_test)
  pred_list_pre_weak   <- vector("list", N_test)

  for (i in 1:N_test) {
    cut_i <- cut_ages[i]

    v_strong <- make_view_subset(test_data$d[i, ], test_data$t[i, ], test_data$rho[i], cut_i, M, strong_cols)
    pred_s <- VB_gaussian_predictive_density(
      hyperparameters = hyper_post,
      M_obs  = v_strong$M_obs, M_part = v_strong$M_part, M_unobs = v_strong$M_unobs,
      d_obs  = v_strong$d_obs, t_obs  = v_strong$t_obs,  d_part  = v_strong$d_part,
      rho    = test_data$rho[i], tau = cut_i, M = M,
      mu0 = mu0, sigma20 = sigma20
    )
    pred_list_pre_strong[[i]] <- pred_s

    v_weak <- make_view_subset(test_data$d[i, ], test_data$t[i, ], test_data$rho[i], cut_i, M, weak_cols)
    pred_w <- VB_gaussian_predictive_density(
      hyperparameters = hyper_post,
      M_obs  = v_weak$M_obs, M_part = v_weak$M_part, M_unobs = v_weak$M_unobs,
      d_obs  = v_weak$d_obs, t_obs  = v_weak$t_obs,  d_part  = v_weak$d_part,
      rho    = test_data$rho[i], tau = cut_i, M = M,
      mu0 = mu0, sigma20 = sigma20
    )
    pred_list_pre_weak[[i]] <- pred_w
  }

  P_after_strong <- matrix(0, N_test, M)
  E_t_after_strong <- matrix(NA_real_, N_test, M)
  P_after_weak <- matrix(0, N_test, M)
  E_t_after_weak <- matrix(NA_real_, N_test, M)

  for (i in 1:N_test) {
    cut_i <- cut_ages[i]
    T_i   <- test_data$tau[i]

    pred_s <- pred_list_pre_strong[[i]]
    P_after_strong[i, ] <- probability_LTHC_by_T(pred_s, hyper_post, T = T_i, tau = cut_i, M = M,
                                                 mu0 = mu0, sigma20 = sigma20)
    E_t_after_strong[i, ] <- expected_LTHC_t_after_tau(pred_s, hyper_post, tau = cut_i, M = M,
                                                       mu0 = mu0, sigma20 = sigma20)

    pred_w <- pred_list_pre_weak[[i]]
    P_after_weak[i, ] <- probability_LTHC_by_T(pred_w, hyper_post, T = T_i, tau = cut_i, M = M,
                                               mu0 = mu0, sigma20 = sigma20)
    E_t_after_weak[i, ] <- expected_LTHC_t_after_tau(pred_w, hyper_post, tau = cut_i, M = M,
                                                     mu0 = mu0, sigma20 = sigma20)
  }

  # Evaluate on the respective subsets
  eval_mask_strong <- eval_mask; eval_mask_strong[,] <- FALSE; eval_mask_strong[, strong_cols] <- eval_mask[, strong_cols]
  y_true_s <- as.integer(is_pos[eval_mask_strong])
  y_prob_s <- as.numeric(P_after_strong[eval_mask_strong])
  y_pred_s <- as.integer(y_prob_s >= 0.5)
  acc_after_strong   <- if (length(y_true_s)) mean(y_pred_s == y_true_s) else NA_real_
  auroc_after_strong <- if (length(y_true_s)) safe_auc(y_true_s, y_prob_s) else NA_real_
  mae_mask_s <- (is_pos & !is.na(E_t_after_strong)); mae_mask_s[,] <- mae_mask_s[,]
  mae_mask_s[,] <- FALSE; mae_mask_s[, strong_cols] <- (is_pos[, strong_cols] & !is.na(E_t_after_strong[, strong_cols]))
  mae_vals_s <- abs(E_t_after_strong[mae_mask_s] - t_true[mae_mask_s])
  mae_after_strong <- if (length(mae_vals_s) > 0) mean(mae_vals_s) else NA_real_

  eval_mask_weak <- eval_mask; eval_mask_weak[,] <- FALSE; eval_mask_weak[, weak_cols] <- eval_mask[, weak_cols]
  y_true_w <- as.integer(is_pos[eval_mask_weak])
  y_prob_w <- as.numeric(P_after_weak[eval_mask_weak])
  y_pred_w <- as.integer(y_prob_w >= 0.5)
  acc_after_weak   <- if (length(y_true_w)) mean(y_pred_w == y_true_w) else NA_real_
  auroc_after_weak <- if (length(y_true_w)) safe_auc(y_true_w, y_prob_w) else NA_real_
  mae_mask_w <- (is_pos & !is.na(E_t_after_weak)); mae_mask_w[,] <- mae_mask_w[,]
  mae_mask_w[,] <- FALSE; mae_mask_w[, weak_cols] <- (is_pos[, weak_cols] & !is.na(E_t_after_weak[, weak_cols]))
  mae_vals_w <- abs(E_t_after_weak[mae_mask_w] - t_true[mae_mask_w])
  mae_after_weak <- if (length(mae_vals_w) > 0) mean(mae_vals_w) else NA_real_

  # --- STRATIFIED FORWARD METRICS ---
  fam_levels <- unique(delay_family)
  len_levels <- c("short","medium","long")
  strat_rows <- list()

  # by family
  for (fam in fam_levels) {
    cols <- which(delay_family == fam)
    if (length(cols) == 0) next
    mask_fam <- eval_mask; mask_fam[,] <- FALSE; mask_fam[, cols] <- eval_mask[, cols]
    y_true_f <- as.integer(is_pos[mask_fam])
    y_prob_f <- as.numeric(P_after[mask_fam])
    y_pred_f <- as.integer(y_prob_f >= 0.5)
    acc_f <- if (length(y_true_f)) mean(y_pred_f == y_true_f) else NA_real_
    auc_f <- if (length(y_true_f)) safe_auc(y_true_f, y_prob_f) else NA_real_

    mae_mask_f <- is_pos & !is.na(E_t_after); mae_mask_f[,] <- FALSE
    mae_mask_f[, cols] <- (is_pos[, cols] & !is.na(E_t_after[, cols]))
    mae_vals_f <- abs(E_t_after[mae_mask_f] - t_true[mae_mask_f])
    mae_f <- if (length(mae_vals_f) > 0) mean(mae_vals_f) else NA_real_

    strat_rows[[length(strat_rows)+1]] <- data.frame(
      tag = tag, var_scale = var_scale, group_type = "family", group = fam,
      acc_after = acc_f, auroc_after = auc_f, mae_after = mae_f, stringsAsFactors = FALSE
    )
  }

  # by length bin
  for (lev in len_levels) {
    cols <- which(as.character(length_bins) == lev)
    if (length(cols) == 0) next
    mask_len <- eval_mask; mask_len[,] <- FALSE; mask_len[, cols] <- eval_mask[, cols]
    y_true_l <- as.integer(is_pos[mask_len])
    y_prob_l <- as.numeric(P_after[mask_len])
    y_pred_l <- as.integer(y_prob_l >= 0.5)
    acc_l <- if (length(y_true_l)) mean(y_pred_l == y_true_l) else NA_real_
    auc_l <- if (length(y_true_l)) safe_auc(y_true_l, y_prob_l) else NA_real_

    mae_mask_l <- is_pos & !is.na(E_t_after); mae_mask_l[,] <- FALSE
    mae_mask_l[, cols] <- (is_pos[, cols] & !is.na(E_t_after[, cols]))
    mae_vals_l <- abs(E_t_after[mae_mask_l] - t_true[mae_mask_l])
    mae_l <- if (length(mae_vals_l) > 0) mean(mae_vals_l) else NA_real_

    strat_rows[[length(strat_rows)+1]] <- data.frame(
      tag = tag, var_scale = var_scale, group_type = "length_bin", group = lev,
      acc_after = acc_l, auroc_after = auc_l, mae_after = mae_l, stringsAsFactors = FALSE
    )
  }

  top <- data.frame(
    tag = tag, var_scale = var_scale,
    cluster_acc = acc, cluster_ari = ari, cluster_nmi = nmi,
    after_acc = acc_after, after_auroc = auroc_after, after_mae = mae_after,
    after_acc_strong = acc_after_strong, after_auroc_strong = auroc_after_strong, after_mae_strong = mae_after_strong,
    after_acc_weak   = acc_after_weak,   after_auroc_weak   = auroc_after_weak,   after_mae_weak   = mae_after_weak,
    # simple deltas (strong - weak)
    delta_acc_strong_minus_weak   = acc_after_strong - acc_after_weak,
    delta_auroc_strong_minus_weak = auroc_after_strong - auroc_after_weak,
    delta_mae_strong_minus_weak   = mae_after_strong - mae_after_weak,
    stringsAsFactors = FALSE
  )
  strat <- dplyr::bind_rows(strat_rows)

  saveRDS(list(top = top, strat = strat, shrink = shrink_df), cache_test_full)
  list(top = top, strat = strat, shrink = shrink_df)
}

# -------------------------
# RUN GRID & SAVE
# -------------------------
all_results <- lapply(var_scales, run_one_setting)

tops   <- bind_rows(lapply(all_results, `[[`, "top"))
strat  <- bind_rows(lapply(all_results, `[[`, "strat"))
shrink <- bind_rows(lapply(all_results, `[[`, "shrink"))

if (!dir.exists(directory)) dir.create(directory, recursive = TRUE)
write.csv(tops,   file.path(directory, "ablation_prior_strength_top.csv"),        row.names = FALSE)
write.csv(strat,  file.path(directory, "ablation_prior_strength_stratified.csv"), row.names = FALSE)
write.csv(shrink, file.path(directory, "ablation_prior_shrinkage_condition.csv"), row.names = FALSE)

cat("\nPRIOR-STRENGTH ABLATION (variance scales)\n")
print(tops %>% arrange(var_scale) %>% mutate(across(where(is.numeric), ~round(., 4))))

cat("\nSHRINKAGE (first few rows)\n")
print(head(shrink %>% arrange(var_scale, condition)))
