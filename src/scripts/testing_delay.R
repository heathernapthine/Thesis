
set.seed(42)

library(mclust)     
library(aricode)    
library(pROC)       
library(ggplot2)    
library(truncnorm)
library(dplyr)
library(tidyr)
library(clue)       # Hungarian alignment (solve_LSAP)
library(MCMCpack)   # rdirichlet

source("src/newerVB.R")  # must export VB_gaussian_update() and VB_infer_with_frozen_globals()

# --- Load data ---
data_all <- readRDS("data/generated_presence_promote_style.rds")

# --- 80/20 train/test split (ProMOTe style) ---
set.seed(42)
n_total   <- nrow(data_all$d)
train_prop <- 0.8

train_idx <- sample(1:n_total, size = floor(train_prop * n_total))
test_idx  <- setdiff(1:n_total, train_idx)

cat("Total patients:", n_total, "\n")
cat("Training patients:", length(train_idx), "\n")
cat("Test patients:", length(test_idx), "\n\n")

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
    cond_list = data$cond_list#,

    # delay_prior_df  = if (!is.null(data$delay_prior_df))  data$delay_prior_df  else NULL,
    # delay_dist_cond = if (!is.null(data$delay_dist_cond)) data$delay_dist_cond else NULL,
    # delay_mu_cond   = if (!is.null(data$delay_mu_cond))   data$delay_mu_cond   else NULL,
    # delay_sd_cond   = if (!is.null(data$delay_sd_cond))   data$delay_sd_cond   else NULL
  )
}

train_data <- subset_data(data_all, train_idx)
test_data  <- subset_data(data_all,  test_idx)

# --- Model configuration (shared) ---
K <- 10
epsilon <- 0.1
M <- train_data$M
cond_list <- train_data$cond_list

# --- Delay priors (shared) ---
delay_mu <- rtruncnorm(M, a = 0, mean = 5, sd = 1.5)
delay_sd <- rep(2, M)
mu0 <- delay_mu
sigma20 <- (delay_sd)^2



# # --- Delay priors (vectorised from file) ---
# df <- train_data$delay_prior_df
# stopifnot(!is.null(df))

# # ensure same order as cond_list
# df <- df[match(cond_list, df$condition), ]

# is_g <- df$delay_dist == "gaussian"
# is_u <- df$delay_dist == "uniform"
# is_m <- df$delay_dist == "mixture2"

# mu0     <- numeric(M)
# sigma20 <- numeric(M)

# # Gaussian
# mu0[is_g]     <- df$delay_mu[is_g]
# sigma20[is_g] <- (df$delay_sd[is_g])^2

# # Uniform
# mu0[is_u]     <- (df$unif_a[is_u] + df$unif_b[is_u]) / 2
# sigma20[is_u] <- (df$unif_b[is_u] - df$unif_a[is_u])^2 / 12

# # Mixture2 -> collapsed
# mix_mean <- df$mix_w1*df$mix_mu1 + (1 - df$mix_w1)*df$mix_mu2
# mix_var  <- df$mix_w1*(df$mix_sd1^2 + (df$mix_mu1 - mix_mean)^2) +
#             (1 - df$mix_w1)*(df$mix_sd2^2 + (df$mix_mu2 - mix_mean)^2)
# mu0[is_m]     <- mix_mean[is_m]
# sigma20[is_m] <- mix_var[is_m]



# --- Weakly-informative hyperparameters (shared) ---
theta <- rep(1, K)
a     <- matrix(1,   M, K)
b     <- matrix(1,   M, K)
u     <- matrix(50,  M, K)
v     <- matrix(0.3, M, K)
alpha <- matrix(5,   M, K)
beta  <- matrix(750, M, K)
hyperparameters <- list(theta, a, b, u, v, alpha, beta)

# --- TRAIN: learn global/cluster parameters on the training set ---
cat("=== TRAINING PHASE ===\n")

N_train <- train_data$N
init_Cstar_train <- t(rdirichlet(N_train, rep(1, K)))  # K x N_train
init_Cstar_train <- t(init_Cstar_train)                 # N_train x K

init_Dstar_train <- matrix(runif(N_train * M, 0, 1),  N_train, M)
init_pstar_train <- matrix(runif(N_train * M, 0, 10), N_train, M)
init_qstar_train <- matrix(runif(N_train * M, 1, 2),  N_train, M)
init_rstar_train <- matrix(runif(N_train * M, 0.01, 0.02), N_train, M)

posterior_train <- VB_gaussian_update(
  t_obs = train_data$t, d = train_data$d, rho = train_data$rho, tau = train_data$tau,
  iota = train_data$iota, hyperparameters = hyperparameters,
  initial_Cstar = init_Cstar_train, initial_Dstar = init_Dstar_train,
  initial_pstar = init_pstar_train, initial_qstar = init_qstar_train,
  initial_rstar = init_rstar_train, N = N_train, M = M,
  K = K, epsilon = epsilon,
  mu0 = mu0, sigma20 = sigma20,
  sex = train_data$sex,
  birth_conds = train_data$birth_conds, male_conds = train_data$male_conds,
  female_conds = train_data$female_conds, cond_list = cond_list
)

saveRDS(posterior_train, file = "src/results/posterior_val_delay_train.rds")
posterior_train <- readRDS("src/results/posterior_val_delay_train.rds")

cat("Training completed in", posterior_train$n_steps, "steps\n\n")

# ---- Extract global posterior means from training (π̂, μ̂, σ̂) ----
pp <- posterior_train$posterior.parameters

# π̂_{m,k} = E[π] = a/(a+b)
pi_hat <- pp$pi_a / (pp$pi_a + pp$pi_b)          # M x K

# For NIG(μ, σ²): σ²_hat = E[σ²] = β/(α-1), for α>1
sigma2_hat <- pp$mu_beta / (pp$mu_alpha - 1)     # M x K
sigma2_hat[!is.finite(sigma2_hat)] <- 1          # guard
sigma_hat  <- sqrt(pmax(sigma2_hat, 1e-6))

# μ̂ = u
mu_hat <- pp$mu_u                                 # M x K

# Mixture weights γ̂
if (!is.null(pp$gamma_alpha)) {
  gamma_hat <- pp$gamma_alpha / sum(pp$gamma_alpha)  # length K
} else {
  gamma_hat <- rep(1/K, K)
}

fixed_globals <- list(
  gamma = gamma_hat,   # length K
  pi    = pi_hat,      # M x K
  mu    = mu_hat,      # M x K
  sigma = sigma_hat    # M x K
)

# --- TEST: infer with FROZEN global params (no refit of globals) ---
cat("=== TEST INFERENCE (globals frozen) ===\n")

N_test <- test_data$N
# Fresh patient-level initialisations (memberships etc.)
init_Cstar_test <- t(rdirichlet(N_test, rep(1, K))); init_Cstar_test <- t(init_Cstar_test)
init_Dstar_test <- matrix(runif(N_test * M, 0, 1),  N_test, M)
init_pstar_test <- matrix(runif(N_test * M, 0, 10), N_test, M)
init_qstar_test <- matrix(runif(N_test * M, 1, 2),  N_test, M)
init_rstar_test <- matrix(runif(N_test * M, 0.01, 0.02), N_test, M)

# Inference on FULL test data to assess cluster recovery (no masking)
test_full <- VB_infer_with_frozen_globals(
  t_obs = test_data$t, d = test_data$d, rho = test_data$rho, tau = test_data$tau,
  iota = test_data$iota,
  fixed_globals = fixed_globals,
  initial_Cstar = init_Cstar_test,
  initial_Dstar = init_Dstar_test,
  initial_pstar = init_pstar_test,
  initial_qstar = init_qstar_test,
  initial_rstar = init_rstar_test,
  N = N_test, M = M, K = K, epsilon = epsilon,
  sex = test_data$sex,
  birth_conds = test_data$birth_conds, male_conds = test_data$male_conds,
  female_conds = test_data$female_conds, cond_list = cond_list
)

saveRDS(test_full, file = "src/results/posterior_val_delay_test.rds")
test_full <- readRDS("src/results/posterior_val_delay_test.rds")


cat("Test inference (full) completed in", test_full$n_steps, "steps\n\n")

# --- Cluster recovery on test (globals frozen) ---
cat("=== CLUSTER RECOVERY ON TEST ===\n")
true_clusters_test <- data_all$z[test_idx]  # 1..K

# Use responsibilities (expected_z), not which.min(C)
C_test <- test_full$posterior.parameters$C_star
expected_z_test <- exp(-C_test); expected_z_test <- expected_z_test / rowSums(expected_z_test)
raw_pred_test <- max.col(expected_z_test, ties.method = "first")  # 1..K

# Hungarian alignment
K_used <- max(K, max(raw_pred_test), max(true_clusters_test))
tab <- table(
  factor(raw_pred_test,   levels = 1:K_used),
  factor(true_clusters_test, levels = 1:K_used)
)
perm <- clue::solve_LSAP(tab, maximum = TRUE)  # predicted row -> truth col
map  <- as.integer(perm)
aligned_pred_test <- map[raw_pred_test]

# Metrics AFTER alignment
acc_test <- mean(aligned_pred_test == true_clusters_test)
ari_test <- mclust::adjustedRandIndex(aligned_pred_test, true_clusters_test)
nmi_test <- aricode::NMI(aligned_pred_test, true_clusters_test)

cat("TEST SET (cluster recovery, aligned):\n")
cat("  Correct assignments:", round(acc_test * 100, 2), "%\n")
cat("  ARI:", round(ari_test, 4), " | NMI:", round(nmi_test, 4), "\n\n")

# --- Forward prediction protocol (ProMOTe): cut ages U(50,90) ---
set.seed(42)
cut_ages <- runif(N_test, min = 50, max = 90)
cut_mat  <- matrix(cut_ages, nrow = N_test, ncol = M)  # broadcast per-row cut

# Helper: mask events AFTER cut (keep before-cut data), for each patient
mask_after_cut <- function(d_row, t_row, cut_i) {
  d_before <- d_row
  d_before[d_row == 1 & t_row > cut_i] <- 0
  list(d_before = d_before, t_obs = t_row)  # keep t; hidden ones have d=0
}

# Build "before-cut" dataset view
d_before <- matrix(0, N_test, M)
t_before <- matrix(NA_real_, N_test, M)
for (i in 1:N_test) {
  res <- mask_after_cut(test_data$d[i, ], test_data$t[i, ], cut_ages[i])
  d_before[i, ] <- res$d_before
  t_before[i, ] <- res$t_obs
}

# Infer memberships using only BEFORE-CUT info, with globals frozen
init_Cstar_cut <- t(rdirichlet(N_test, rep(1, K))); init_Cstar_cut <- t(init_Cstar_cut)
init_Dstar_cut <- matrix(runif(N_test * M, 0, 1),  N_test, M)
init_pstar_cut <- matrix(runif(N_test * M, 0, 10), N_test, M)
init_qstar_cut <- matrix(runif(N_test * M, 1, 2),  N_test, M)
init_rstar_cut <- matrix(runif(N_test * M, 0.01, 0.02), N_test, M)

test_before <- VB_infer_with_frozen_globals(
  t_obs = t_before, d = d_before, rho = test_data$rho, tau = test_data$tau,
  iota = test_data$iota,
  fixed_globals = fixed_globals,
  initial_Cstar = init_Cstar_cut,
  initial_Dstar = init_Dstar_cut,
  initial_pstar = init_pstar_cut,
  initial_qstar = init_qstar_cut,
  initial_rstar = init_rstar_cut,
  N = N_test, M = M, K = K, epsilon = epsilon,
  sex = test_data$sex,
  birth_conds = test_data$birth_conds, male_conds = test_data$male_conds,
  female_conds = test_data$female_conds, cond_list = cond_list
)

saveRDS(test_before, file = "src/results/posterior_val_delay_test_before.rds")
test_before <- readRDS("src/results/posterior_val_delay_test_before.rds")

C_before <- test_before$posterior.parameters$C_star  # N_test x K
expected_z_before <- exp(-C_before); expected_z_before <- expected_z_before / rowSums(expected_z_before)

# --- Forward predictions for AFTER-CUT window ---
pnorm_safe <- function(x) pnorm(x)
dnorm_safe <- function(x) dnorm(x)

P_after   <- matrix(0, N_test, M)       # predicted prob event after cut
E_t_after <- matrix(NA_real_, N_test, M) # predicted E[t | event after cut]

for (i in 1:N_test) {
  for (m in 1:M) {
    one_minus_cdf <- 1 - pnorm_safe((cut_ages[i] - mu_hat[m, ]) / pmax(sigma_hat[m, ], 1e-6))
    prob_k <- pi_hat[m, ] * one_minus_cdf                 # length K
    P_after[i, m] <- sum(expected_z_before[i, ] * prob_k) # mixture over clusters

    # E[t | t>c] for each cluster (trunc normal lower=cut)
    a_k <- (cut_ages[i] - mu_hat[m, ]) / pmax(sigma_hat[m, ], 1e-6)
    denom_k <- pmax(1 - pnorm_safe(a_k), 1e-12)
    Et_k <- mu_hat[m, ] + pmax(sigma_hat[m, ], 1e-6) * (dnorm_safe(a_k) / denom_k)

    # Mix Et_k by posterior over clusters given "after" (normalize by P_after component)
    w_k <- expected_z_before[i, ] * prob_k
    if (sum(w_k) > 0) {
      w_k <- w_k / sum(w_k)
      E_t_after[i, m] <- sum(w_k * Et_k)
    } else {
      E_t_after[i, m] <- NA_real_
    }
  }
}

# --- Build ground-truth labels for AFTER-CUT evaluation ---
d_true  <- test_data$d
t_true  <- test_data$t
tau_true<- test_data$tau

is_pos <- (d_true == 1) & (t_true > cut_mat) & (t_true <= tau_true)   # observed after-cut event
is_neg <- (d_true == 0) | ((d_true == 1) & (t_true <= cut_mat))       # not after-cut
is_unk <- (d_true == 1) & (t_true > tau_true)                         # right-censored after cut

eval_mask <- (is_pos | is_neg) & !is_unk

# --- Disease presence metrics (accuracy, AUROC) on AFTER-CUT ---
y_true <- as.integer(is_pos[eval_mask])    # 1 for pos, 0 for neg
y_prob <- as.numeric(P_after[eval_mask])

# Accuracy with threshold 0.5
y_pred <- as.integer(y_prob >= 0.5)
acc_after <- mean(y_pred == y_true)

# AUROC (if both classes present)
roc_after <- NA_real_
if (length(unique(y_true)) > 1) {
  roc_obj <- pROC::roc(response = y_true, predictor = y_prob, quiet = TRUE)
  roc_after <- as.numeric(pROC::auc(roc_obj))
}

cat("=== FORWARD PREDICTION (AFTER-CUT) ===\n")
cat("Disease presence accuracy:", round(acc_after * 100, 2), "%\n")
cat("Disease presence AUROC:", ifelse(is.na(roc_after), "NA", round(roc_after, 4)), "\n\n")

# --- Diagnosis-age MAE for actually observed AFTER-CUT events ---
mae_mask <- is_pos & !is.na(E_t_after)
mae_vals <- abs(E_t_after[mae_mask] - t_true[mae_mask])
mae_after <- if (length(mae_vals) > 0) mean(mae_vals) else NA_real_

cat("Diagnosis age MAE (observed after-cut events):",
    ifelse(is.na(mae_after), "NA", round(mae_after, 3)), "years\n\n")

# Training-set cluster recovery with alignment ---
true_clusters_train <- data_all$z[train_idx]
C_train <- posterior_train$posterior.parameters$C_star
expected_z_train <- exp(-C_train); expected_z_train <- expected_z_train / rowSums(expected_z_train)
raw_pred_train <- max.col(expected_z_train, ties.method = "first")

K_used_tr <- max(K, max(raw_pred_train), max(true_clusters_train))
tab_tr <- table(
  factor(raw_pred_train,   levels = 1:K_used_tr),
  factor(true_clusters_train, levels = 1:K_used_tr)
)
perm_tr <- clue::solve_LSAP(tab_tr, maximum = TRUE)
map_tr  <- as.integer(perm_tr)
aligned_pred_train <- map_tr[raw_pred_train]

acc_train <- mean(aligned_pred_train == true_clusters_train)
ari_train <- mclust::adjustedRandIndex(aligned_pred_train, true_clusters_train)
nmi_train <- aricode::NMI(aligned_pred_train, true_clusters_train)

cat("TRAINING SET (aligned):\n")
cat("  Correct assignments:", round(acc_train * 100, 2), "%\n")
cat("  ARI:", round(ari_train, 4), " | NMI:", round(nmi_train, 4), "\n\n")

# ============================
# BASELINE (NO-DELAY) MODEL
# ============================
cat("\n============================\n")
cat(" BASELINE (NO-DELAY) MODEL \n")
cat("============================\n")

source("src/ProMOTe_VB.R")  # provides VB_gaussian_update_old()

# --- TRAIN baseline on the same training set ---
cat("=== TRAINING (baseline, no delay) ===\n")

N_train <- train_data$N
init_Cstar_train_old <- t(rdirichlet(N_train, rep(1, K))); init_Cstar_train_old <- t(init_Cstar_train_old)
init_Dstar_train_old <- matrix(runif(N_train * M, 0, 1),  N_train, M)
init_pstar_train_old <- matrix(runif(N_train * M, 0, 10), N_train, M)
init_qstar_train_old <- matrix(runif(N_train * M, 1, 2),  N_train, M)
init_rstar_train_old <- matrix(runif(N_train * M, 0.01, 0.02), N_train, M)

posterior_train_old <- VB_gaussian_update_old(
  d = train_data$d, t = train_data$t, rho = train_data$rho, tau = train_data$tau,
  iota = train_data$iota, hyperparameters = hyperparameters,
  initial_Cstar = init_Cstar_train_old, initial_Dstar = init_Dstar_train_old,
  initial_pstar = init_pstar_train_old, initial_qstar = init_qstar_train_old,
  initial_rstar = init_rstar_train_old, N = N_train, M = M, K = K, epsilon = epsilon,
  sex = train_data$sex, birth_conds = train_data$birth_conds,
  male_conds = train_data$male_conds, female_conds = train_data$female_conds,
  cond_list = cond_list
)

saveRDS(posterior_train_old, file = "src/results/posterior_train_old.rds")
posterior_train_old <- readRDS("src/results/posterior_train_old.rds")

cat("Baseline training completed in", posterior_train_old$n_steps, "steps\n\n")

# ---- Extract global posterior means (baseline) ----
pp_old <- posterior_train_old$posterior.parameters

pi_hat_old <- pp_old$a_star / (pp_old$a_star + pp_old$b_star)   # M x K
sigma2_hat_old <- pp_old$beta_star / (pp_old$alpha_star - 1)    # M x K
sigma2_hat_old[!is.finite(sigma2_hat_old)] <- 1
sigma_hat_old <- sqrt(pmax(sigma2_hat_old, 1e-6))
mu_hat_old <- pp_old$u_star                                     # M x K

if (!is.null(pp_old$theta_star)) {
  gamma_hat_old <- pp_old$theta_star / sum(pp_old$theta_star)    # length K
} else {
  gamma_hat_old <- rep(1/K, K)
}

fixed_globals_old <- list(
  gamma = gamma_hat_old,
  pi    = pi_hat_old,
  mu    = mu_hat_old,
  sigma = sigma_hat_old
)

# --- TEST inference (globals frozen) - FULL test for cluster recovery ---
cat("=== TEST (baseline, globals frozen) — full data ===\n")

N_test <- test_data$N
init_Cstar_test_old <- t(rdirichlet(N_test, rep(1, K))); init_Cstar_test_old <- t(init_Cstar_test_old)
init_Dstar_test_old <- matrix(runif(N_test * M, 0, 1),  N_test, M)
init_pstar_test_old <- matrix(runif(N_test * M, 0, 10), N_test, M)
init_qstar_test_old <- matrix(runif(N_test * M, 1, 2),  N_test, M)
init_rstar_test_old <- matrix(runif(N_test * M, 0.01, 0.02), N_test, M)

test_full_old <- VB_infer_with_frozen_globals_nodelay(
  t_obs = test_data$t, d = test_data$d, rho = test_data$rho, tau = test_data$tau,
  iota = test_data$iota,
  fixed_globals = fixed_globals_old,
  initial_Cstar = init_Cstar_test_old,
  initial_Dstar = init_Dstar_test_old,
  initial_pstar = init_pstar_test_old,
  initial_qstar = init_qstar_test_old,
  initial_rstar = init_rstar_test_old,
  N = N_test, M = M, K = K, epsilon = epsilon,
  sex = test_data$sex,
  birth_conds = test_data$birth_conds, male_conds = test_data$male_conds,
  female_conds = test_data$female_conds, cond_list = cond_list
)

saveRDS(test_full_old, file = "src/results/posterior_val_test_full_old.rds")
test_full_old <- readRDS("src/results/posterior_val_test_full_old.rds")

cat("Baseline test (full) completed in", test_full_old$n_steps, "steps\n\n")

# --- Cluster recovery (baseline, aligned) ---
truth_test <- data_all$z[test_idx]
C_test_old <- test_full_old$posterior.parameters$C_star
ez_test_old <- exp(-C_test_old); ez_test_old <- ez_test_old / rowSums(ez_test_old)
raw_pred_test_old <- max.col(ez_test_old, ties.method = "first")

K_used_old <- max(K, max(raw_pred_test_old), max(truth_test))
tab_old <- table(factor(raw_pred_test_old, levels = 1:K_used_old),
                 factor(truth_test,         levels = 1:K_used_old))
perm_old <- clue::solve_LSAP(tab_old, maximum = TRUE)
map_old  <- as.integer(perm_old)
aligned_pred_test_old <- map_old[raw_pred_test_old]

acc_test_old <- mean(aligned_pred_test_old == truth_test)
ari_test_old <- mclust::adjustedRandIndex(aligned_pred_test_old, truth_test)
nmi_test_old <- aricode::NMI(aligned_pred_test_old, truth_test)

cat("BASELINE — TEST (cluster recovery, aligned):\n")
cat("  Correct assignments:", round(acc_test_old * 100, 2), "%\n")
cat("  ARI:", round(ari_test_old, 4), " | NMI:", round(nmi_test_old, 4), "\n\n")

# --- Forward prediction (baseline) using SAME cut_ages and masks as delay-aware ---
# Reuse cut_ages / cut_mat, d_before, t_before if they already exist.
if (!exists("cut_ages")) {
  set.seed(42)
  cut_ages <- runif(N_test, min = 50, max = 90)
}
cut_mat <- matrix(cut_ages, nrow = N_test, ncol = M)

if (!exists("d_before") || !exists("t_before")) {
  mask_after_cut <- function(d_row, t_row, cut_i) {
    d_before <- d_row
    d_before[d_row == 1 & t_row > cut_i] <- 0
    list(d_before = d_before, t_obs = t_row)
  }
  d_before <- matrix(0, N_test, M)
  t_before <- matrix(NA_real_, N_test, M)
  for (i in 1:N_test) {
    res <- mask_after_cut(test_data$d[i, ], test_data$t[i, ], cut_ages[i])
    d_before[i, ] <- res$d_before
    t_before[i, ] <- res$t_obs
  }
}

init_Cstar_cut_old <- t(rdirichlet(N_test, rep(1, K))); init_Cstar_cut_old <- t(init_Cstar_cut_old)
init_Dstar_cut_old <- matrix(runif(N_test * M, 0, 1),  N_test, M)
init_pstar_cut_old <- matrix(runif(N_test * M, 0, 10), N_test, M)
init_qstar_cut_old <- matrix(runif(N_test * M, 1, 2),  N_test, M)
init_rstar_cut_old <- matrix(runif(N_test * M, 0.01, 0.02), N_test, M)

test_before_old <- VB_infer_with_frozen_globals_nodelay(
  t_obs = t_before, d = d_before, rho = test_data$rho, tau = test_data$tau,
  iota = test_data$iota,
  fixed_globals = fixed_globals_old,
  initial_Cstar = init_Cstar_cut_old,
  initial_Dstar = init_Dstar_cut_old,
  initial_pstar = init_pstar_cut_old,
  initial_qstar = init_qstar_cut_old,
  initial_rstar = init_rstar_cut_old,
  N = N_test, M = M, K = K, epsilon = epsilon,
  sex = test_data$sex,
  birth_conds = test_data$birth_conds, male_conds = test_data$male_conds,
  female_conds = test_data$female_conds, cond_list = cond_list
)

saveRDS(test_before_old, file = "src/results/posterior_val_test_before_old.rds")
test_before_old <- readRDS("src/results/posterior_val_test_before_old.rds")

C_before_old <- test_before_old$posterior.parameters$C_star
ez_before_old <- exp(-C_before_old); ez_before_old <- ez_before_old / rowSums(ez_before_old)

# Predictions after cut (baseline)
pnorm_safe <- function(x) pnorm(x)
dnorm_safe <- function(x) dnorm(x)

P_after_old   <- matrix(0, N_test, M)
E_t_after_old <- matrix(NA_real_, N_test, M)

for (i in 1:N_test) {
  for (m in 1:M) {
    one_minus_cdf_old <- 1 - pnorm_safe((cut_ages[i] - mu_hat_old[m, ]) / pmax(sigma_hat_old[m, ], 1e-6))
    prob_k_old <- pi_hat_old[m, ] * one_minus_cdf_old
    P_after_old[i, m] <- sum(ez_before_old[i, ] * prob_k_old)

    a_k_old <- (cut_ages[i] - mu_hat_old[m, ]) / pmax(sigma_hat_old[m, ], 1e-6)
    denom_k_old <- pmax(1 - pnorm_safe(a_k_old), 1e-12)
    Et_k_old <- mu_hat_old[m, ] + pmax(sigma_hat_old[m, ], 1e-6) * (dnorm_safe(a_k_old) / denom_k_old)

    w_k_old <- ez_before_old[i, ] * prob_k_old
    if (sum(w_k_old) > 0) {
      w_k_old <- w_k_old / sum(w_k_old)
      E_t_after_old[i, m] <- sum(w_k_old * Et_k_old)
    } else {
      E_t_after_old[i, m] <- NA_real_
    }
  }
}

# Ground truth masks (reuse same as delay-aware)
d_true  <- test_data$d
t_true  <- test_data$t
tau_true<- test_data$tau

is_pos <- (d_true == 1) & (t_true > cut_mat) & (t_true <= tau_true)
is_neg <- (d_true == 0) | ((d_true == 1) & (t_true <= cut_mat))
is_unk <- (d_true == 1) & (t_true > tau_true)

eval_mask <- (is_pos | is_neg) & !is_unk

# Presence metrics (baseline)
y_true_old <- as.integer(is_pos[eval_mask])
y_prob_old <- as.numeric(P_after_old[eval_mask])

y_pred_old <- as.integer(y_prob_old >= 0.5)
acc_after_old <- mean(y_pred_old == y_true_old)

roc_after_old <- NA_real_
if (length(unique(y_true_old)) > 1) {
  roc_obj_old <- pROC::roc(response = y_true_old, predictor = y_prob_old, quiet = TRUE)
  roc_after_old <- as.numeric(pROC::auc(roc_obj_old))
}

cat("BASELINE — FORWARD PREDICTION (AFTER-CUT):\n")
cat("  Disease presence accuracy:", round(acc_after_old * 100, 2), "%\n")
cat("  Disease presence AUROC:", ifelse(is.na(roc_after_old), "NA", round(roc_after_old, 4)), "\n\n")

# Diagnosis-age MAE (baseline)
mae_mask_old <- is_pos & !is.na(E_t_after_old)
mae_vals_old <- abs(E_t_after_old[mae_mask_old] - t_true[mae_mask_old])
mae_after_old <- if (length(mae_vals_old) > 0) mean(mae_vals_old) else NA_real_

cat("BASELINE — Diagnosis age MAE (observed after-cut events):",
    ifelse(is.na(mae_after_old), "NA", round(mae_after_old, 3)), "years\n\n")


