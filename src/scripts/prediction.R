set.seed(42)

library(mclust)     
library(aricode)    
library(pROC)       
library(ggplot2)    
library(truncnorm)
library(dplyr)
library(tidyr)
library(clue)        
library(MCMCpack) 

# Get the delay aware VB implementation.
source("src/delay_VB.R")  # VB_gaussian_update().

data_all <- readRDS("data/generated_onset_promote_style.rds")

# Create an 80/20 train test split.
set.seed(42)
n_total   <- nrow(data_all$d)
train_prop <- 0.8

train_idx <- sample(1:n_total, size = floor(train_prop * n_total))
test_idx  <- setdiff(1:n_total, train_idx)

cat("Total patients:", n_total, "\n")
cat("Training patients:", length(train_idx), "\n")
cat("Test patients:", length(test_idx), "\n\n")

# Define a helper to slice the dataset by index.
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

    # Delay priors from data file are optional and can be added here.
    # delay_prior_df  = if (!is.null(data$delay_prior_df))  data$delay_prior_df  else NULL,
    # delay_dist_cond = if (!is.null(data$delay_dist_cond)) data$delay_dist_cond else NULL,
    # delay_mu_cond   = if (!is.null(data$delay_mu_cond))   data$delay_mu_cond   else NULL,
    # delay_sd_cond   = if (!is.null(data$delay_sd_cond))   data$delay_sd_cond   else NULL
  )
}

# Build train and test splits.
train_data <- subset_data(data_all, train_idx)
test_data  <- subset_data(data_all,  test_idx)

# Configure shared model settings.
K <- 10
epsilon <- 0.1
M <- train_data$M
cond_list <- train_data$cond_list

# Build generic delay priors for each condition.
delay_mu <- rtruncnorm(M, a = 0, mean = 5, sd = 1.5)
delay_sd <- rep(2, M)
mu0 <- delay_mu
sigma20 <- (delay_sd)^2

# # More complex prior setting for mixed prior dataset.
# df <- train_data$delay_prior_df
# stopifnot(!is.null(df))

# # Align the prior rows with the condition order.
# df <- df[match(cond_list, df$condition), ]

# # Mark rows for each family.
# is_g <- df$delay_dist == "gaussian"
# is_u <- df$delay_dist == "uniform"
# is_m <- df$delay_dist == "mixture2"

# # Allocate containers for mean and variance.
# mu0     <- numeric(M)
# sigma20 <- numeric(M)

# # Fill Gaussian entries.
# mu0[is_g]     <- df$delay_mu[is_g]
# sigma20[is_g] <- (df$delay_sd[is_g])^2

# # Fill Uniform entries using mean and variance.
# mu0[is_u]     <- (df$unif_a[is_u] + df$unif_b[is_u]) / 2
# sigma20[is_u] <- (df$unif_b[is_u] - df$unif_a[is_u])^2 / 12

# # Collapse two component mixture to mean and variance.
# mix_mean <- df$mix_w1*df$mix_mu1 + (1 - df$mix_w1)*df$mix_mu2
# mix_var  <- df$mix_w1*(df$mix_sd1^2 + (df$mix_mu1 - mix_mean)^2) +
#             (1 - df$mix_w1)*(df$mix_sd2^2 + (df$mix_mu2 - mix_mean)^2)
# mu0[is_m]     <- mix_mean[is_m]
# sigma20[is_m] <- mix_var[is_m]

# Set weakly informative priors for global parameters.
theta <- rep(1, K)
a     <- matrix(1,   M, K)
b     <- matrix(1,   M, K)
u     <- matrix(50,  M, K)
v     <- matrix(0.3, M, K)
alpha <- matrix(5,   M, K)
beta  <- matrix(750, M, K)
hyperparameters <- list(theta, a, b, u, v, alpha, beta)

# Train the delay aware model on the training set.
cat("TRAINING PHASE\n")

N_train <- train_data$N
init_Cstar_train <- t(rdirichlet(N_train, rep(1, K)))  # K by N_train.
init_Cstar_train <- t(init_Cstar_train) # N_train by K.

init_Dstar_train <- matrix(runif(N_train * M, 0, 1),  N_train, M)
init_pstar_train <- matrix(runif(N_train * M, 0, 10), N_train, M)
init_qstar_train <- matrix(runif(N_train * M, 1, 2),  N_train, M)
init_rstar_train <- matrix(runif(N_train * M, 0.01, 0.02), N_train, M)

# posterior_train <- VB_gaussian_update(
#   t_obs = train_data$t, d = train_data$d, rho = train_data$rho, tau = train_data$tau,
#   iota = train_data$iota, hyperparameters = hyperparameters,
#   initial_Cstar = init_Cstar_train, initial_Dstar = init_Dstar_train,
#   initial_pstar = init_pstar_train, initial_qstar = init_qstar_train,
#   initial_rstar = init_rstar_train, N = N_train, M = M,
#   K = K, epsilon = epsilon,
#   mu0 = mu0, sigma20 = sigma20,
#   sex = train_data$sex,
#   birth_conds = train_data$birth_conds, male_conds = train_data$male_conds,
#   female_conds = train_data$female_conds, cond_list = cond_list
# )

# # Save the trained posterior.
# saveRDS(posterior_train, file = "src/resultsonsetdata/posterior_val_delay_train.rds")
posterior_train <- readRDS("src/resultsonsetdata/posterior_val_delay_train.rds")

# Source delay aware predictive utilities.
source("src/ProMOTe_LTCby_delay.R")   # Probability by time.
source("src/ProMOTe_LTCt_delay.R")    # Expected time after cut.
source("src/ProMOTe_Predictive_delay.R")  # Predictive density.
source("src/ProMOTe_utility_delay.R")     # Helpers.

# Collect posterior hyperparameters for prediction.
pp <- posterior_train$posterior.parameters

a_post     <- pp$pi_a
b_post     <- pp$pi_b
u_post     <- pp$mu_u
v_post     <- pp$v_star
alpha_post <- pp$mu_alpha
beta_post  <- pp$mu_beta

theta_post <- if (!is.null(pp$gamma_alpha)) {
  pp$gamma_alpha / sum(pp$gamma_alpha)   # Normalise.
} else {
  rep(1 / K, K)
}

hyper_post <- list(theta_post, a_post, b_post, u_post, v_post, alpha_post, beta_post)

# Build a per patient view splitting observed, partial and unobserved conditions.
# Keep presence only before baseline empty by default.
make_view <- function(d_row, t_row, rho_i, tau_i, M) {
  M_obs_idx  <- which(d_row == 1 & !is.na(t_row) & t_row >= rho_i & t_row <= tau_i) # Fully observed in window.
  M_part_idx <- which(d_row == 1 & !is.na(t_row) & t_row <  rho_i)                  # Left censored before rho.
  all_idx    <- seq_len(M)
  M_unobs_idx <- setdiff(all_idx, union(M_obs_idx, M_part_idx))

  d_obs  <- rep.int(1L, length(M_obs_idx))
  t_obs  <- if (length(M_obs_idx)) t_row[M_obs_idx] else numeric(0)
  d_part <- rep.int(1L, length(M_part_idx))

  list(M_obs=M_obs_idx, M_part=M_part_idx, M_unobs=M_unobs_idx,
       d_obs=d_obs, t_obs=t_obs, d_part=d_part)
}

# Define location scale t distribution helpers.
plst <- function(x, df, mu, sigma) {
  sigma <- pmax(sigma, 1e-12)              
  pt((x - mu) / sigma, df = df)
}

dlst <- function(x, df, mu, sigma) {
  sigma <- pmax(sigma, 1e-12)              
  dt((x - mu) / sigma, df = df) / sigma
}

# Evaluate cluster recovery on the full observation window.
cat("TEST: cluster recovery via predictive \n")

N_test <- test_data$N
phi_mat <- matrix(NA_real_, N_test, K)

for (i in 1:N_test) {
  vview <- make_view(
    d_row = test_data$d[i, ],
    t_row = test_data$t[i, ],
    rho_i = test_data$rho[i],
    tau_i = test_data$tau[i],
    M = M
  )
  pred <- VB_gaussian_predictive_density(
    hyperparameters = hyper_post,
    M_obs  = vview$M_obs,
    M_part = vview$M_part,
    M_unobs= vview$M_unobs,
    d_obs  = vview$d_obs,
    t_obs  = vview$t_obs,
    d_part = vview$d_part,
    rho    = test_data$rho[i],
    tau    = test_data$tau[i],
    M      = M,
    mu0 =mu0,
    sigma20 = sigma20
  )
  phi_mat[i, ] <- pred$phi
}

# Align predicted cluster labels to true clusters using the Hungarian method.
true_clusters_test <- data_all$z[test_idx]
raw_pred_test <- max.col(phi_mat, ties.method = "first")  # One based index.

K_used <- max(K, max(raw_pred_test), max(true_clusters_test))
tab <- table(
  factor(raw_pred_test,      levels = 1:K_used),
  factor(true_clusters_test, levels = 1:K_used)
)
perm <- clue::solve_LSAP(tab, maximum = TRUE)
map  <- as.integer(perm)
aligned_pred_test <- map[raw_pred_test]

aligned_preds_test <- aligned_pred_test

# Compute clustering metrics.
acc_test <- mean(aligned_pred_test == true_clusters_test)
ari_test <- mclust::adjustedRandIndex(aligned_pred_test, true_clusters_test)
nmi_test <- aricode::NMI(aligned_pred_test, true_clusters_test)

cat("TEST SET (cluster recovery, aligned):\n")
cat("  Correct assignments:", round(acc_test * 100, 2), "%\n")
cat("  ARI:", round(ari_test, 4), " | NMI:", round(nmi_test, 4), "\n\n")

# Evaluate forward prediction after a random cut age.
cat("TEST: forward prediction via predictive \n")

set.seed(42)
cut_ages <- runif(N_test, min = 50, max = 90)
cut_mat  <- matrix(cut_ages, nrow = N_test, ncol = M)

# Recompute phi using only pre cut evidence.
phi_pre_mat <- matrix(NA_real_, N_test, K)

# Store predictive components for later mixing.
pred_list_pre <- vector("list", N_test)

for (i in 1:N_test) {
  cut_i <- cut_ages[i]
  vview <- make_view(
    d_row = test_data$d[i, ],
    t_row = test_data$t[i, ],
    rho_i = test_data$rho[i],
    tau_i = cut_i,     # Pre cut window ends at the current age.
    M = M
  )
  pred <- VB_gaussian_predictive_density(
    hyperparameters = hyper_post,
    M_obs  = vview$M_obs,
    M_part = vview$M_part,
    M_unobs= vview$M_unobs,
    d_obs  = vview$d_obs,
    t_obs  = vview$t_obs,
    d_part = vview$d_part,
    rho    = test_data$rho[i],
    tau    = cut_i,
    M      = M,
    mu0 =mu0,
    sigma20 = sigma20
  )
  phi_pre_mat[i, ] <- pred$phi
  pred_list_pre[[i]] <- pred
}

# Compute after cut presence probabilities and expected times.
P_after <- matrix(0, N_test, M)       # Probability of event in the interval.
E_t_after <- matrix(NA_real_, N_test, M)  # Expected time given survival past the cut.

for (i in 1:N_test) {
  cut_i <- cut_ages[i]
  T_i   <- test_data$tau[i]   # Predict to the end of observation.
  pred  <- pred_list_pre[[i]]

  # Mix using the cluster weights from the predictive parameters.
  probs_i <- probability_LTHC_by_T(
    parameters = pred,
    hyperparameters = hyper_post,
    T = T_i, tau = cut_i, M = M,
    mu0 =mu0,
    sigma20 = sigma20
  )
  P_after[i, ] <- probs_i

  # Compute expected time after the cut.
  Et_i <- expected_LTHC_t_after_tau(
    parameters = pred,
    hyperparameters = hyper_post,
    tau = cut_i,
    M = M,
    mu0 =mu0,
    sigma20 = sigma20
  )
  E_t_after[i, ] <- Et_i
}

# Build ground truth labels for after cut evaluation.
d_true   <- test_data$d
t_true   <- test_data$t
tau_true <- test_data$tau

is_pos <- (d_true == 1) & (t_true > cut_mat) & (t_true <= tau_true)   # Observed after cut event.
is_neg <- (d_true == 0) | ((d_true == 1) & (t_true <= cut_mat))       # Not after cut.
is_unk <- (d_true == 1) & (t_true > tau_true)                         # Right censored after cut.

eval_mask <- (is_pos | is_neg) & !is_unk

# Compute presence metrics and AUROC.
y_true <- as.integer(is_pos[eval_mask])
y_prob <- as.numeric(P_after[eval_mask])

y_pred <- as.integer(y_prob >= 0.5)
acc_after <- mean(y_pred == y_true)

roc_after <- NA_real_
if (length(unique(y_true)) > 1) {
  roc_obj <- pROC::roc(response = y_true, predictor = y_prob, quiet = TRUE)
  roc_after <- as.numeric(pROC::auc(roc_obj))
}

cat("FORWARD PREDICTION (AFTER-CUT):\n")
cat("  Disease presence accuracy:", round(acc_after * 100, 2), "%\n")
cat("  Disease presence AUROC:", ifelse(is.na(roc_after), "NA", round(roc_after, 4)), "\n\n")

# Compute diagnosis age mean absolute error for observed after cut events.
mae_mask <- is_pos & !is.na(E_t_after)
mae_vals <- abs(E_t_after[mae_mask] - t_true[mae_mask])
mae_after <- if (length(mae_vals) > 0) mean(mae_vals) else NA_real_

cat("  Diagnosis age MAE (observed after-cut events):",
    ifelse(is.na(mae_after), "NA", round(mae_after, 3)), "years\n\n")


# Evaluate the no delay baseline using the original code.
cat("NO-DELAY BASELINE — test script \n")

# Source the original no delay implementation.
source("src/ProMOTe_VB.R")           # VB_gaussian_update_old.
source("src/ProMOTe_Predictive.R")   # VB_gaussian_predictive_density.
source("src/ProMOTe_LTCby.R")        # probability_LTHC_by_T.
source("src/ProMOTe_LTCt.R")         # expected_LTHC_t_after_tau.
source("src/ProMOTe_utility.R")      # Helpers.

# Use weakly informative priors for the baseline.
theta <- rep(1, K)
a     <- matrix(1,   M, K)
b     <- matrix(1,   M, K)
u     <- matrix(50,  M, K)
v     <- matrix(0.3, M, K)
alpha <- matrix(5,   M, K)
beta  <- matrix(750, M, K)
hyperparameters <- list(theta, a, b, u, v, alpha, beta)

# Train the original no delay VB model.
cat("TRAINING PHASE (no-delay)\n")

N_train <- train_data$N
init_Cstar_train <- t(rdirichlet(N_train, rep(1, K))); init_Cstar_train <- t(init_Cstar_train)
init_Dstar_train <- matrix(runif(N_train * M, 0, 1),  N_train, M)
init_pstar_train <- matrix(runif(N_train * M, 0, 10), N_train, M)
init_qstar_train <- matrix(runif(N_train * M, 1, 2),  N_train, M)
init_rstar_train <- matrix(runif(N_train * M, 0.01, 0.02), N_train, M)

# fit_nd <- VB_gaussian_update(
#   d = train_data$d, t = train_data$t, rho = train_data$rho, tau = train_data$tau, iota = train_data$iota,
#   hyperparameters = hyperparameters,
#   initial_Cstar = init_Cstar_train, initial_Dstar = init_Dstar_train,
#   initial_pstar = init_pstar_train, initial_qstar = init_qstar_train,
#   initial_rstar = init_rstar_train,
#   N = N_train, M = M, K = K, epsilon = epsilon,
#   sex = train_data$sex,
#   birth_conds = train_data$birth_conds, male_conds = train_data$male_conds,
#   female_conds = train_data$female_conds, cond_list = cond_list
# )

# # Save the baseline posterior to disk.
# saveRDS(fit_nd, file = "src/resultsonsetdata/posterior_val_no_delay_train.rds")
fit_nd <- readRDS("src/resultsonsetdata/posterior_val_no_delay_train.rds")

# Gather posterior parameters for baseline prediction.
pp <- fit_nd$posterior.parameters
theta_post  <- if (!is.null(pp$theta_star)) pp$theta_star / sum(pp$theta_star) else rep(1/K, K)
a_post      <- pp$a_star
b_post      <- pp$b_star
u_post      <- pp$u_star
v_post      <- pp$v_star
alpha_post  <- pp$alpha_star
beta_post   <- pp$beta_star
hyper_post_nd <- list(theta_post, a_post, b_post, u_post, v_post, alpha_post, beta_post)

# Build the patient view for the baseline.
make_view <- function(d_row, t_row, rho_i, tau_i, M) {
  M_obs_idx  <- which(d_row == 1 & !is.na(t_row) & t_row >= rho_i & t_row <= tau_i)  # Fully observed in window.
  M_part_idx <- which(d_row == 1 & !is.na(t_row) & t_row <  rho_i)                   # Left censored before rho.
  all_idx    <- seq_len(M)
  M_unobs_idx<- setdiff(all_idx, union(M_obs_idx, M_part_idx))
  list(
    M_obs  = M_obs_idx,
    M_part = M_part_idx,
    M_unobs= M_unobs_idx,
    d_obs  = rep.int(1L, length(M_obs_idx)),
    t_obs  = if (length(M_obs_idx)) t_row[M_obs_idx] else numeric(0),
    d_part = rep.int(1L, length(M_part_idx))
  )
}

# Provide local helpers if utilities were not sourced.
plst <- function(x, df, mu, sigma) { sigma <- pmax(sigma, 1e-12); pt((x - mu)/sigma, df=df) }
dlst <- function(x, df, mu, sigma) { sigma <- pmax(sigma, 1e-12); dt((x - mu)/sigma, df=df)/sigma }

# Evaluate cluster recovery for the baseline on the full window.
cat("TEST: cluster recovery (no-delay) \n")

N_test <- test_data$N
phi_mat <- matrix(NA_real_, N_test, K)

for (i in 1:N_test) {
  vview <- make_view(test_data$d[i,], test_data$t[i,], test_data$rho[i], test_data$tau[i], M)
  pred <- VB_gaussian_predictive_density(
    hyperparameters = hyper_post_nd,
    M_obs  = vview$M_obs,
    M_part = vview$M_part,
    M_unobs= vview$M_unobs,
    d_obs  = vview$d_obs,
    t_obs  = vview$t_obs,
    d_part = vview$d_part,
    rho    = test_data$rho[i],
    tau    = test_data$tau[i],
    M      = M
  )
  phi_mat[i, ] <- pred$phi
}

# Align predicted clusters and compute clustering metrics.
true_clusters_test <- data_all$z[test_idx]
raw_pred <- max.col(phi_mat, ties.method = "first")

K_used <- max(K, max(raw_pred), max(true_clusters_test))
tab <- table(factor(raw_pred, levels=1:K_used),
             factor(true_clusters_test, levels=1:K_used))
perm <- clue::solve_LSAP(tab, maximum = TRUE)
map  <- as.integer(perm)
aligned_pred <- map[raw_pred]

acc_test <- mean(aligned_pred == true_clusters_test)
ari_test <- mclust::adjustedRandIndex(aligned_pred, true_clusters_test)
nmi_test <- aricode::NMI(aligned_pred, true_clusters_test)

cat("NO-DELAY TEST (aligned):\n")
cat("  Correct assignments:", round(acc_test*100, 2), "%\n")
cat("  ARI:", round(ari_test, 4), " | NMI:", round(nmi_test, 4), "\n\n")

# Evaluate forward prediction for the baseline with a random cut in the range.
cat("TEST: forward prediction (no-delay)\n")

set.seed(42)
cut_ages <- runif(N_test, min = 50, max = 90)
cut_mat  <- matrix(cut_ages, nrow = N_test, ncol = M)

phi_pre_mat   <- matrix(NA_real_, N_test, K)
pred_list_pre <- vector("list", N_test)

for (i in 1:N_test) {
  cut_i <- cut_ages[i]
  vview <- make_view(test_data$d[i,], test_data$t[i,], test_data$rho[i], cut_i, M)
  pred <- VB_gaussian_predictive_density(
    hyperparameters = hyper_post_nd,
    M_obs  = vview$M_obs,
    M_part = vview$M_part,
    M_unobs= vview$M_unobs,
    d_obs  = vview$d_obs,
    t_obs  = vview$t_obs,
    d_part = vview$d_part,
    rho    = test_data$rho[i],
    tau    = cut_i,
    M      = M
  )
  phi_pre_mat[i, ] <- pred$phi
  pred_list_pre[[i]] <- pred
}

# Compute after cut probabilities and expected times for the baseline.
P_after   <- matrix(0, N_test, M)
E_t_after <- matrix(NA_real_, N_test, M)

for (i in 1:N_test) {
  cut_i <- cut_ages[i]
  T_i   <- test_data$tau[i]
  pred  <- pred_list_pre[[i]]

  P_after[i, ]   <- probability_LTHC_by_T(pred, hyper_post_nd, T=T_i, tau=cut_i, M=M)
  E_t_after[i, ] <- expected_LTHC_t_after_tau(pred, hyper_post_nd, tau=cut_i, M=M)
}

# Build labels for evaluation after the cut.
d_true   <- test_data$d
t_true   <- test_data$t
tau_true <- test_data$tau

is_pos <- (d_true == 1) & (t_true > cut_mat) & (t_true <= tau_true)
is_neg <- (d_true == 0) | ((d_true == 1) & (t_true <= cut_mat))
is_unk <- (d_true == 1) & (t_true > tau_true)
eval_mask <- (is_pos | is_neg) & !is_unk

# Compute presence metrics for the baseline.
y_true <- as.integer(is_pos[eval_mask])
y_prob <- as.numeric(P_after[eval_mask])
y_pred <- as.integer(y_prob >= 0.5)

acc_after <- mean(y_pred == y_true)
roc_after <- if (length(unique(y_true)) > 1) {
  as.numeric(pROC::auc(pROC::roc(y_true, y_prob, quiet = TRUE)))
} else NA_real_

# Compute diagnosis age mean absolute error for observed after cut events.
mae_mask <- is_pos & !is.na(E_t_after)
mae_vals <- abs(E_t_after[mae_mask] - t_true[mae_mask])
mae_after <- if (length(mae_vals) > 0) mean(mae_vals) else NA_real_

cat("NO-DELAY FORWARD PREDICTION:\n")
cat("  Presence accuracy:", round(acc_after * 100, 2), "%\n")
cat("  Presence AUROC:", ifelse(is.na(roc_after), "NA", round(roc_after, 4)), "\n")
cat("  Diagnosis age MAE:", ifelse(is.na(mae_after), "NA", round(mae_after, 3)), "years\n\n")
