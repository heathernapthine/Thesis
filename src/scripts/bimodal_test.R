# BIMODAL-DELAY TEST SCRIPT WITH CENTERED SINGLE-GAUSSIAN DELAY PRIOR.
set.seed(42)

suppressPackageStartupMessages({
  library(mclust)
  library(aricode)
  library(pROC)
  library(ggplot2)
  library(truncnorm)
  library(dplyr)
  library(tidyr)
  library(clue)
  library(MCMCpack)
})

# PATHS AND SOURCE FILES.
data_path <- "data/generated_promote_style_bimodal_delays.rds"
out_dir   <- "src/closebimodal"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# DELAY-AWARE VB AND UTILITIES.
source("src/delay_VB.R")                 # VB_gaussian_update().
source("src/ProMOTe_LTCby_delay.R")      # probability_LTHC_by_T().
source("src/ProMOTe_LTCt_delay.R")       # expected_LTHC_t_after_tau().
source("src/ProMOTe_Predictive_delay.R") # VB_gaussian_predictive_density().
source("src/ProMOTe_utility_delay.R")    # Helper utilities for delay-aware code.

# NO-DELAY BASELINE VB AND UTILITIES.
source("src/ProMOTe_VB.R")               # VB_gaussian_update() baseline.
source("src/ProMOTe_Predictive.R")       # VB_gaussian_predictive_density() baseline.
source("src/ProMOTe_LTCby.R")            # probability_LTHC_by_T() baseline.
source("src/ProMOTe_LTCt.R")             # expected_LTHC_t_after_tau() baseline.
source("src/ProMOTe_utility.R")          # Helper utilities for baseline code.

# LOAD DATA (BIMODAL).
data_all <- readRDS(data_path)

# THE DATASET IS EXPECTED TO CONTAIN THE FOLLOWING FIELDS.
# THE FIELDS INCLUDE d, t, rho, tau, iota, cond_list, M, and optionally z for metrics, and for priors we expect is_bimodal, early_mean, late_mean, early_sd, late_sd, delay_mu, and delay_sd.
# FOR EVALUATION OF EARLY OR LATE WE USE delay_component AS AN N BY M MATRIX WITH VALUES 'early', 'late', 'unimodal', OR NA.
stopifnot(all(c("d","t","rho","tau","iota","M","cond_list") %in% names(data_all)))
stopifnot(all(c("is_bimodal","early_mean","late_mean","early_sd","late_sd",
                "delay_mu","delay_sd","delay_component") %in% names(data_all)))

# TRAIN AND TEST SPLIT.
n_total <- nrow(data_all$d)
train_prop <- 0.8
train_idx <- sample(seq_len(n_total), size = floor(train_prop * n_total))
test_idx  <- setdiff(seq_len(n_total), train_idx)

cat("Total patients:", n_total, "\n")
cat("Training patients:", length(train_idx), "\n")
cat("Test patients:", length(test_idx), "\n\n")

# DEFINE A HELPER TO SLICE THE DATASET BY INDEX.
subset_data <- function(data, idx) {
  list(
    d = data$d[idx, , drop = FALSE],
    t = data$t[idx, , drop = FALSE],
    rho = data$rho[idx],
    tau = data$tau[idx],
    iota = data$iota[idx],
    onset = if (!is.null(data$onset)) data$onset[idx, , drop = FALSE] else NULL,
    delay_component = data$delay_component[idx, , drop = FALSE],
    N = length(idx),
    M = data$M,
    sex = if (!is.null(data$sex)) data$sex[idx] else NULL,
    birth_conds = data$birth_conds,
    male_conds = data$male_conds,
    female_conds = data$female_conds,
    cond_list = data$cond_list,
    # COPY PRIORS THROUGH FOR CONVENIENCE.
    is_bimodal = data$is_bimodal,
    early_mean = data$early_mean,
    late_mean  = data$late_mean,
    early_sd   = data$early_sd,
    late_sd    = data$late_sd,
    delay_mu   = data$delay_mu,
    delay_sd   = data$delay_sd
  )
}

train_data <- subset_data(data_all, train_idx)
test_data  <- subset_data(data_all,  test_idx)

# SHARED CONFIGURATION.
K <- 10
epsilon <- 0.1
M <- train_data$M
cond_list <- train_data$cond_list

# BUILD CENTERED SINGLE-GAUSSIAN DELAY PRIORS PER CONDITION.
mu0 <- numeric(M)
sigma20 <- numeric(M)

for (m in seq_len(M)) {
  if (!is.na(train_data$is_bimodal[m]) && train_data$is_bimodal[m] == 1L) {
    em <- train_data$early_mean[m]
    lm <- train_data$late_mean[m]
    es <- train_data$early_sd[m]
    ls <- train_data$late_sd[m]
    # SET THE PRIOR MEAN AT THE MIDPOINT BETWEEN THE TWO MODES.
    mu0[m] <- (em + lm) / 2
    # USE THE VARIANCE OF AN EQUAL WEIGHT TWO COMPONENT MIXTURE.
    sigma20[m] <- 0.5*(es^2 + ls^2) + 0.25*(lm - em)^2
  } else {
    # USE THE UNIMODAL FALLBACK PRIOR.
    mu0[m]     <- train_data$delay_mu[m]
    sigma20[m] <- (train_data$delay_sd[m])^2
  }
}
sigma20 <- pmax(sigma20, 1e-6)

# WEAKLY INFORMATIVE GLOBAL PRIORS.
theta <- rep(1, K)
a     <- matrix(1,   M, K)
b     <- matrix(1,   M, K)
u     <- matrix(50,  M, K)
v     <- matrix(0.3, M, K)
alpha <- matrix(5,   M, K)
beta  <- matrix(750, M, K)
hyperparameters <- list(theta, a, b, u, v, alpha, beta)

# DELAY-AWARE MODEL TRAINING.
cat("TRAINING PHASE (delay-aware)\n")

N_train <- train_data$N
init_Cstar_train <- t(rdirichlet(N_train, rep(1, K))); init_Cstar_train <- t(init_Cstar_train)
init_Dstar_train <- matrix(runif(N_train * M, 0, 1),  N_train, M)
init_pstar_train <- matrix(runif(N_train * M, 0, 10), N_train, M)
init_qstar_train <- matrix(runif(N_train * M, 1, 2),  N_train, M)
init_rstar_train <- matrix(runif(N_train * M, 0.01, 0.02), N_train, M)

posterior_delay <- VB_gaussian_update(
  t_obs = train_data$t, d = train_data$d,
  rho = train_data$rho, tau = train_data$tau, iota = train_data$iota,
  hyperparameters = hyperparameters,
  initial_Cstar = init_Cstar_train, initial_Dstar = init_Dstar_train,
  initial_pstar = init_pstar_train, initial_qstar = init_qstar_train,
  initial_rstar = init_rstar_train,
  N = N_train, M = M, K = K, epsilon = epsilon,
  mu0 = mu0, sigma20 = sigma20,
  sex = train_data$sex,
  birth_conds = train_data$birth_conds,
  male_conds = train_data$male_conds,
  female_conds = train_data$female_conds,
  cond_list = cond_list
)

saveRDS(posterior_delay, file = file.path(out_dir, "posterior_delay_train.rds"))
posterior_delay <- readRDS(file.path(out_dir, "posterior_delay_train.rds"))

# COLLECT POSTERIOR FOR PREDICTION FOR THE DELAY-AWARE MODEL.
ppd <- posterior_delay$posterior.parameters
a_post_d     <- ppd$pi_a
b_post_d     <- ppd$pi_b
u_post_d     <- ppd$mu_u
v_post_d     <- ppd$v_star
alpha_post_d <- ppd$mu_alpha
beta_post_d  <- ppd$mu_beta
theta_post_d <- if (!is.null(ppd$gamma_alpha)) ppd$gamma_alpha / sum(ppd$gamma_alpha) else rep(1/K, K)
hyper_post_delay <- list(theta_post_d, a_post_d, b_post_d, u_post_d, v_post_d, alpha_post_d, beta_post_d)

# NO-DELAY BASELINE TRAINING.
cat("TRAINING PHASE (no-delay)\n")

init_Cstar_train <- t(rdirichlet(N_train, rep(1, K))); init_Cstar_train <- t(init_Cstar_train)
init_Dstar_train <- matrix(runif(N_train * M, 0, 1),  N_train, M)
init_pstar_train <- matrix(runif(N_train * M, 0, 10), N_train, M)
init_qstar_train <- matrix(runif(N_train * M, 1, 2),  N_train, M)
init_rstar_train <- matrix(runif(N_train * M, 0.01, 0.02), N_train, M)

posterior_base <- VB_gaussian_update(
  d = train_data$d, t = train_data$t, rho = train_data$rho, tau = train_data$tau, iota = train_data$iota,
  hyperparameters = hyperparameters,
  initial_Cstar = init_Cstar_train, initial_Dstar = init_Dstar_train,
  initial_pstar = init_pstar_train, initial_qstar = init_qstar_train,
  initial_rstar = init_rstar_train,
  N = N_train, M = M, K = K, epsilon = epsilon,
  sex = train_data$sex,
  birth_conds = train_data$birth_conds,
  male_conds = train_data$male_conds,
  female_conds = train_data$female_conds,
  cond_list = cond_list
)

saveRDS(posterior_base, file = file.path(out_dir, "posterior_nodelay_train.rds"))
posterior_base <- readRDS(file.path(out_dir, "posterior_nodelay_train.rds"))

# COLLECT POSTERIOR FOR PREDICTION FOR THE BASELINE MODEL.
ppb <- posterior_base$posterior.parameters
theta_post_b <- if (!is.null(ppb$theta_star)) ppb$theta_star / sum(ppb$theta_star) else rep(1/K, K)
a_post_b     <- ppb$a_star
b_post_b     <- ppb$b_star
u_post_b     <- ppb$u_star
v_post_b     <- ppb$v_star
alpha_post_b <- ppb$alpha_star
beta_post_b  <- ppb$beta_star
hyper_post_base <- list(theta_post_b, a_post_b, b_post_b, u_post_b, v_post_b, alpha_post_b, beta_post_b)

# TEST CLUSTER RECOVERY ON THE FULL WINDOW.
cat("TEST (full window): cluster recovery\n")

# DEFINE A HELPER TO COMPUTE PHI MATRICES.
get_phi_matrix <- function(d_mat, t_mat, rho_vec, tau_vec, M, hyper, delay=FALSE, mu0=NULL, sigma20=NULL) {
  N <- nrow(d_mat)
  phi_mat <- matrix(NA_real_, N, K)
  for (i in 1:N) {
    M_obs  <- which(d_mat[i,] == 1 & !is.na(t_mat[i,]) & t_mat[i,] >= rho_vec[i] & t_mat[i,] <= tau_vec[i])
    M_part <- which(d_mat[i,] == 1 & !is.na(t_mat[i,]) & t_mat[i,] <  rho_vec[i])
    all_idx <- seq_len(M)
    M_unobs <- setdiff(all_idx, union(M_obs, M_part))
    if (delay) {
      pred <- VB_gaussian_predictive_density(
        hyperparameters = hyper,
        M_obs  = M_obs, M_part = M_part, M_unobs = M_unobs,
        d_obs  = rep.int(1L, length(M_obs)),
        t_obs  = if (length(M_obs)) t_mat[i, M_obs] else numeric(0),
        d_part = rep.int(1L, length(M_part)),
        rho    = rho_vec[i],
        tau    = tau_vec[i],
        M      = M,
        mu0    = mu0,
        sigma20= sigma20
      )
    } else {
      pred <- VB_gaussian_predictive_density(
        hyperparameters = hyper,
        M_obs  = M_obs, M_part = M_part, M_unobs = M_unobs,
        d_obs  = rep.int(1L, length(M_obs)),
        t_obs  = if (length(M_obs)) t_mat[i, M_obs] else numeric(0),
        d_part = rep.int(1L, length(M_part)),
        rho    = rho_vec[i],
        tau    = tau_vec[i],
        M      = M
      )
    }
    phi_mat[i, ] <- pred$phi
  }
  phi_mat
}

N_test <- test_data$N

phi_delay <- get_phi_matrix(
  d_mat = test_data$d, t_mat = test_data$t,
  rho_vec = test_data$rho, tau_vec = test_data$tau,
  M = M, hyper = hyper_post_delay, delay = TRUE, mu0 = mu0, sigma20 = sigma20
)
phi_base <- get_phi_matrix(
  d_mat = test_data$d, t_mat = test_data$t,
  rho_vec = test_data$rho, tau_vec = test_data$tau,
  M = M, hyper = hyper_post_base, delay = FALSE
)

raw_pred_delay <- max.col(phi_delay, ties.method = "first")
raw_pred_base  <- max.col(phi_base,  ties.method = "first")

if (!is.null(data_all$z)) {
  true_clusters_test <- data_all$z[test_idx]
  K_used <- max(K, max(raw_pred_delay), max(raw_pred_base), max(true_clusters_test))
  # ALIGNMENT IS OPTIONAL FOR REPORTING BECAUSE THE INDEPENDENCE TEST DOES NOT NEED IT.
  tab_d <- table(factor(raw_pred_delay, levels = 1:K_used),
                 factor(true_clusters_test, levels = 1:K_used))
  map_d <- as.integer(clue::solve_LSAP(tab_d, maximum = TRUE))
  aligned_pred_delay <- map_d[raw_pred_delay]

  tab_b <- table(factor(raw_pred_base, levels = 1:K_used),
                 factor(true_clusters_test, levels = 1:K_used))
  map_b <- as.integer(clue::solve_LSAP(tab_b, maximum = TRUE))
  aligned_pred_base <- map_b[raw_pred_base]

  acc_d <- mean(aligned_pred_delay == true_clusters_test)
  acc_b <- mean(aligned_pred_base  == true_clusters_test)
  ari_d <- mclust::adjustedRandIndex(aligned_pred_delay, true_clusters_test)
  ari_b <- mclust::adjustedRandIndex(aligned_pred_base,  true_clusters_test)
  nmi_d <- aricode::NMI(aligned_pred_delay, true_clusters_test)
  nmi_b <- aricode::NMI(aligned_pred_base,  true_clusters_test)

  cat("Delay-aware  (aligned): Acc=", round(acc_d*100,2), "%, ARI=", round(ari_d,4),
      ", NMI=", round(nmi_d,4), "\n", sep = "")
  cat("No-delay base(aligned): Acc=", round(acc_b*100,2), "%, ARI=", round(ari_b,4),
      ", NMI=", round(nmi_b,4), "\n\n", sep = "")
}

# TEST FORWARD PREDICTION FROM A RANDOM CUT.
cat("TEST: forward prediction from random cut\n")
set.seed(42)
cut_ages <- runif(N_test, min = 50, max = 90)
cut_mat  <- matrix(cut_ages, nrow = N_test, ncol = M)

# BUILD PREDICTIVE OBJECTS USING ONLY PRE CUT EVIDENCE.
pred_list_pre_delay <- vector("list", N_test)
pred_list_pre_base  <- vector("list", N_test)

phi_pre_delay <- matrix(NA_real_, N_test, K)
phi_pre_base  <- matrix(NA_real_, N_test, K)

for (i in 1:N_test) {
  cut_i <- cut_ages[i]
  # BUILD THE VIEW UP TO THE CUT AGE.
  M_obs  <- which(test_data$d[i,] == 1 & !is.na(test_data$t[i,]) &
                    test_data$t[i,] >= test_data$rho[i] &
                    test_data$t[i,] <= cut_i)
  M_part <- which(test_data$d[i,] == 1 & !is.na(test_data$t[i,]) &
                    test_data$t[i,] < test_data$rho[i])
  all_idx <- seq_len(M)
  M_unobs <- setdiff(all_idx, union(M_obs, M_part))

  # COMPUTE THE DELAY AWARE PREDICTIVE OBJECT.
  pd <- VB_gaussian_predictive_density(
    hyperparameters = hyper_post_delay,
    M_obs  = M_obs, M_part = M_part, M_unobs = M_unobs,
    d_obs  = rep.int(1L, length(M_obs)),
    t_obs  = if (length(M_obs)) test_data$t[i, M_obs] else numeric(0),
    d_part = rep.int(1L, length(M_part)),
    rho    = test_data$rho[i],
    tau    = cut_i,
    M      = M,
    mu0    = mu0,
    sigma20= sigma20
  )
  phi_pre_delay[i, ] <- pd$phi
  pred_list_pre_delay[[i]] <- pd

  # COMPUTE THE BASELINE PREDICTIVE OBJECT.
  pb <- VB_gaussian_predictive_density(
    hyperparameters = hyper_post_base,
    M_obs  = M_obs, M_part = M_part, M_unobs = M_unobs,
    d_obs  = rep.int(1L, length(M_obs)),
    t_obs  = if (length(M_obs)) test_data$t[i, M_obs] else numeric(0),
    d_part = rep.int(1L, length(M_part)),
    rho    = test_data$rho[i],
    tau    = cut_i,
    M      = M
  )
  phi_pre_base[i, ] <- pb$phi
  pred_list_pre_base[[i]] <- pb
}

# COMPUTE AFTER CUT PROBABILITIES AND EXPECTED TIMES.
P_after_delay   <- matrix(0, N_test, M)
E_t_after_delay <- matrix(NA_real_, N_test, M)

P_after_base   <- matrix(0, N_test, M)
E_t_after_base <- matrix(NA_real_, N_test, M)

for (i in 1:N_test) {
  cut_i <- cut_ages[i]
  T_i   <- test_data$tau[i]
  pd    <- pred_list_pre_delay[[i]]
  pb    <- pred_list_pre_base[[i]]

  P_after_delay[i, ] <- probability_LTHC_by_T(
    parameters = pd, hyperparameters = hyper_post_delay,
    T = T_i, tau = cut_i, M = M, mu0 = mu0, sigma20 = sigma20
  )
  E_t_after_delay[i, ] <- expected_LTHC_t_after_tau(
    parameters = pd, hyperparameters = hyper_post_delay,
    tau = cut_i, M = M, mu0 = mu0, sigma20 = sigma20
  )

  P_after_base[i, ] <- probability_LTHC_by_T(
    pb, hyper_post_base, T = T_i, tau = cut_i, M = M
  )
  E_t_after_base[i, ] <- expected_LTHC_t_after_tau(
    pb, hyper_post_base, tau = cut_i, M = M
  )
}

# BUILD GROUND TRUTH MASKS FOR AFTER CUT EVALUATION.
d_true   <- test_data$d
t_true   <- test_data$t
tau_true <- test_data$tau

is_pos <- (d_true == 1) & (t_true > cut_mat) & (t_true <= tau_true)
is_neg <- (d_true == 0) | ((d_true == 1) & (t_true <= cut_mat))
is_unk <- (d_true == 1) & (t_true > tau_true)
eval_mask <- (is_pos | is_neg) & !is_unk

# COMPUTE PRESENCE METRICS OVERALL.
y_true <- as.integer(is_pos[eval_mask])
y_prob_delay <- as.numeric(P_after_delay[eval_mask])
y_prob_base  <- as.numeric(P_after_base[eval_mask])

y_pred_delay <- as.integer(y_prob_delay >= 0.5)
y_pred_base  <- as.integer(y_prob_base  >= 0.5)

acc_after_delay <- mean(y_pred_delay == y_true)
acc_after_base  <- mean(y_pred_base  == y_true)

roc_after_delay <- if (length(unique(y_true)) > 1) {
  as.numeric(pROC::auc(pROC::roc(y_true, y_prob_delay, quiet = TRUE)))
} else NA_real_
roc_after_base <- if (length(unique(y_true)) > 1) {
  as.numeric(pROC::auc(pROC::roc(y_true, y_prob_base, quiet = TRUE)))
} else NA_real_

cat("FORWARD PREDICTION (presence):\n")
cat("  Delay-aware  Acc=", round(acc_after_delay*100,2), "%, AUROC=",
    ifelse(is.na(roc_after_delay), "NA", round(roc_after_delay,4)), "\n", sep = "")
cat("  Baseline     Acc=", round(acc_after_base*100,2), "%, AUROC=",
    ifelse(is.na(roc_after_base), "NA", round(roc_after_base,4)), "\n\n", sep = "")

# COMPUTE MAE OVER OBSERVED AFTER CUT EVENTS.
mae_mask <- is_pos & !is.na(E_t_after_delay) & !is.na(E_t_after_base)
mae_vals_delay <- abs(E_t_after_delay[mae_mask] - t_true[mae_mask])
mae_vals_base  <- abs(E_t_after_base[mae_mask]  - t_true[mae_mask])
mae_after_delay <- if (length(mae_vals_delay) > 0) mean(mae_vals_delay) else NA_real_
mae_after_base  <- if (length(mae_vals_base)  > 0) mean(mae_vals_base)  else NA_real_

cat("FORWARD PREDICTION (diagnosis age MAE, observed after-cut):\n")
cat("  Delay-aware  MAE=", round(mae_after_delay,3), " years\n", sep = "")
cat("  Baseline     MAE=", round(mae_after_base,3),  " years\n\n", sep = "")

# CONDITION WISE MAE FOR LATE PATIENTS AND CHI SQUARED EARLY VERSUS LATE SEPARATION.
# DEFINE A HELPER TO COMPUTE CONDITION WISE MAE AMONG LATE OBSERVATIONS FOR AFTER CUT EVENTS.
get_mae_by_condition_late <- function(is_late, mae_mask_mat, err_mat) {
  out <- numeric(M); out[] <- NA_real_
  for (m in seq_len(M)) {
    idx <- which(is_late[, m] & mae_mask_mat[, m])
    if (length(idx) > 0) {
      out[m] <- mean(err_mat[cbind(idx, rep(m, length(idx)))])
    }
  }
  out
}

# IDENTIFY LATE AND EARLY LABELS PER CONDITION FROM GENERATING LABELS.
is_late <- (test_data$delay_component == "late")
is_early <- (test_data$delay_component == "early")

# BUILD THE MAE MASK PER CONDITION FOR OBSERVED AFTER CUT EVENTS.
mae_mask_mat <- is_pos & !is.na(E_t_after_delay) & !is.na(E_t_after_base)

err_delay <- abs(E_t_after_delay - t_true)
err_base  <- abs(E_t_after_base  - t_true)

mae_delay_late <- get_mae_by_condition_late(is_late, mae_mask_mat, err_delay)
mae_base_late  <- get_mae_by_condition_late(is_late, mae_mask_mat, err_base)
mae_diff_late  <- mae_delay_late - mae_base_late

# RUN A CHI SQUARED TEST OF INDEPENDENCE BETWEEN EARLY OR LATE AND PREDICTED CLUSTERS FOR EACH CONDITION.
# WE USE RAW PREDICTED CLUSTERS WITHOUT ALIGNMENT BECAUSE INDEPENDENCE TESTING DOES NOT REQUIRE ALIGNMENT.
chi_pvals <- function(group_matrix, pred_clusters, is_bimodal, cond_names) {
  res <- tibble(condition = cond_names, p_value = NA_real_)
  for (m in which(is_bimodal == 1L)) {
    grp <- group_matrix[, m]
    valid <- (grp == "early" | grp == "late") & !is.na(grp)
    if (sum(valid) < 10) next
    tab <- table(grp[valid], pred_clusters[valid])
    tab <- tab[, colSums(tab) > 0, drop = FALSE]
    if (ncol(tab) < 2) next
    suppressWarnings({
      p <- tryCatch(chisq.test(tab)$p.value, error = function(e) NA_real_)
    })
    res$p_value[m] <- p
  }
  res
}

chi_delay <- chi_pvals(test_data$delay_component, raw_pred_delay, data_all$is_bimodal, cond_list)
chi_base  <- chi_pvals(test_data$delay_component, raw_pred_base,  data_all$is_bimodal, cond_list)

# ASSEMBLE THE SUMMARY TABLE.
summary_stats <- tibble(
  condition = cond_list,
  mae_delay_late = mae_delay_late,
  mae_base_late  = mae_base_late,
  mae_diff_late  = mae_diff_late,
  p_base  = chi_base$p_value,
  p_delay = chi_delay$p_value
) %>%
  mutate(
    logp_base  = -log10(pmax(p_base,  1e-300)),
    logp_delay = -log10(pmax(p_delay, 1e-300)),
    is_bimodal = data_all$is_bimodal
  )

# SAVE NUMERIC RESULTS.
write.csv(summary_stats, file = file.path(out_dir, "summary_stats_bimodal.csv"), row.names = FALSE)

# VISUALIZATION PLOTS.
# DEFINE A SIMPLE COLOR SCHEME.
light_blue <- "#9ecae1"
light_pink <- "#fbb4b9"

# PLOT THE MAE DIFFERENCE BARPLOT FOR LATE PATIENTS FOR BIMODAL CONDITIONS ONLY.
p1 <- summary_stats %>%
  filter(is_bimodal == 1L, !is.na(mae_diff_late)) %>%
  ggplot(aes(x = reorder(condition, mae_diff_late), y = mae_diff_late)) +
  geom_bar(stat = "identity", fill = light_blue) +
  coord_flip() +
  labs(
    title = "MAE Increase from Baseline to Delay-Aware (Late Patients, after-cut)",
    x = "Condition", y = "MAE Difference (delay - base, years)"
  ) +
  theme_classic(base_size = 12)

ggsave(file.path(out_dir, "mae_diff_late_barplot.png"), p1, width = 9, height = 7, dpi = 300)

# PLOT THE CHI SQUARED NEGATIVE LOG TEN P BARPLOT FOR EARLY VERSUS LATE CLUSTER SEPARATION.
df_long_chi <- summary_stats %>%
  filter(is_bimodal == 1L, !is.na(logp_base) | !is.na(logp_delay)) %>%
  select(condition, logp_base, logp_delay) %>%
  pivot_longer(cols = c("logp_base", "logp_delay"),
               names_to = "model", values_to = "logp") %>%
  mutate(model = recode(model, logp_base = "Baseline", logp_delay = "Delay-Aware"))

p2 <- ggplot(df_long_chi, aes(x = reorder(condition, logp), y = logp, fill = model)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed") +
  geom_hline(yintercept = -log10(0.01), linetype = "dashed") +
  annotate("text", x = 1, y = -log10(0.05) + 0.1, label = "p = 0.05", hjust = 0, size = 3) +
  annotate("text", x = 1, y = -log10(0.01) + 0.1, label = "p = 0.01", hjust = 0, size = 3) +
  coord_flip() +
  scale_fill_manual(values = c("Baseline" = light_pink, "Delay-Aware" = light_blue)) +
  labs(title = "-log10(p): Early vs Late Cluster Separation (Chi-squared)",
       x = "Condition", y = "-log10(p-value)") +
  theme_classic(base_size = 12)

ggsave(file.path(out_dir, "chi2_logp_barplot.png"), p2, width = 9, height = 7, dpi = 300)

cat("Saved:\n  ", file.path(out_dir, "summary_stats_bimodal.csv"), "\n  ",
    file.path(out_dir, "mae_diff_late_barplot.png"), "\n  ",
    file.path(out_dir, "chi2_logp_barplot.png"), "\n", sep = "")
