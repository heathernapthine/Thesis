set.seed(4)

library(extraDistr) 
library(MASS)
library(truncnorm)

# Parameters.
# Number of individuals and clusters, and full list of condition names.
N <- 200000  # number of individuals
K <- 10   # number of clusters
cond_list <- c(
  "Allergic and chronic rhinitis", "Asbestosis", "Asthma", "Atrial fibrillation",
  "Bronchiectasis", "Cardiomyopathy", "Chronic liver disease", "Chronic renal disease", 
  "Coeliac disease", "Conduction disorders and other arrhythmias", "Coronary heart disease",
  "Cystic Fibrosis", "Diabetes NOS", "Diverticular disease of intestine", "Erectile dysfunction", 
  "Fatty Liver", "Gastro-oesophageal reflux, gastritis and similar", "Gout", "Heart failure", 
  "Heart valve disorders", "Hyperplasia of prostate", "Hypertension", "Hypo or hyperthyroidism", 
  "Inflammatory arthritis and other inflammatory conditions", "Inflammatory bowel disease", 
  "Irritable bowel syndrome", "Non-acute cystitis", "Osteoporosis and vertebral crush fractures", 
  "Peptic ulcer disease", "Peripheral arterial disease", "Primary pulmonary hypertension", 
  "Sleep apnoea", "Spinal stenosis", "Stroke", "Transient ischaemic attack", "Type 1 diabetes", 
  "Type 2 diabetes", "Urinary Incontinence", "Addisons disease", "Alcohol Problems", 
  "Anorexia and bulimia nervosa", "Anxiety disorders", "Autism and Asperger’s syndrome", 
  "Benign neoplasm of brain and other parts of CNS", "Bipolar affective disorder and mania", 
  "Cerebral Palsy", "Dementia", "Down’s syndrome", "Epilepsy", "Hearing loss", 
  "HIV", "Immunodeficiencies", "Intellectual disability", 
  "Iron and vitamin deficiency anaemia conditions", "Macular degeneration", "Migraine", 
  "Meniere disease", "Motor neuron disease", "Multiple sclerosis", "Myasthenia gravis", 
  "Non-melanoma skin malignancies", "Obsessive-compulsive disorder", 
  "Other psychoactive substance misuse", "Parkinson’s disease", "Peripheral or autonomic neuropathy", 
  "Post-traumatic stress disorder", "Postviral fatigue syndrome, neurasthenia and fibromyalgia", 
  "Psoriasis", "Sarcoidosis", "Schizophrenia, schizotypal and delusional disorders", 
  "Sickel-cell anaemia", "Solid organ malignancies", "Thalassaemia", 
  "Tuberculosis", "Visual impairment and blindness"
)

# Number of conditions.
M <- length(cond_list)

# HELPER.
# Map condition names to indices.
idx_of <- function(names_vec) {
  idx <- match(names_vec, cond_list)
  if (anyNA(idx)) {
    missing <- names_vec[is.na(idx)]
    warning("These condition names were not found in cond_list: ",
            paste(missing, collapse = ", "))
  }
  idx[!is.na(idx)]
}

# TRUE DELAY PATTERNS PER CONDITION.
# Start with a pool of all conditions to form groups.
pool <- cond_list

# Helper to sample non overlapping groups from the pool.
pick_group <- function(pool, n) {
  chosen <- sample(pool, size = n, replace = FALSE)
  pool <- setdiff(pool, chosen)
  list(group = chosen, pool = pool)
}

# Sample six groups of five conditions to assign delay families.
res <- pick_group(pool, 5);  grp_uniform_asymptomatic <- res$group; pool <- res$pool
res <- pick_group(pool, 5);  grp_early_narrow         <- res$group; pool <- res$pool
res <- pick_group(pool, 5);  grp_late_wide            <- res$group; pool <- res$pool
res <- pick_group(pool, 5);  grp_psoriasis_medium     <- res$group; pool <- res$pool
res <- pick_group(pool, 5);  grp_mix_two_modes        <- res$group; pool <- res$pool
res <- pick_group(pool, 5);  grp_near_zero            <- res$group; pool <- res$pool

# INITIALISE DELAY FAMILY LABELS AND PARAMETERS.
# Default families and parameters before applying group specific overrides.
delay_dist_cond <- rep("gaussian", M)   # "gaussian", "uniform", "mixture2"
delay_group <- rep("gaussian_other", M)
delay_mu_cond <- rep(5, M)
delay_sd_cond <- rep(1.5, M)
unif_a <- rep(NA_real_, M)
unif_b <- rep(NA_real_, M)
mix_w1  <- rep(NA_real_, M)
mix_mu1 <- rep(NA_real_, M)
mix_sd1 <- rep(NA_real_, M)
mix_mu2 <- rep(NA_real_, M)
mix_sd2 <- rep(NA_real_, M)

# APPLY GROUP SPECIFIC DELAY SETTINGS.
# Near zero delays for selected conditions.
ix <- idx_of(grp_near_zero)
delay_dist_cond[ix] <- "gaussian"
delay_mu_cond[ix]   <- 0.05
delay_sd_cond[ix]   <- 0.05
delay_group[ix]     <- "gaussian_near_zero"

# Uniform broad delays for asymptomatic like conditions.
ix <- idx_of(grp_uniform_asymptomatic)
delay_dist_cond[ix] <- "uniform"
unif_a[ix] <- 0
unif_b[ix] <- 15
delay_mu_cond[ix] <- NA_real_
delay_sd_cond[ix] <- NA_real_
delay_group[ix]   <- "uniform"

# Early tight gaussian delays.
ix <- idx_of(grp_early_narrow)
delay_dist_cond[ix] <- "gaussian"
delay_mu_cond[ix]   <- 3
delay_sd_cond[ix]   <- 0.75
delay_group[ix]     <- "gaussian_early"

# Late wide gaussian delays.
ix <- idx_of(grp_late_wide)
delay_dist_cond[ix] <- "gaussian"
delay_mu_cond[ix]   <- 8
delay_sd_cond[ix]   <- 3
delay_group[ix]     <- "gaussian_late"

# Medium gaussian delays.
ix <- idx_of(grp_psoriasis_medium)
delay_dist_cond[ix] <- "gaussian"
delay_mu_cond[ix]   <- 5
delay_sd_cond[ix]   <- 1.5
delay_group[ix]     <- "gaussian_medium"

# Two mode mixture delays.
ix <- idx_of(grp_mix_two_modes)
delay_dist_cond[ix] <- "mixture2"
mix_w1[ix]   <- 0.5
mix_mu1[ix]  <- 1.5
mix_sd1[ix]  <- 0.5
mix_mu2[ix]  <- 7.0
mix_sd2[ix]  <- 2.0
delay_mu_cond[ix] <- NA_real_
delay_sd_cond[ix] <- NA_real_
delay_group[ix]   <- "mixture2"

# CLUSTER ASSIGNMENT.
# Draw cluster labels using random mixture proportions.
cluster_sizes <- runif(K, 0.03, 0.15)
cluster_sizes <- cluster_sizes / sum(cluster_sizes)
z <- sample(1:K, N, replace = TRUE, prob = cluster_sizes)

# DISEASE PRESENCE MODEL.
# Presence is completely independent of clusters (uniform across clusters).
pi_m  <- rbeta(M, 0.07, 0.49)              # per-condition baseline prevalence
pi_mk <- matrix(pi_m, nrow = M, ncol = K)  # same value for all clusters

# Sample binary presence per person and condition.
d <- matrix(0, N, M)
for (n in 1:N) {
  d[n, ] <- rbinom(M, 1, prob = pi_mk[, z[n]])  
}

# # DISEASE PRESENCE MODEL.
# # Presence is weakly informative and near independent of clusters.
# pi_m <- rbeta(M, 0.07, 0.49)
# pi_mk <- matrix(rep(pi_m, each = K), nrow = M, ncol = K)
# epsilon <- matrix(rnorm(M * K, 0, 0.01), nrow = M, ncol = K)
# pi_mk <- pmin(pmax(pi_mk + epsilon, 0.0001), 0.9999)

# # Sample binary presence per person and condition.
# d <- matrix(0, N, M)
# for (n in 1:N) {
#   d[n, ] <- rbinom(M, 1, prob = pi_mk[, z[n]])
# }

# CLUSTER SPECIFIC ONSET PARAMETERS WITH LESS SEPARATION.
# Build means per cluster on a narrow grid and draw variances.
mu_mk <- matrix(NA, M, K)
sigma2_mk <- matrix(NA, M, K)

# Create overlapping cluster mean grid.
cluster_mean_grid <- seq(30, 80, length.out = K)

for (m in 1:M) {
  for (k in 1:K) {
    sigma2 <- extraDistr::rinvgamma(1, alpha = 4, beta = 150)

    mu <- rnorm(1, mean = cluster_mean_grid[k], sd = 4)  # A bit wider around each mean.
    mu_mk[m, k] <- mu
    sigma2_mk[m, k] <- sigma2
  }
}

# CONDITION SPECIFIC DIAGNOSIS DELAYS.
# Draw a per condition delay from the assigned family.
delay_mu <- truncnorm::rtruncnorm(M, a = 0, mean = 6, sd = 3)  # Bigger mean and spread.
delay_sd <- runif(M, 2.0, 4.0)

# GENERATE ONSET DELAY AND DIAGNOSIS TIMES.
# For each present condition simulate onset time and add delay to get diagnosis time.
onset <- matrix(0, N, M)
delay <- matrix(0, N, M)
t <- matrix(0, N, M)
for (n in 1:N) {
  for (m in 1:M) {
    if (d[n, m] == 1) {
      mu    <- mu_mk[m, z[n]]
      sigma <- sqrt(sigma2_mk[m, z[n]])

      # Sample onset given the cluster.
      onset[n, m] <- max(rnorm(1, mean = mu, sd = sigma), 0)

         distm <- delay_dist_cond[m]
      if (distm == "gaussian") {
        delay[n, m] <- truncnorm::rtruncnorm(1, a = 0,
                                             mean = delay_mu_cond[m],
                                             sd   = delay_sd_cond[m])
      } else if (distm == "uniform") {
        delay[n, m] <- runif(1, unif_a[m], unif_b[m])
      } else if (distm == "mixture2") {
        comp <- rbinom(1, 1, prob = mix_w1[m])
        if (comp == 1) {
          delay[n, m] <- truncnorm::rtruncnorm(1, a = 0, mean = mix_mu1[m], sd = mix_sd1[m])
        } else {
          delay[n, m] <- truncnorm::rtruncnorm(1, a = 0, mean = mix_mu2[m], sd = mix_sd2[m])
        }
      } else {
        stop("Unknown delay_dist_cond: ", distm)
      }

      # Compute diagnosis time as onset plus delay.
      t[n, m] <- onset[n, m] + delay[n, m]
    }
  }
}

# INDIVIDUAL CENSORING.
# Simulate study entry age and follow up window plus death indicator.
rho <- runif(N, 20, 60)        # baseline age
tau <- rho + 30                # current age
iota <- rbinom(N, 1, 0.8)      # 1 = dead, 0 = right-censored

# BUILD DELAY PRIOR DATA FRAME FOR DOWNSTREAM INFERENCE.
delay_prior_df <- data.frame(
  condition = cond_list,
  delay_dist = delay_dist_cond,  # families used by simulator/inference
  delay_group = factor(delay_group, levels = c(
    "gaussian_near_zero","gaussian_early","gaussian_late","gaussian_medium",
    "gaussian_other","uniform","mixture2"
  )),
  delay_mu   = delay_mu_cond,    # for gaussian; NA otherwise
  delay_sd   = delay_sd_cond,    # for gaussian; NA otherwise
  unif_a     = unif_a,           # for uniform
  unif_b     = unif_b,
  mix_w1     = mix_w1,           # for mixture2
  mix_mu1    = mix_mu1,
  mix_sd1    = mix_sd1,
  mix_mu2    = mix_mu2,
  mix_sd2    = mix_sd2,
  stringsAsFactors = FALSE
)

# SAVE GROUP MEMBERS FOR REFERENCE.
prior_groups <- list(
  uniform_asymptomatic = grp_uniform_asymptomatic,
  early_narrow         = grp_early_narrow,
  late_wide            = grp_late_wide,
  psoriasis_medium     = grp_psoriasis_medium,
  mix_two_modes        = grp_mix_two_modes,
  near_zero            = grp_near_zero
)

# SAVE DATASET.

data_list <- list(
  z = z, d = d, t = t, onset = onset, delay = delay,
  rho = rho, tau = tau, iota = iota,
  N = N, K = K, M = M, cond_list = cond_list,
  pi_mk = pi_mk, mu_mk = mu_mk, sigma2_mk = sigma2_mk,
  delay_dist_cond = delay_dist_cond,
  delay_mu_cond   = delay_mu_cond,
  delay_sd_cond   = delay_sd_cond,
  delay_prior_df  = delay_prior_df,
  prior_groups    = prior_groups
)

# Print shapes of key items for a quick check.
cat("Shapes of key items in data_list:\n")
for (name in c("z","d","t","onset","delay","rho","tau","iota",
               "pi_mk","mu_mk","sigma2_mk","delay_prior_df")) {
  item <- data_list[[name]]
  item_shape <- if (!is.null(dim(item))) {
    paste0("(", paste(dim(item), collapse = ", "), ")")
  } else if (is.data.frame(item)) {
    paste0("data.frame: (", nrow(item), ", ", ncol(item), ")")
  } else {
    paste0("Length: ", length(item))
  }
  cat(sprintf("%-18s : %s\n", name, item_shape))
}

# Save the generated dataset to an RDS file.
saveRDS(data_list, "data/generated_promote_style_mixed_delays.rds")
