set.seed(4)

library(extraDistr) 
library(MASS)
library(truncnorm)

# Parameters.
N <- 200000  # number of individuals
K <- 10      # number of clusters
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

# Cluster assignment.
cluster_sizes <- runif(K, 0.03, 0.15)
cluster_sizes <- cluster_sizes / sum(cluster_sizes)
z <- sample(1:K, N, replace = TRUE, prob = cluster_sizes)

# Disease presence is uninformative for clustering.
pi_m <- rbeta(M, 0.07, 0.49)
pi_mk <- matrix(rep(pi_m, each = K), nrow = M, ncol = K)
epsilon <- matrix(rnorm(M * K, 0, 0.01), nrow = M, ncol = K)
pi_mk <- pmin(pmax(pi_mk + epsilon, 0.0001), 0.9999)

# Sample condition presence per individual and condition.
d <- matrix(0L, N, M)
for (n in 1:N) {
  d[n, ] <- rbinom(M, 1, prob = pi_mk[, z[n]])
}

# Cluster specific onset parameters with tighter separation.
mu_mk <- matrix(NA_real_, M, K)
sigma2_mk <- matrix(NA_real_, M, K)

# Build a grid of cluster means to induce overlap.
cluster_mean_grid <- seq(50, 70, length.out = K)

for (m in 1:M) {
  for (k in 1:K) {
    sigma2 <- extraDistr::rinvgamma(1, alpha = 4, beta = 150)
    mu <- rnorm(1, mean = cluster_mean_grid[k], sd = 4)  
    mu_mk[m, k] <- mu
    sigma2_mk[m, k] <- sigma2
  }
}

# Condition specific diagnosis delays with overlapping bimodals.
delay_mu_base <- truncnorm::rtruncnorm(M, a = 0, mean = 6, sd = 3)
delay_sd_base <- runif(M, 2.0, 4.0)

# Choose which conditions are bimodal.
bimodal_frac <- 0.5
is_bimodal <- as.integer(runif(M) < bimodal_frac)

# Define per condition mixture weights and overlapping component parameters.
mix_weight_early <- rep(NA_real_, M)  # Probability of early component.
early_mean <- rep(NA_real_, M)
late_mean  <- rep(NA_real_, M)
early_sd   <- rep(NA_real_, M)
late_sd    <- rep(NA_real_, M)

for (m in 1:M) {
  if (is_bimodal[m] == 1L) {
    mix_weight_early[m] <- rbeta(1, 5, 5)
    delta <- runif(1, 2.0, 4.0)
    early_mean[m] <- max(0, delay_mu_base[m] - delta)
    late_mean[m]  <- delay_mu_base[m] + delta
    s_early <- runif(1, 2.5, 4.5)
    s_late  <- runif(1, 2.5, 4.5)
    early_sd[m] <- s_early
    late_sd[m]  <- s_late
  } else {
    mix_weight_early[m] <- NA_real_
    early_mean[m] <- NA_real_
    late_mean[m]  <- NA_real_
    early_sd[m]   <- NA_real_
    late_sd[m]    <- NA_real_
  }
}

# Generate onset delay and diagnosis times.
onset <- matrix(0, N, M)
delay <- matrix(0, N, M)
t <- matrix(0, N, M)

# Record the generating component per observation.
delay_component <- matrix(NA_character_, N, M)

for (n in 1:N) {
  zn <- z[n]
  for (m in 1:M) {
    if (d[n, m] == 1L) {
      mu    <- mu_mk[m, zn]
      sigma <- sqrt(sigma2_mk[m, zn])
      onset[n, m] <- max(rnorm(1, mean = mu, sd = sigma), 0)
      if (is_bimodal[m] == 1L) {
        is_early <- rbinom(1, 1, mix_weight_early[m]) == 1
        if (is_early) {
          delay[n, m] <- truncnorm::rtruncnorm(1, a = 0,
                                               mean = early_mean[m],
                                               sd   = early_sd[m])
          delay_component[n, m] <- "early"
        } else {
          delay[n, m] <- truncnorm::rtruncnorm(1, a = 0,
                                               mean = late_mean[m],
                                               sd   = late_sd[m])
          delay_component[n, m] <- "late"
        }
      } else {
        delay[n, m] <- truncnorm::rtruncnorm(1, a = 0,
                                             mean = delay_mu_base[m],
                                             sd   = delay_sd_base[m])
        delay_component[n, m] <- "unimodal"
      }
      t[n, m] <- onset[n, m] + delay[n, m]
    } else {
      delay_component[n, m] <- NA_character_
    }
  }
}

# Individual censoring.
rho <- runif(N, 20, 60)        # Baseline age.
tau <- rho + 30                # Current age.
iota <- rbinom(N, 1, 0.8)      # One equals dead and zero equals right censored.

data_list <- list(
  # Core data.
  z = z,
  d = d,
  t = t,
  onset = onset,
  delay = delay,
  rho = rho,
  tau = tau,
  iota = iota,
  N = N,
  K = K,
  M = M,
  cond_list = cond_list,
  pi_mk = pi_mk,
  mu_mk = mu_mk,
  sigma2_mk = sigma2_mk,
  # Delay params for unimodal baseline.
  delay_mu = delay_mu_base,
  delay_sd = delay_sd_base,
  # Bimodal configuration per condition.
  is_bimodal = is_bimodal,
  mix_weight_early = mix_weight_early,
  early_mean = early_mean,
  late_mean  = late_mean,
  early_sd   = early_sd,
  late_sd    = late_sd,
  # Per observation component label.
  delay_component = delay_component
)

# Print shapes for checking.
cat("Shapes of all items in data_list:\n")
for (name in names(data_list)) {
  item <- data_list[[name]]
  item_shape <- if (!is.null(dim(item))) {
    paste0("(", paste(dim(item), collapse = ", "), ")")
  } else {
    paste0("Length: ", length(item))
  }
  cat(sprintf("%-20s : %s\n", name, item_shape))
}

# Write the dataset to an RDS file.
saveRDS(data_list, "data/generated_promote_style_bimodal_delays.rds")
