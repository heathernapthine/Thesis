set.seed(4)

library(extraDistr) 
library(MASS)
library(truncnorm)

# PARAMETERS.
# Define population size, cluster count, and condition names.
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

# Derive the number of conditions.
M <- length(cond_list)

# CLUSTER ASSIGNMENT.
# Draw mixture proportions and sample cluster labels.
cluster_sizes <- runif(K, 0.03, 0.15)
cluster_sizes <- cluster_sizes / sum(cluster_sizes)
z <- sample(1:K, N, replace = TRUE, prob = cluster_sizes)

# DISEASE PRESENCE MODEL.
# Build near independent per condition presence and add small noise.
pi_m <- rbeta(M, 0.07, 0.49)
pi_mk <- matrix(rep(pi_m, each = K), nrow = M, ncol = K)
epsilon <- matrix(rnorm(M * K, 0, 0.01), nrow = M, ncol = K)
pi_mk <- pmin(pmax(pi_mk + epsilon, 0.0001), 0.9999)

# Sample binary presence for each person and condition.
d <- matrix(0, N, M)
for (n in 1:N) {
  d[n, ] <- rbinom(M, 1, prob = pi_mk[, z[n]])
}

# CLUSTER SPECIFIC ONSET PARAMETERS WITH LESS SEPARATION.
# Create overlapping cluster specific means and sample variances.
mu_mk <- matrix(NA, M, K)
sigma2_mk <- matrix(NA, M, K)

# Build a narrow grid of means to induce overlap.
cluster_mean_grid <- seq(50, 70, length.out = K)

for (m in 1:M) {
  for (k in 1:K) {
    sigma2 <- extraDistr::rinvgamma(1, alpha = 4, beta = 150)

    mu <- rnorm(1, mean = cluster_mean_grid[k], sd = 4)  # A bit wider around each mean.
    mu_mk[m, k] <- mu
    sigma2_mk[m, k] <- sigma2
  }
}

# CONDITION SPECIFIC DIAGNOSIS DELAYS.
# Draw per condition delay distribution parameters.
delay_mu <- rtruncnorm(M, a = 0, mean = 5, sd = 1.5)
delay_sd <- rep(2, M)

# GENERATE ONSET DELAY AND DIAGNOSIS TIMES.
# Simulate onset given cluster and add delay to get diagnosis time.
onset <- matrix(0, N, M)
delay <- matrix(0, N, M)
t <- matrix(0, N, M)
for (n in 1:N) {
  for (m in 1:M) {
    if (d[n, m] == 1) {
      mu    <- mu_mk[m, z[n]]
      sigma <- sqrt(sigma2_mk[m, z[n]])

      # Sample onset age for present condition.
      onset[n, m] <- max(rnorm(1, mean = mu, sd = sigma), 0)

      # Sample diagnosis delay from truncated normal.
      delay[n, m] <- truncnorm::rtruncnorm(1, a = 0,
                                           mean = delay_mu[m],
                                           sd   = delay_sd[m])

      # Compute diagnosis time as onset plus delay.
      t[n, m] <- onset[n, m] + delay[n, m]
    }
  }
}

# INDIVIDUAL CENSORING.
# Simulate entry age, follow up window, and event indicator.
rho <- runif(N, 20, 60)        # baseline age
tau <- rho + 30                # current age
iota <- rbinom(N, 1, 0.8)      # 1 = dead, 0 = right-censored

# SAVE DATASET.
data_list <- list(
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
  delay_mu = delay_mu,
  delay_sd = delay_sd
)

saveRDS(data_list, "data/generated_onset_promote_style.rds")
