

set.seed(4)
library(extraDistr) 
library(MASS)
library(truncnorm)

# Parameters.
N <- 2000  # Use 200000 for original simulated ProMOTe study
K <- 10     # Number of clusters
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


M <- length(cond_list)

# Cluster Assignments.
cluster_sizes <- runif(K, 0.03, 0.15)
cluster_sizes <- cluster_sizes / sum(cluster_sizes)
z <- sample(1:K, size = N, replace = TRUE, prob = cluster_sizes)

# Condition presence probabilities pi_{m,k} ~ Beta(0.07, 0.49)
pi_mk <- matrix(rbeta(M * K, 0.07, 0.49), nrow = M, ncol = K)

# Condition presence d_{n,m} ~ Bernoulli(pi_{m,z_n})
d <- matrix(0, nrow = N, ncol = M)
for (n in 1:N) {
  d[n, ] <- rbinom(M, 1, prob = pi_mk[, z[n]])
}

# Onset age parameters from NIG prior: mu ~ N(50, sigma2/0.3), sigma2 ~ InvGamma(5, 300)
mu_mk <- matrix(NA, M, K)
sigma2_mk <- matrix(NA, M, K)

for (m in 1:M) {
  for (k in 1:K) {
    sigma2 <- rinvgamma(1, 5, 300)
    mu <- rnorm(1, mean = 50, sd = sqrt(sigma2 / 0.3))
    mu_mk[m, k] <- mu
    sigma2_mk[m, k] <- sigma2
  }
}

# Condition-specific (not cluster-specific) diagnosis delays
delay_mu <- rtruncnorm(M, a = 0, mean = 5, sd = 2)
# Larger variance in diagnosis delay (makes recovery harder)
delay_sd <- rtruncnorm(M, a = 0.1, mean = 4, sd = 1.5)


# Generate onset, delay, and diagnosis time t[n,m] (only if d == 1)
t <- matrix(0, nrow = N, ncol = M)
onset <- matrix(0, nrow = N, ncol = M)
delay <- matrix(0, nrow = N, ncol = M)

for (n in 1:N) {
  for (m in 1:M) {
    if (d[n, m] == 1) {
      mu <- mu_mk[m, z[n]]
      sigma <- sqrt(sigma2_mk[m, z[n]])
      onset[n, m] <- max(rnorm(1, mu, sigma), 0)  # onset determined by cluster
      delay[n, m] <- rtruncnorm(1, a = 0, mean = delay_mu[m], sd = delay_sd[m])
      t[n, m] <- onset[n, m] + delay[n, m] # diagnosis distorted by non-cluster specfic delay

    }
  }
}

# Individual censoring
rho <- runif(N, 20, 60)         # baseline age
tau <- rho + 30                 # current age
iota <- rbinom(N, 1, 0.8)       # 1 = death, 0 = censored

# Save data
data_list <- list(
  z = z,
  d = d,
  t = t,
  onset = onset,
  rho = rho,
  tau = tau,
  iota = iota,
  N = N,
  K = K,
  M = M,
  cond_list = cond_list,
  pi_mk = pi_mk,
  mu_mk = mu_mk,
  sigma2_mk = sigma2_mk
)

saveRDS(data_list, "data/generated_presence_promote_style.rds")