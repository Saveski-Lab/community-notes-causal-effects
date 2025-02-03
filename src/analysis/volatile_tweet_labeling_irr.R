# Taken from:
# https://link.springer.com/article/10.1186/s12874-016-0200-9#MOESM3

library(readr)
library(dplyr)
library(tidyr)

k_alpha <- function(ratings_t, alpha_q = 0.05, nboot = 1000, scaling = "nominal") {

  # functions for Fleiss' K and Krippendorff's alpha
  # function for the estimation of Fleiss' K
  k_func = function(N, n, k, ratings, categ) {
    # n_ij = number of raters who classed subject i in category j
    n_ij = matrix(ncol = k, nrow = N)
    step = 1
    for (j in categ) {
      for (i in 1:N) {
        n_ij[i, step] = sum(as.numeric(ratings[i,] == j))
      }
      step = step + 1
    }
    # estimation of K_j
    p_j = apply(n_ij, 2, sum) / (N * n)
    q_j = 1 - p_j
    k_j = 1 - apply(n_ij * (n - n_ij), 2, sum) / (N * n * (n - 1) * p_j * q_j)
    # estimation of the overall K
    k_t = sum(p_j * q_j * k_j) / sum(p_j * q_j)
    return(list(k_t, p_j, q_j))
  }

  # Function for the estimation of alpha
  alpha_func = function(k, n, N, ratings, categ) {
    # conicidence matrix
    CM = matrix(ncol = k, nrow = k, 0)
    vn <- function(datavec) sum(!is.na(datavec))
    if (any(is.na(ratings)))  mc = apply(ratings, 1, vn) - 1    else mc = rep(n - 1, N)
    for (i in 1:N) {
      for (j in 1:(n - 1)) {
        for (jt in (j + 1):n) {
          if (!is.na(ratings[i, j]) && !is.na(ratings[i, jt])) {
            index1 = which(categ == ratings[i, j])
            index2 = which(categ == ratings[i, jt])
            CM[index1, index2] = CM[index1, index2] + (1 + (index1 == index2)) / mc[i]
            if (index1 != index2) {
              CM[index2, index1] = CM[index1, index2]
            }
          }
        }
      }
    }
    nmv <- sum(apply(CM, 2, sum))
    nc = apply(CM, 1, sum)
    ncnk = matrix(0, nrow = k, ncol = k)

    # matrix of expected disagreement
    D_e = matrix(0, ncol = k, nrow = k)
    for (C in 1:k) {
      for (Ct in 1:k) {
        if (C == Ct) {
          D_e[C, Ct] = nc[C] * (nc[Ct] - 1) / (nmv - 1)
        }
        if (C != Ct) {
          D_e[C, Ct] = nc[C] * nc[Ct] / (nmv - 1)
        }
        ncnk[C, Ct] = nc[C] * nc[Ct]
        ncnk[Ct, C] = ncnk[C, Ct]
      }
    }

    # matrix of metric differences
    diff2 = matrix(0, nrow = k, ncol = k)
    # nominal
    if (match(scaling[1], "nominal", 0)) {
      diff2 = matrix(1, ncol = k, nrow = k)
      diag(diff2) = 0
    }
    # ordinal
    if (match(scaling[1], "ordinal", 0)) {
      for (C in 1:k) {
        for (Ct in 1:k) {
          if (C != Ct) {
            tmp = nc[C:Ct]
            diff2[C, Ct] = (sum(tmp) - nc[C] / 2 - nc[Ct] / 2)^2
            diff2[Ct, C] = diff2[C, Ct]
          }
        }
      }
    }
    # interval
    if (match(scaling[1], "interval", 0)) {
      for (C in 1:k) {
        for (Ct in 1:k) {
          if (C != Ct) {
            diff2[C, Ct] = (as.numeric(categ)[C] - as.numeric(categ)[Ct])^2
            diff2[Ct, C] = diff2[C, Ct]
          }
        }
      }
    }
    # ratio
    if (match(scaling[1], "ratio", 0)) {
      for (C in 1:k) {
        for (Ct in 1:k) {
          if (C != Ct) {
            diff2[C, Ct] = ((as.numeric(categ)[C] - as.numeric(categ)[Ct]) /
              (as.numeric(categ)[C] + as.numeric(categ)[Ct]))^2
            diff2[Ct, C] = diff2[C, Ct]
          }
        }
      }
    }
    # point estimator of Krippendorff's alpha
    tmp = diff2 * CM
    num = sum(tmp)
    tmp = diff2 * D_e
    den = sum(tmp)
    if (den > 0) {
      alpha_boot = 1 - num / den
    }
    if (den <= 0) {
      alpha_est = NA
    }
    return(alpha_boot)
  }


  ################################################
  ############ Fleiss' K #####################
  ################################################

  # check, if measurement scale is nominal
  if (match(scaling[1], "nominal", 0)) {
    # deleting all subjects with missing values
    ratings_c <- as.matrix(na.omit(ratings_t))
    # N = number of subjects, n = number of raters, k = number of categories
    N_c = nrow(ratings_c)

    v = function(dat) { min(dat) == max(dat) }
    agr_k = sum(apply(ratings_c, 1, v)) / N_c

    # check, if there are at least two individuals without missing values
    if (N_c < 2) {
      print("There are less than two subjects withour missing values. Therefore, Fleiss' K cannot be                               calculated.")
    }
    if (N_c >= 2) {
      n_c = ncol(ratings_c)
      categ_c = levels(as.factor(ratings_c))
      k_c = length(categ_c)

      # point estimator of Fleiss` K
      k_ = k_func(N_c, n_c, k_c, ratings_c, categ_c)
      k_est = k_[[1]]
      p_j = k_[[2]]
      q_j = k_[[3]]

      ########### asymptotic confidence interval ######################
      # estimation of the standard error
      se_k = (sqrt(2) / (sum(p_j * q_j) * sqrt(N_c * n_c * (n_c - 1)))) * sqrt(sum(p_j * q_j)^2 - sum(p_j * q_j * (q_j - p_j)))
      # asymptotic confidence interval for Fleiss' K
      CI_asymp_k = k_est + c(-1, 1) * qnorm(1 - alpha_q / 2) * se_k
    }
  }

  #######################################################
  ############ Krippendorff's alpha #####################
  #######################################################

  # deleting all subject with less than two ratings
  f = function(x)sum(!is.na(x))
  # deleting all subjects with only one rating
  ratings = as.matrix(ratings_t[apply(ratings_t, 1, f) > 1,])
  # N = number of subjects, n = number of raters, k = number of categories
  N_kr = nrow(ratings)

  v = function(dat) { min(dat, na.rm = TRUE) == max(dat, na.rm = TRUE) }
  agr_alpha = sum(apply(ratings, 1, v)) / N_kr

  n_kr = ncol(ratings)
  categ = levels(as.factor(ratings))
  k_kr = length(categ)
  # point estimator of Krippendorff's alpha
  alpha_est = alpha_func(k_kr, n_kr, N_kr, ratings, categ)


  ##################################################################
  ########## Bootstrap confidence intervals #########################
  ##################################################################

  # K and alpha in each Bootstrap sample
  k_boot = 0
  alpha_boot = 0
  for (iboot in 1:nboot) {
    if (match(scaling[1], "nominal", 0)) {
      index.new = sample(seq(1, N_c, 1), N_c, replace = TRUE)
      ratings_b = ratings_c[index.new,]
      n = ncol(ratings_b)
      categ = levels(as.factor(ratings_b))
      k = length(categ)
      k.b <- k_func(N_c, n, k, ratings_b, categ)[[1]]
      k_boot = c(k_boot, k.b)
    }
    f = function(x)sum(!is.na(x))
    # deleting all subjects with only one rating
    index.new = sample(seq(1, N_kr, 1), N_kr, replace = TRUE)
    ratings_b = ratings[index.new,]
    n = ncol(ratings)
    categ = levels(as.factor(ratings))
    k = length(categ)
    alpha_b = alpha_func(k, n, N_kr, ratings_b, categ)
    alpha_boot = c(alpha_boot, alpha_b)
  }
  # confidence interval using the percentiles from the Bootstrap samples
  if (match(scaling[1], "nominal", 0)) {
    CI_boot_k = quantile(k_boot[-1], probs = c(alpha_q / 2, 1 - alpha_q / 2), na.rm = TRUE)
  }
  # confidence interval using the percentiles from the Bootstrap samples
  CI_boot_alpha = quantile(alpha_boot[-1], probs = c(alpha_q / 2, 1 - alpha_q / 2), na.rm = TRUE)


  ##########################################
  ############# Output #####################
  ##########################################

  print(paste("The measurement scale is ", scaling, "."))
  print("###### Fleiss' K ######")
  if (!match(scaling[1], "nominal", 0)) {
    print(paste("Fleiss' K cannot be calculated, because it is only appropriate for nominal data."))
  }
  if (match(scaling[1], "nominal", 0)) {
    print(paste("The observed agreement in all complete cases is ", round(agr_k * 100, 1), "%."))
    print(paste("N (number of subjects without missing values) =", N_c))
    print(paste("n (number of ratings) =", n_c))
    print(paste("k (number of categories) = ", k_c))
    print(paste("Point estimator of Fleiss' K =", round(k_est, 4)))
    print(paste("Asymptotic two-sided ", (1 - alpha_q) * 100, "% confidence interval for Fleiss' K:", round(CI_asymp_k[1], 4), " ; ", min(1, round(CI_asymp_k[2], 4))))
    print(paste("Two-sided ", (1 - alpha_q) * 100, "% Bootstrap confidence interval for Fleiss' K:", round(CI_boot_k[1], 4), " ; ", round(CI_boot_k[2], 4)))
  }

  print("###### Krippendorff's alpha ######")
  print(paste("The observed agreement in all cases with at least two ratings is ", round(agr_alpha * 100, 1), "%."))
  print(paste("N (number of subjects with two or more ratings) =", N_kr))
  print(paste("n (number of ratings) =", n_kr))
  print(paste("k (number of categories) = ", k_kr))
  print(paste("Point estimator of Krippendorff's alpha =", round(alpha_est, 4)))
  print(paste("Two-sided ", (1 - alpha_q) * 100, "% Bootstrap confidence interval for Krippendorff's alpha:", round(CI_boot_alpha[1], 4), " ; ", round(CI_boot_alpha[2], 4)))

  if (match(scaling[1], "nominal", 0)) {
    return(invisible(list(obs.agr.k = agr_k, est.k = k_est, ci.asympt.k = CI_asymp_k, ci.boot.k = CI_boot_k,
                          obs.agr.alpha = agr_alpha, est.alpha = alpha_est, ci.boot.alpha = CI_boot_alpha)))
  }
  if (!match(scaling[1], "nominal", 0)) {
    return(invisible(list(obs.agr.alpha = agr_alpha, est.alpha = alpha_est, ci.boot.alpha = CI_boot_alpha)))
  }


}

# Read in the data from disk
joint_annotations <- readr::read_csv(
  "src/analysis/volatile_tweet_labeling_round_2/annotation_output/volatile-tweet-labeling/annotated_instances.csv",

  # Make sure ID is read in as a character
  col_types = cols(
    "instance_id" = col_character()
  )
) %>%

  # Make sure that volatility gets stored as a boolean
  dplyr::mutate(
    `is_volatile:::No` = !is.na(`is_volatile:::No`)
  ) %>%

  # Pivot from wide to long (so each row corresponds to an image and the two columns correspond to two raters
  tidyr::pivot_wider(
    id_cols = instance_id,
    names_from = user,
    values_from = `is_volatile:::No`
  ) %>%

  # Remove instance ID
  dplyr::select(-instance_id) %>%

  # Drops rows with NA values
  dplyr::filter(!is.na(`is28@uw.edu`), !is.na(`msaveski@uw.edu`))

# Calculate Fleiss' K and Krippendorff's alpha, with CI
set.seed(14132)
k_alpha(joint_annotations, alpha_q = 0.05, nboot = 1000)