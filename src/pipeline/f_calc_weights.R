## Load libraries #######################################################################################################

# Load utils file
source("src/utils.R")

pkgs <- c(
  "tools",
  "fs",
  "dplyr",
  "tibble",
  "arrow",
  "readr",
  "lubridate",
  "ggplot2",
  "scales",
  "log4r",
  "foreach",
  "remotes",
  "uuid",
  "argparse",
  "jsonlite",
  "progressr",
  "digest",
  "Matrix",
  "data.table",
  "doMC"
)

# Install any missing packages
for (pkg in pkgs) {
  check_and_install(pkg)
}

# Install pensynth
remotes::install_github("isaacOnline/pensynth", ref = "main")

# Set global options ###################################################################################################

# Initialize cluster
options(future.globals.maxSize = 40 * 1024^3)

# Register handlers for progress bar
handlers(global = TRUE)
handlers("progress")

parser <- argparse::ArgumentParser(description = "Calculate synthetic control weights")

parser$add_argument("--config", type = "character", required = TRUE, help = "Path to the config file")

args <- parser$parse_args()

config <- jsonlite::read_json(args$config)

if (config$shuffle) {
  NUM_WORKERS <- 6
} else {
  NUM_WORKERS <- 30
}

# Fill in default value, if we've been asked to
if (config$dev == "DEVICE_DEFAULT") {
  config$dev <- Sys.info()[["nodename"]] == "is-is28m16x"
}

# Set directory for storing intermediate results
if (config$dev) {
  intermediate_dir <- "cn_effect_intermediate_dev"
} else {
  intermediate_dir <- "cn_effect_intermediate_prod"
}

# Make sure we have all necessary config values
necessary_config <- c(
  "time_freq",
  "dev",
  "volatile_tweet_filtering",
  "train_backdate",
  "pre_break_min_time",
  "pre_break_max_time",
  "post_break_min_time",
  "matching_metrics",
  "replace_calculated_when_missing",
  "missing_metric_action",
  "include_permutations",
  "lambda",
  "restrict_donor_pool",
  "restricted_pool_size",
  "shuffle"
)
for (config_key in necessary_config) {
  if (!(config_key %in% names(config))) {
    stop(paste0(
      "Please specify config value '",
      config_key,
      "' in config file '",
      args$config,
      ".'"
    ))
  }
}

optional_config <- c(
  "control_list"
)

# Remove any unneeded keys
for (config_key in names(config)) {
  if (!(config_key %in% necessary_config) && !(config_key %in% optional_config)) {
    config <- config[!names(config) %in% config_key]
  }
}


# Define the metric_parents dictionary
metric_parents <- list(
  "replies" = "metrics",
  "likes" = "metrics",
  "impressions" = "metrics",
  "retweets" = "metrics",
  "calculated_replies" = "calculated_replies",
  "calculated_retweets" = "calculated_retweets",
  "author_n_followers" = "author",
  "rt_cascade_width" = "rt_cascade",
  "rt_cascade_depth" = "rt_cascade",
  "rt_cascade_wiener_index" = "rt_cascade"
)
for (embd_dim in c(20, 58)) {
  for (i in 0:(embd_dim - 1)) {
    metric_parents[[sprintf("PCA_%d_embd_%d", embd_dim, i)]] <- "author"
  }
}

# Compute tweet_metrics
config$tweet_metrics <- Filter(function(m) {
  metric_parents[[m]] %in% c(
    "metrics", "calculated_retweets", "calculated_replies",
    "rt_graph", "reply_graph", "reply_tree", "rt_cascade"
  )
}, config$matching_metrics)


# Compute backup_tweet_metrics
if (config$replace_calculated_when_missing) {
  config$backup_tweet_metrics <- gsub("calculated_", "", Filter(function(m) {
    grepl("calculated_", m)
  }, config$matching_metrics))
} else {
  config$backup_tweet_metrics <- c()
}

# Compute author_metrics
config$author_metrics <- Filter(function(m) {
  metric_parents[[m]] == "author"
}, config$matching_metrics)

# Convert lambda to a vector
config$lambda <- unlist(config$lambda)

# Set input and output directories
merge_dir <- fs::path(
  data_root,
  intermediate_dir,
  "b_merged"
)
control_dir <- fs::path(
  data_root,
  intermediate_dir,
  "c_find_controls"
)
output_dir <- fs::path(
  data_root,
  intermediate_dir,
  "f_calc_weights"
)
log_dir <- fs::path(
  data_root,
  intermediate_dir,
  "logs"
)

# Make sure output dir exists
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}

train_backdate <- -convert_to_hours(config$train_backdate)

# Set up logging
logger <- log4r::logger(
  "INFO",
  appenders = list(
    log4r::console_appender(),
    log4r::file_appender(fs::path(log_dir, paste0("f_calc_weights", gsub("\\.", "_", format(Sys.time(), "%Y-%m-%d_%H-%M-%OS6")), ".log")))
  )
)
log4r::info(logger, paste0("Starting new run on device ", Sys.info()[["nodename"]], "."))

log4r::info(logger, paste0("Using config file: ", args$config))
log4r::info(logger, paste0("Using ", names(config), ": ", config, "\n"))

## Load seeds ##########################################################################################################

log4r::info(logger, "Loading seeds")
tweet_seeds_path <- fs::path(data_root, "__tweet_seeds.json")
tweet_seeds <- jsonlite::read_json(tweet_seeds_path)

get_tweet_seed <- function(tweet_id) {
  if (tweet_id %in% names(tweet_seeds)) {
    return(tweet_seeds[[tweet_id]])
  } else {
    stop(sprintf("tweet_id '%s' not found in tweet_seeds.", tweet_id))
  }
}

## Load standard deviations to normalize by ############################################################################

matching_metrics_vector <- unlist(config$matching_metrics)
matching_metrics_entry <- paste0(matching_metrics_vector, collapse = ",")

sds_config <- config[!names(config) %in% c(
  "title",
  "include_permutations",
  "lambda",
  "restricted_pool_size",
  "restrict_donor_pool",
  "tweet_metrics",
  "backup_tweet_metrics",
  "author_metrics",
  "control_list"
)]
sds_config$matching_metrics <- matching_metrics_entry

treatment_sds <- read_from_table(fs::path(control_dir, "metric_sds.parquet"), sds_config)  %>%
  dplyr::mutate(sd = round(sd, 5))

# Make sure there are more than 0 rows
if (nrow(treatment_sds) == 0) {
  log4r::info(logger, paste0("No standard deviations found for matching metrics: ", matching_metrics_entry))
  stop(paste0("No standard deviations found for matching metrics: ", matching_metrics_entry))
}

## Read author data ####################################################################################################

log4r::info(logger, "Loading author data.")
author_data_path <- fs::path(data_root, "tweet_metadata.parquet")

author_metrics <- arrow::read_parquet(author_data_path) %>%
  dplyr::select(tweet_id, author_n_followers) %>%
  dplyr::filter(!is.na(author_n_followers))

log4r::info(logger, "Author data has been loaded.")

## Read control configurations #########################################################################################

log4r::info(logger, "Loading synthetic control configuration files")

read_control_configs <- function() {
  # Get the name for all the treatment tweet files
  all_tweet_fnames <- list.files(control_dir, pattern = "*.json.gz", full.names = F)


  # If we are only using a subset of controls, read the subset here
  if (!is.null(config$control_list)) {
    small_control_set <- readr::read_csv(config$control_list,
                                    col_types = list(
                                      "tweet_id" = readr::col_character())
    ) %>% pull(tweet_id)
  }


  # Filter to exclude any config items that weren't needed/included in input data
  input_config <- config[!names(config) %in% c(
    "title",
    "include_permutations",
    "lambda",
    "restricted_pool_size",
    "restrict_donor_pool",
    "tweet_metrics",
    "backup_tweet_metrics",
    "author_metrics",
    "control_list"
  )]

  pbar <- progressr::progressor(along = all_tweet_fnames)

  config_read_start_time <- Sys.time()

  acc <- foreach(
    fname = all_tweet_fnames,
    .packages = pkgs,
    .export = c("read_from_table", "control_dir")
  ) %dopar% {
    tid <- sub(".json.gz", "", fname)
    tweet_config <- read_from_table(fs::path(
      control_dir, fname
    ), input_config)


    if (length(tweet_config) != 1) {
      return(NULL)
    }

    # If we are only using a subset of controls, filter down here
    if (!is.null(config$control_list)) {
      tweet_config[[1]]$control_tweet_ids <- tweet_config[[1]]$control_tweet_ids %>%
        unlist() %>%
        intersect(small_control_set) %>%
        as.list()
    }


    pbar(
      paste0(
        "Reading control configs. Time Elapsed=",
        pretty_dur(lubridate::as.duration(Sys.time() - config_read_start_time))
      )
    )


    return(setNames(tweet_config, tid))
  }

  acc <- Filter(Negate(is.null), acc)

  return(do.call(c, acc))
}

all_control_configs <- read_control_configs()

log4r::info(logger, "Synthetic control configuration files have been loaded")


read_tweet_data <- function(tid, cfg) {
  read_data <- read_from_table(
    fs::path(merge_dir, paste0(tid, ".parquet")),
    cfg[names(cfg) %in% c(
      "dev", "volatile_tweet_filtering", "time_freq"
    )]
  )

  return(read_data)
}

read_all_metrics <- function(config_to_read) {
  all_tweet_metric_files <- list.files(merge_dir, pattern = "*.parquet", full.names = TRUE)

  # Get all available metrics across tweets
  all_metrics <- unlist(c(
    config_to_read$tweet_metrics,
    config_to_read$backup_tweet_metrics
  ))

  pbar <- progressr::progressor(along = all_tweet_metric_files)
  read_start_time <- Sys.time()

  # Now process the slices in parallel
  result <- foreach(
    fname = all_tweet_metric_files,
    .packages = pkgs,
    .export = c("all_metrics", "config_to_read", "read_tweet_data", "read_from_table", "merge_dir", "pretty_dur", "read_start_time"),
    .noexport = c("logger")
  ) %dopar% {

    tweet_id <- sub(".parquet", "", basename(fname))
    tweet_data <- read_tweet_data(tweet_id, config_to_read) %>%
      data.table::as.data.table()

    local_time_vals <- numeric(0)
    local_tid_vals <- character(0)
    local_metric_vals <- character(0)
    local_x_vals <- numeric(0)

    # For each metric, create a sparse matrix with tweets as columns
    for (col_idx in seq_along(all_metrics)) {
      metric <- all_metrics[col_idx]

      if (metric %in% colnames(tweet_data)) {
        # Extract only the necessary columns for this metric
        tweet_metric_data <- tweet_data[, c("time_since_publication", metric), with = FALSE]

        # For each non-NA value, add to our sparse matrix triplets
        non_na_rows <- which(!is.na(tweet_metric_data[[metric]]))
        if (length(non_na_rows) > 0) {
          times <- tweet_metric_data$time_since_publication[non_na_rows]
          values <- tweet_metric_data[[metric]][non_na_rows]

          # Store in local variables
          local_metric_vals <- c(local_metric_vals, rep(metric, length(times)))
          local_time_vals <- c(local_time_vals, times)
          local_tid_vals <- c(local_tid_vals, rep(tweet_id, length(times)))
          local_x_vals <- c(local_x_vals, values)

        }
      }

    }

    # Update progress bar
    pbar(paste0(
      "Reading tweets; Time Elapsed=",
      pretty_dur(lubridate::as.duration(Sys.time() - read_start_time))
    ))

    # Return local results as a list
    list(
      metric = local_metric_vals,
      time = local_time_vals,
      tid = local_tid_vals,
      x = local_x_vals
    )
  }

  # Combine results from parallel processing
  metric_vals <- unlist(lapply(result, function(x) x$metric))
  time_vals <- unlist(lapply(result, function(x) x$time))
  tid_vals <- unlist(lapply(result, function(x) x$tid))
  x_vals <- unlist(lapply(result, function(x) x$x))

  # Get the unique times/tweet IDs
  all_times <- unique(time_vals)
  all_times <- sort(all_times)
  present_tweet_ids <- unique(tid_vals)
  present_tweet_ids <- sort(present_tweet_ids)

  result <- list()

  for (metric in all_metrics) {
    # Filter to only include the metric we want
    metric_indexer <- which(metric_vals == metric)
    metric_time_vals <- time_vals[metric_indexer]
    metric_tid_vals <- tid_vals[metric_indexer]
    metric_x_vals <- x_vals[metric_indexer]

    # Get the row and column indices for the sparse matrix
    metric_i_vals <- match(metric_time_vals, all_times)
    metric_j_vals <- match(metric_tid_vals, present_tweet_ids)

    # Create the sparse matrix
    sparse_mat <- Matrix::sparseMatrix(
      i = metric_i_vals,
      j = metric_j_vals,
      x = metric_x_vals,
      repr = "C",
      dims = c(length(all_times), length(present_tweet_ids)),
      dimnames = list(all_times, present_tweet_ids)
    )

    # Store in result list with time information
    result[[metric]] <- list(
      matrix = sparse_mat,
      times = all_times
    )
  }

  return(result)

}

log4r::info(logger, "Loading tweet time series data.")


atm <- read_all_metrics(config)

mem_size <- utils::object.size(atm) / 1024^2  # Convert to MB
log4r::info(logger, paste("Main process memory usage:", mem_size, "MB"))

log4r::info(logger, "Tweet time series data has been loaded.")

## Helper Functions ####################################################################################################
standardize_X <- function(X1, X0, sd = NULL, mu = NULL) {
  # Function to scale the X matrices in synthetic control
  if (!is.null(mu)) {
    return(list(X1 = (X1 - mu) / sd, X0 = (X0 - mu) / sd))
  } else {
    return(list(X1 = X1 / sd, X0 = X0 / sd))
  }
}

get_closest_control_ids <- function(X0, X1, num_controls) {
  colnames(X1) <- "treatment"

  # Calculate euclidean distance between treatment and each control
  distances <- apply(X0, 2, function(col) {
    sqrt(sum((X1 - col)^2))
  })

  # Return the IDs of the closest controls
  tibble::tibble(
    tid = colnames(X0),
    distance = distances
  ) %>%
    dplyr::arrange(distance) %>%
    dplyr::slice(1:num_controls) %>%
    dplyr::pull(tid)
}

get_time_range_for_metric <- function(metric, matching_time_range) {
  for (tr in matching_time_range) {
    if (tr[[1]] == metric) {
      return(tr)
    }
  }
}

get_author_metrics <- function(aut_m, tid, metric) {
  # Get an author metric, e.g. author_n_followers

  # Filter author metrics dataframe
  aut_m %>%
    # Filter to the tweet ID requested
    dplyr::filter(
      tweet_id == !!tid
    ) %>%
    # Get the metric requested
    dplyr::pull(!!sym(metric))
}

get_standardization_vector <- function(control_config, tsds) {
  # Iterate through each of the metrics, and get the time ranges for matching
  matching_time_ranges <- list()
  max_matching_times <- list()
  for (metric in unlist(control_config$metrics_present_for_tweet)) {
    if (!metric %in% control_config$author_metrics) {
      time_ranges <- get_time_range_for_metric(metric, control_config$matching_timestamps)[[2]] %>% unlist()
      matching_time_ranges[[metric]] <- time_ranges
      max_matching_times[[metric]] <- time_ranges %>% max
    }
  }

  # Find the maximum matching time for all metrics
  if (length(max_matching_times) > 0) {

    max_matching_time <- max(unlist(max_matching_times))

    # Add 15 minutes to the max time to get the slap time
    first_crh_time <- (
      max_matching_time
        + lubridate::duration(15, "minutes")
    ) %>%
      as.numeric("seconds")

    # Make a time-to-slap/time-since-publication dataframe
    tsds <- tsds %>%
      dplyr::mutate(
        time_since_publication = note_0_time_since_first_crh + first_crh_time,
      )
  }

  # Slice each dataframe to only include timestamps we want to match on
  metric_slices <- list()
  for (metric in unlist(control_config$metrics_present_for_tweet)) {
    if (!metric %in% control_config$author_metrics) {
      # If this is a tweet metric, slice the dataframe to only include the time points we want to match on
      # And columns for tweet Ids we'll be using (both treatment and control)
      metric_slices[[metric]] <- tsds %>%
        dplyr::filter(
          metric == !!metric
        ) %>%
        dplyr::select(
          time_since_publication,
          sd
        ) %>%
        dplyr::filter(
          time_since_publication %in% matching_time_ranges[[metric]]
        ) %>%
        dplyr::arrange(time_since_publication)
    } else {
      metric_slices[[metric]] <- tsds %>%
        dplyr::filter(
          metric == !!metric
        ) %>%
        dplyr::select(
          time_since_publication,
          sd
        )
    }
  }

  # Merge the metrics into a vector
  sd_vec <- do.call(rbind, metric_slices) %>%
    pull("sd")

  return(sd_vec)
}


load_tweet_metrics <- function(fname, control_config, aut_m) {
  # Get the ID for this tweet based on its file name
  tid <- sub(".json.gz", "", fname)

  # Read the matching parameters from disk (e.g. what metrics to match on)

  # Get the list of control tweet IDs that we'll be using
  control_ids <- unlist(control_config$control_tweet_ids)


  # Iterate through each of the metrics, and get the time ranges for matching
  metric_slices <- list()
  for (metric in unlist(control_config$metrics_present_for_tweet)) {
    if (!metric %in% control_config$author_metrics) {
      time_ranges <- get_time_range_for_metric(metric, control_config$matching_timestamps)[[2]] %>% unlist()
      rows <- which(atm[[metric]][["times"]] %in% time_ranges) %>% sort()
      columns <- c(tid, control_ids)

      metric_slices[[metric]] <- atm[[metric]][["matrix"]][rows, columns] %>%
        Matrix::as.matrix() %>%
        as.data.frame()

      colnames(metric_slices[[metric]]) <- columns
    } else {
      author_data <- list()
      for (tweet_id in c(tid, control_ids)) {
        # Get the author metric, e.g. author_n_followers
        result <- get_author_metrics(aut_m, tweet_id, metric)
        author_data[[tweet_id]] <- if (length(result) == 0) NA else result
      }
      # Create a dataframe with the author metric for each tweet
      author_df <- data.frame(
        matrix(unlist(author_data), nrow = 1, byrow = TRUE)
      )
      # Impute NAs
      author_df[is.na(author_df)] <- mean(unlist(author_df), na.rm = TRUE)
      colnames(author_df) <- names(author_data)
      metric_slices[[metric]] <- author_df
    }
  }

  # Merge the metrics into a single dataframe
  tweet_df <- do.call(rbind, metric_slices)

  treatment_data <- tweet_df[, tid]

  control_data <- tweet_df[, -which(colnames(tweet_df) == tid)]

  return(list(
    treatment = treatment_data,
    control = control_data
  ))
}

remove_nas_and_degenerates <- function(X1_to_clean, X0_to_clean, tweet_fname) {
  log4r::info(logger, paste0("X0 and X1 are of shapes ", paste(dim(X0_to_clean), collapse = "x"), " and ", paste(dim(X1_to_clean), collapse = "x"), "respectively for  tweet ", tweet_fname))


  # 1) Error if X1 contains any NAs
  if (any(is.na(X1_to_clean))) {
    log4r::error(logger, paste0("For ", tweet_fname, ", X1 contains missing values at rows."))
  }

  # 2) Error if X0 contains any NAs
  if (any(is.na(X0_to_clean))) {
    log4r::error(logger, paste0("For ", tweet_fname, ", X0 contains missing values at rows."))
  }

  # Filter out metric observations where all donors have the same value
  observations_to_drop <- apply(X0_to_clean, 1, function(x) sd(x) == 0)
  cleaned_X1 <- X1_to_clean[!observations_to_drop, , drop = FALSE]
  cleaned_X0 <- X0_to_clean[!observations_to_drop, , drop = FALSE]


  if (sum(observations_to_drop) > 0) {
    log4r::warn(logger, paste0(
      "For ", tweet_fname, ", removed ", sum(observations_to_drop),
      " matching variables due to the metric being identical across all donors at this point in time. ",
      "The shape of X0 and X1 are now ", paste(dim(cleaned_X0), collapse = "x"), " and ", paste(dim(cleaned_X1), collapse = "x"), " respectively. ",
      "The point(s) dropped were: ", paste(names(observations_to_drop[observations_to_drop]), collapse = ", ")))
  }

  # Compute hashes for each post to identify duplicates
  post_hashes <- apply(cleaned_X0, 2, digest::digest)

  # Find unique hashes and group duplicate posts
  hash_groups <- split(seq_along(post_hashes), post_hashes)

  # Initialize lists to store posts to keep and drop
  posts_to_drop <- integer(0)

  # Initialize list to store logs for duplicates
  duplicate_logs <- list()

  # Set a seed for reproducibility
  tweet_id <- gsub(".json.gz", "", tweet_fname)
  tweet_seed <- get_tweet_seed(tweet_id)
  set.seed(tweet_seed)

  # Get the column names of cleaned_X0
  col_names <- colnames(cleaned_X0)

  # Filter to only hashes that have duplicates
  hash_groups <- hash_groups[sapply(hash_groups, length) > 1]

  # Loop through each group of duplicates
  for (hash in names(hash_groups)) {
    indices <- hash_groups[[hash]]
    col_names_group <- col_names[indices]

    # This is a pseudo-random way to select which post to keep,
    # which does not require seed setting to be reproducible.
    # Look up controls tweet seeds, concatinate treatment tweet seed, hash
    hashed_post_ids <- col_names_group %>%
      gsub(".json.gz", "", .) %>%
      sapply(function(X0_tweet_id) get_tweet_seed(X0_tweet_id)) %>%
      paste0(tweet_seed) %>%
      sapply(function(h) digest::digest(h, algo = "blake3"))

    names(hashed_post_ids) <- col_names_group

    hashed_post_ids <- sort(hashed_post_ids)
    itr_post_to_keep <- names(hashed_post_ids[1])
    itr_posts_to_drop <- setdiff(col_names_group, itr_post_to_keep)

    # Update the lists of posts to keep and drop
    posts_to_drop <- c(posts_to_drop, itr_posts_to_drop)

    # Log the duplicates
    duplicate_logs[[hash]] <- list(
      duplicate_posts = col_names_group,
      kept = itr_post_to_keep,
      dropped = itr_posts_to_drop
    )
  }

  # Get the indices of the posts to keep
  posts_to_keep <- setdiff(col_names, posts_to_drop)

  # Subset the cleaned_X0 matrix to keep only the selected posts
  cleaned_X0 <- cleaned_X0[, posts_to_keep]

  # Logging the duplicates with column names
  if (length(posts_to_drop) > 0) {
    total_duplicates <- length(posts_to_drop)

    log4r::warn(logger, paste0(
      "For ", tweet_fname, ", removed ", total_duplicates,
      " posts from donor pool due to having identical metrics."
    ))

    for (hash in names(duplicate_logs)) {
      dup_info <- duplicate_logs[[hash]]

      # Get the column names for duplicate posts, kept post, and dropped posts
      kept_post_name <- dup_info$kept
      dropped_post_names <- dup_info$dropped

      log4r::warn(logger, paste0(
        "Duplicate posts with hash ", hash, " for post ", tweet_fname, ".  ",
        "Kept post ", kept_post_name,
        ", dropped posts ", paste(dropped_post_names, collapse = ", "), "."
      ))
    }
  }

  return(list(X1 = cleaned_X1, X0 = cleaned_X0))
}

## Main function for Weight Calculation ################################################################################
calc_pensynth_weights <- function() {
  # Read covariates to match on
  matching_metrics_vector <- unlist(config$matching_metrics)
  matching_metrics_entry <- paste0(matching_metrics_vector, collapse = ",")

  # set standardization
  standardize <- TRUE

  # Get lambda sequence
  lseq <- config$lambda

  # Get restricted_pool_size
  if (config$restrict_donor_pool) {
    restricted_pool_size <- config$restricted_pool_size
  } else {
    restricted_pool_size <- NULL
  }

  log4r::info(logger, paste0(
    "Calculating weights. ",
    "Standardized: ", standardize, ". Restricted pool size: ", restricted_pool_size
  ))

  # Get the name for all the treatment tweet files
  all_tweet_ids <- names(all_control_configs)

  tweets_to_calculate <- list()
  for (tweet_id in all_tweet_ids) {
    fname <- paste0(tweet_id, ".json.gz")

    tconfig <- all_control_configs[[tweet_id]]
    if (tconfig$use_tweet) {
      tweets_to_calculate <- append(tweets_to_calculate, fname)
    }
  }
  # Log number of tweets
  log4r::info(logger, paste0("Found ", length(tweets_to_calculate), " tweet files to process."))


  # Create progress bar
  pbar <- progressr::progressor(along = tweets_to_calculate)
  parallelization_start_time <- Sys.time()


  gc(full = TRUE)


  # Iterate through tweets
  foreach(
    tweet_fname = tweets_to_calculate,
    .packages = pkgs,
    .inorder = FALSE,
    .errorhandling = "pass",
    .noexport = c("atm", "all_control_configs", "pbar")
  ) %dopar% {

    tweet_id <- gsub(".json.gz", "", tweet_fname)
    control_config <- all_control_configs[[tweet_id]]

    log4r::info(logger, paste0("Begining tweet ", tweet_fname, ". (Using worker ", Sys.getpid(), ")"))

    output_subdir <- fs::path(output_dir)

    if (!dir.exists(output_subdir)) {
      dir.create(output_subdir)
    }

    output_path <- fs::path(
      output_subdir,
      paste0(tools::file_path_sans_ext(tweet_fname, compression = TRUE), ".parquet")
    )


    # Load any weights that have been calculated in previous runs, or using other methods
    if (file.exists(output_path)) {
      previous_weights <- arrow::read_parquet(output_path)

      # Check if weights file contains "shuffle" column, and if not add False default
      if (!"shuffle" %in% colnames(previous_weights)) {
        previous_weights$shuffle <- FALSE
      }

      if (!"control_list" %in% colnames(previous_weights)) {
        previous_weights$control_list <- NA
      }


      # Find out if weights have already been calculated using the same parameters
      tbd <- train_backdate
      weights_for_this_run <- previous_weights %>%
        dplyr::filter(optimizer == "clarabel") %>%
        dplyr::filter(train_backdate == tbd) %>%
        dplyr::filter(donor_pool_restricted == !is.null(restricted_pool_size)) %>%
        dplyr::filter(standardized == standardize) %>%
        dplyr::filter(dev == config$dev) %>%
        dplyr::filter(time_freq == config$time_freq) %>%
        dplyr::filter(volatile_tweet_filtering == config$volatile_tweet_filtering) %>%
        dplyr::filter(pre_break_min_time == config$pre_break_min_time) %>%
        dplyr::filter(pre_break_max_time == config$pre_break_max_time) %>%
        dplyr::filter(post_break_min_time == config$post_break_min_time) %>%
        dplyr::filter(replace_calculated_when_missing == config$replace_calculated_when_missing) %>%
        dplyr::filter(missing_metric_action == config$missing_metric_action) %>%
        dplyr::filter(lambda %in% lseq) %>%
        dplyr::filter(matching_metrics == matching_metrics_entry) %>%
        dplyr::filter(shuffle == config$shuffle)

      # Filter by pool size
      if (!is.null(restricted_pool_size)) {
        weights_for_this_run <- weights_for_this_run %>%
          dplyr::filter(num_donors <= restricted_pool_size)
      }

      if ("control_list" %in% names(config)) {
        weights_for_this_run <- weights_for_this_run %>%
          dplyr::filter(control_list == config$control_list)
      } else {
        weights_for_this_run <- weights_for_this_run %>%
          dplyr::filter(is.na(control_list))
      }

      # If they have already been calculated, skip
      if (nrow(weights_for_this_run) >= length(lseq)) {
        log4r::info(logger, paste0("Weights for ", tweet_fname, " already calculated. Skipping. (Using worker ", Sys.getpid(), ")"))
        should_continue <- FALSE
      } else {
        should_continue <- TRUE
      }
    } else {
      should_continue <- TRUE
      previous_weights <- tibble::tibble()
    }

    if (should_continue) {
      # Read in the tweet file with the treatment and control data for the target metric
      log4r::info(logger, paste0("Reading metrics for tweet ", tweet_fname, ". (Using worker ", Sys.getpid(), ")"))


      tdata <- load_tweet_metrics(
        fname = tweet_fname,
        control_config = control_config,
        aut_m = author_metrics
      )


      if (length(tdata) == 0) {
        log4r::info(logger, paste0("No treatment/control data for ", tweet_fname, ". Skipping. (Using worker ", Sys.getpid(), ")"))
        should_continue <- FALSE
      } else if (length(tdata$treatment) == 0) {
        log4r::info(logger, paste0("No treatment data for ", tweet_fname, ". Skipping. (Using worker ", Sys.getpid(), ")"))
        should_continue <- FALSE
      } else if (length(tdata$control) == 0) {
        log4r::info(logger, paste0("No control data for ", tweet_fname, ". Skipping. (Using worker ", Sys.getpid(), ")"))
        should_continue <- FALSE
      }

      if (should_continue) {


        # Get all control ids
        all_control_ids <- names(tdata$control)

        n_control <- length(all_control_ids)


        # Get time between cutoff and CRH slap, to use as target
        log4r::info(logger, paste0("Cleaning and standardizing ", tweet_fname, " which has ", n_control, " potential controls. (Using worker ", Sys.getpid(), ")"))


        # Select treatment sequences for training period
        treatment_matrix <- tdata$treatment %>%
          as.matrix()

        # Select control seq for training period
        control_matrix <- tdata$control %>%
          as.matrix()


        # Make sure no elements of standardization vector are NA
        std_vector <- get_standardization_vector(control_config, treatment_sds)
        if (any(is.na(std_vector))) {
          log4r::error(logger, paste0("For ", tweet_fname, ", standardization vector contains NA. (Using worker ", Sys.getpid(), ")"))
        }

        # Standardize, if necessary
        if (standardize) {
          standardized <- standardize_X(treatment_matrix, control_matrix,
                                        sd = std_vector)
          treatment_matrix <- standardized$X1
          control_matrix <- standardized$X0
        }


        cleaned <- remove_nas_and_degenerates(treatment_matrix, control_matrix, tweet_fname)

        treatment_matrix <- cleaned$X1
        control_matrix <- cleaned$X0

        # If we are using a restricted pool, get the closest control ids
        # Otherwise, we will use all control ids
        if (!is.null(restricted_pool_size)) {
          cids <- get_closest_control_ids(control_matrix, treatment_matrix, restricted_pool_size)
          if (n_control > restricted_pool_size) {
            log4r::info(logger, paste0("For ", tweet_fname, ", restricting donor pool to ", restricted_pool_size, " closest controls. (Using worker ", Sys.getpid(), ")"))
            control_matrix <- control_matrix[, cids, drop = FALSE]
            cleaned <- remove_nas_and_degenerates(treatment_matrix, control_matrix, tweet_fname)
            treatment_matrix <- cleaned$X1
            control_matrix <- cleaned$X0
          }
        } else {
          cids <- all_control_ids
        }

        # Assign equal weight to each point in sequence
        v <- rep(1, nrow(control_matrix))

        start_time <- Sys.time()

        log4r::info(logger, "Starting optimization for tweet ", tweet_fname, ". (Using worker ", Sys.getpid(), ")")

        # Start optimization
        res <- pensynth::pensynth(
          X0 = control_matrix, X1 = treatment_matrix, v = v, standardize = FALSE, # We don't standardize in pensynth, since we already do it above
          return_solver_info = TRUE,
          lambda = lseq,
          opt_pars = clarabel::clarabel_control(
            max_iter = 1000,
            verbose = FALSE,
            tol_feas = 1e-09,
            tol_gap_abs = 1e-09,
            tol_gap_rel = 0
          )
        )


        # End optimization
        end_time <- Sys.time()
        log4r::info(logger, paste0("Completed optimization for tweet ", tweet_fname, ". Took: ", lubridate::as.duration(end_time - start_time), " (Using worker ", Sys.getpid(), ")"))

        # Get weights from object
        new_weights <- res$w_path


        rownames(new_weights) <- colnames(control_matrix)

        new_weights <- new_weights %>%
          t() %>%
          as_tibble()


        new_results <- tibble::tibble(
          weight_id = uuid::UUIDgenerate(n = nrow(new_weights)),
          optimizer = "clarabel",
          train_backdate = train_backdate,
          donor_pool_restricted = !is.null(restricted_pool_size),
          standardized = standardize,
          dev = config$dev,
          time_freq = config$time_freq,
          volatile_tweet_filtering = config$volatile_tweet_filtering,
          pre_break_min_time = config$pre_break_min_time,
          pre_break_max_time = config$pre_break_max_time,
          post_break_min_time = config$post_break_min_time,
          replace_calculated_when_missing = config$replace_calculated_when_missing,
          missing_metric_action = config$missing_metric_action,
          matching_metrics = rep(matching_metrics_entry, nrow(new_weights)),
          num_donors = length(cids),
          tweet_id = gsub(".json.gz", "", tweet_fname),
          lambda = res$lseq,
          validation_mse = res$mse_path,
          run_time = res$solve_time,
          solver_status = res$status,
          solver_n_iter = res$iter,
          shuffle = config$shuffle,
          control_list = config$control_list
        ) %>%
          dplyr::bind_cols(new_weights)


        # Save results
        if (nrow(previous_weights) > 0) {
          all_results <- dplyr::bind_rows(previous_weights, new_results)
        } else {
          all_results <- new_results
        }


        arrow::write_parquet(all_results, output_path)

        log4r::info(logger, paste0("Saved data for tweet ", tweet_fname, ". (Using worker ", Sys.getpid(), ")"))


      }
    }
    pbar(
      paste0(
        "Time Elapsed=",
        lubridate::duration(
          round(as.numeric(Sys.time() - parallelization_start_time)),
          "seconds"
        )
      )
    )

    return(NULL)

  }
}


### Perform Calculation ################################################################################################

doMC::registerDoMC(cores = NUM_WORKERS)

calc_pensynth_weights()


#
# # Format this file
# styler::style_file("src/pipeline/f_calc_weights.R")
