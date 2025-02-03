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
  "doParallel",
  "devtools",
  "uuid",
  "argparse",
  "jsonlite",
  "progressr",
  "bigmemory",
  "doFuture",
  "digest"
)

# Install any missing packages
for (pkg in pkgs) {
  check_and_install(pkg)
}

# Install pensynth
devtools::install_github("isaacOnline/pensynth", ref = "main")

handlers(global = TRUE)

# Set global options ###################################################################################################

NUM_WORKERS <- 30

parser <- argparse::ArgumentParser(description = "Calculate synthetic control weights")

parser$add_argument("--config", type = "character", required = TRUE, help = "Path to the config file")

args <- parser$parse_args()

config <- jsonlite::read_json(args$config)

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
  "use_bookmark_tweets",
  "use_backup_tweets",
  "volatile_tweet_filtering",
  "max_date",
  "train_backdate",
  "val_backdate",
  "pre_break_min_time",
  "pre_break_max_time",
  "post_break_min_time",
  "matching_metrics",
  "standardize_when_calculating_weights",
  "replace_calculated_when_missing",
  "missing_metric_action",
  "weighting_strategy",
  "include_permutations",
  "lambda",
  "restrict_donor_pool",
  "restricted_pool_size"
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

# Remove any unneeded keys
for (config_key in names(config)) {
  if (!(config_key %in% necessary_config)) {
    config <- config[!names(config) %in% config_key]
  }
}


# Define the metric_parents dictionary
metric_parents <- list(
  "replies" = "metrics",
  "quotes" = "metrics",
  "likes" = "metrics",
  "impressions" = "metrics",
  "retweets" = "metrics",
  "like_through_rate" = "metrics",
  "calculated_replies" = "calculated_replies",
  "calculated_retweets" = "calculated_retweets",
  "calculated_replies_per_impression" = "calculated_replies",
  "calculated_retweets_per_impression" = "calculated_retweets",
  "author_n_followers" = "author",
  "author_n_friends" = "author",
  "author_n_statuses" = "author",
  "rt_graph_num_nodes" = "rt_graph",
  "rt_graph_density" = "rt_graph",
  "reply_graph_num_nodes" = "reply_graph",
  "reply_graph_density" = "reply_graph",
  "reply_tree_width" = "reply_tree",
  "reply_tree_depth" = "reply_tree",
  "reply_tree_wiener_index" = "reply_tree",
  "rt_cascade_width" = "rt_cascade",
  "rt_cascade_depth" = "rt_cascade",
  "rt_cascade_wiener_index" = "rt_cascade"
)

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


# Assign equal weight to each point in sequence
if (!config$weighting_strategy %in% "unweighted") {
  stop(paste0("Unknown weighting strategy: ", config$weighting_strategy))
}

# Set max_date to date
config$max_date <- lubridate::as_datetime(config$max_date)

# Convert lambda to a vector
config$lambda <- unlist(config$lambda)

# Set input and output directories
merge_dir <- fs::path(
  local_data_root,
  intermediate_dir,
  "b_merged"
)
control_dir <- fs::path(
  local_data_root,
  intermediate_dir,
  "c_find_controls"
)
swappable_trt_dir <- fs::path(
  local_data_root,
  intermediate_dir,
  "d_find_valid_swaps"
)
permutation_dir <- fs::path(
  local_data_root,
  intermediate_dir,
  "e_permute_treatments"
)
output_dir <- fs::path(
  local_data_root,
  intermediate_dir,
  "f_calc_weights"
)
log_dir <- fs::path(
  local_data_root,
  intermediate_dir,
  "logs"
)

# Make sure output dir exists
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}

train_backdate <- -convert_to_hours(config$train_backdate)
val_backdate <- -convert_to_hours(config$val_backdate)

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

# options(future.globals.maxSize = 100 * 1024^3)

# Initialize cluster
registerDoFuture()
plan(multicore, workers = NUM_WORKERS)


# Register handlers for progress bar
handlers(global = TRUE)
handlers("progress")


## Load standard deviations to normalize by ############################################################################

matching_metrics_vector <- unlist(config$matching_metrics)
matching_metrics_entry <- paste0(matching_metrics_vector, collapse = ",")

sds_config <- config[!names(config) %in% c(
  "title",
  "weighting_strategy",
  "standardize_when_calculating_weights",
  "include_permutations",
  "val_backdate",
  "lambda",
  "restricted_pool_size",
  "restrict_donor_pool",
  "tweet_metrics",
  "backup_tweet_metrics",
  "author_metrics"
)]
sds_config$matching_metrics <- matching_metrics_entry

treatment_sds <- read_from_table(fs::path(control_dir, "metric_sds.parquet"), sds_config)

## Read author data ####################################################################################################

log4r::info(logger, "Loading author data.")
author_data_path <- fs::path(local_data_root, "cn_effect_input", "author_metrics.csv")

author_metrics <- readr::read_csv(
  author_data_path,
  col_types = list(
    "tweet_id" = readr::col_character(),
    "author_id" = readr::col_character()
  )
)

log4r::info(logger, "Author data has been loaded.")

## Read control configurations #########################################################################################

log4r::info(logger, "Loading synthetic control configuration files")

read_control_configs <- function() {
  # Get the name for all the treatment tweet files
  all_tweet_fnames <- list.files(control_dir, pattern = "*.json.gz", full.names = F)

  # Filter to exclude any config items that weren't needed/included in input data
  input_config <- config[!names(config) %in% c(
    "title",
    "weighting_strategy",
    "standardize_when_calculating_weights",
    "include_permutations",
    "val_backdate",
    "lambda",
    "restricted_pool_size",
    "restrict_donor_pool",
    "tweet_metrics",
    "backup_tweet_metrics",
    "author_metrics"
  )]

  pbar <- progressr::progressor(along = all_tweet_fnames)

  config_read_start_time <- Sys.time()

  all_control_configs <- foreach(
    fname = all_tweet_fnames,
    .packages = pkgs,
    .export = c("read_from_table", "input_config", "control_dir", "logger", "config_read_start_time")
  ) %dopar% {
    tryCatch(
      {
        tweet_id <- sub(".json.gz", "", fname)
        tweet_config <- read_from_table(fs::path(
          control_dir, fname
        ), input_config)
      },
      error = function(e) {
        log4r::error(logger, "Error processing file:", fname, "\n", e$message)
      }
    )

    pbar(
      paste0(
        "Reading control configs. tweet_id=",
        tweet_id,
        "; Time Elapsed=",
        lubridate::as.duration(
          Sys.time() - config_read_start_time
        )
      )
    )

    if (length(tweet_config) != 1) {
      log4r::warn(logger, paste0("Config of length ", length(tweet_config), " for file: ", fname))
      return(NULL)
    }


    return(setNames(tweet_config, tweet_id))
  }

  all_control_configs <- Filter(Negate(is.null), all_control_configs)

  return(do.call(c, all_control_configs))
}

all_control_configs <- read_control_configs()

log4r::info(logger, "Synthetic control configuration files have been loaded")


## Read tweet_metrics ####################################################################################################

log4r::info(logger, "Loading tweet metrics.")


as.big.matrix.list <- function(x, ...) {
  # Get date columns
  date_cols <- dplyr::select(x, where(lubridate::is.timepoint)) %>% colnames()

  # Convert to seconds
  for (col in date_cols) {
    x[[col]] <- x[[col]] %>%
      as.numeric() %>%
      as.integer()
  }

  # Get difftime columns
  diff_cols <- dplyr::select(x, where(lubridate::is.difftime)) %>% colnames()

  # Convert to seconds
  for (col in diff_cols) {
    x[[col]] <- x[[col]] %>%
      as.numeric() %>%
      as.integer()
  }

  # Get character columns, so we can save factor mappings
  char_cols <- dplyr::select(x, where(is.character)) %>% colnames()

  # Convert to factors
  for (col in char_cols) {
    x[[col]] <- x[[col]] %>% as.factor()
  }

  # Save levels
  factor_levels <- lapply(x[char_cols], levels)

  # Now convert to numeric
  for (col in char_cols) {
    x[[col]] <- x[[col]] %>% as.numeric()
  }

  return(
    list(
      date_cols = date_cols,
      diff_cols = diff_cols,
      char_cols = char_cols,
      factor_levels = factor_levels,
      big.matrix = as.big.matrix(as.data.frame(x), shared = TRUE, ...)
    )
  )
}


as.tibble.from.big.matrix.list <- function(x) {
  # Unload date cols
  date_cols <- x$date_cols

  # Unload difftimes
  diff_cols <- x$diff_cols

  # Unload character cols
  char_cols <- x$char_cols

  # Unload factor levels
  factor_levels <- x$factor_levels

  # Convert back to matrix
  x <- bigmemory::as.matrix(x$big.matrix)

  # Convert back to tibble
  x <- tibble::as_tibble(x)

  # Convert date cols back
  for (col in date_cols) {
    x[[col]] <- x[[col]] %>% as_datetime()
  }

  # Convert difftimes back
  for (col in diff_cols) {
    x[[col]] <- x[[col]] %>% lubridate::as.difftime(units = "secs")
  }

  # Convert character cols back
  for (col in char_cols) {
    col_factor_levels <- factor_levels[[col]]
    names(col_factor_levels) <- seq_along(col_factor_levels)
    x[[col]] <- x[[col]] %>% purrr::map_chr(~ col_factor_levels[.x])
  }

  return(x)
}

read_tweet_data <- function(tweet_id, config_to_read) {
  tweet_data <- read_from_table(
    fs::path(merge_dir, paste0(tweet_id, ".parquet")),
    config_to_read[names(config_to_read) %in% c(
      "dev", "use_bookmark_tweets", "use_backup_tweets", "volatile_tweet_filtering", "time_freq", "max_date"
    )]
  )

  tweet_data <- tryCatch(
    {
      tweet_data$calculated_retweets_per_impression <- tweet_data$calculated_retweets / tweet_data$impressions
      tweet_data
    },
    error = function(e) {
      tweet_data$calculated_retweets_per_impression <- NA
      tweet_data
    }
  )

  tweet_data <- tryCatch(
    {
      tweet_data$calculated_replies_per_impression <- tweet_data$calculated_replies / tweet_data$impressions
      tweet_data
    },
    error = function(e) {
      tweet_data$calculated_replies_per_impression <- NA
      tweet_data
    }
  )

  return(tweet_data)
}

read_all_metrics <- function() {
  all_tweet_metric_files <- list.files(merge_dir, pattern = "*.parquet", full.names = TRUE)
  pbar <- progressr::progressor(along = all_tweet_metric_files)

  metric_read_start_time <- Sys.time()

  all_tweet_metrics <- foreach(
    fname = all_tweet_metric_files,
    .packages = pkgs,
    .export = c("read_from_table", "logger", "metric_read_start_time", "pbar", "config")
  ) %do% {
    tweet_id <- sub(".parquet", "", basename(fname))

    tweet_metrics <- read_tweet_data(basename(fname), config_to_read)

    # Find which columns are present
    cols_to_keep <- c(
      "tweet_id", "timestamp", "created_at", "time_since_publication", "note_0_time_since_first_crh",
      config$tweet_metrics %>% unlist(), config$backup_tweet_metrics %>% unlist()
    )

    cols_to_keep <- cols_to_keep[cols_to_keep %in% colnames(tweet_metrics)]

    tweet_metrics <- tweet_metrics %>%
      dplyr::select(all_of(cols_to_keep)) %>%
      as.big.matrix.list()


    pbar(
      paste0(
        "Reading tweet metrics. tweet_id=",
        tweet_id,
        "; Time Elapsed=",
        lubridate::as.duration(
          Sys.time() - metric_read_start_time
        )
      )
    )

    # Make a named list where the name is the tweet_id, and the value is the big.matrix.list
    return(setNames(list(tweet_metrics), tweet_id))
  }



  # Bind list of lists into a single list
  all_tweet_metrics <- do.call(c, all_tweet_metrics)

  return(all_tweet_metrics)
}

## Helper Functions ####################################################################################################
standardize_X <- function(X1, X0, sd = NULL, mu = NULL) {
  # Function to scale the X matrices in synthetic control
  if (!is.null(mu)) {
    return(list(X1 = (X1 - mu) / sd, X0 = (X0 - mu) / sd))
  } else {
    return(list(X1 = X1 / sd, X0 = X0 / sd))
  }
}

get_control_ids <- function(tweet_df, swap_df, assignment_matrix) {
  # Get control IDs in observed data
  all_control_ids <- colnames(tweet_df) %>%
    setdiff(c(
      "time_since_publication",
      "treatment",
      "treatment_note_0_time_since_first_crh",
      "treatment_any_crh"
    ))

  # Get control IDs according to this assignment matrix,
  # if we're permuting
  if (!is.null(assignment_matrix)) {
    # Get the swappable treatments
    swappable_trt_ids <- colnames(swap_df) %>%
      setdiff(c(
        "time_since_publication",
        "treatment",
        "treatment_note_0_time_since_first_crh",
        "treatment_any_crh"
      ))

    # Add these as potential controls
    all_control_ids <- union(all_control_ids, swappable_trt_ids)


    # Find the tweets that have been assigned as control in this
    # assignment matrix
    potential_controls <- assignment_matrix %>%
      dplyr::select(-c(treatment_tweet)) %>%
      dplyr::summarize_all(sum) %>%
      select(where(~ .x == 0)) %>%
      colnames()

    # Intersect to find all controls in this assignment matrix
    # that are valid swaps for the specific tweet
    all_control_ids <- intersect(potential_controls, all_control_ids)
  }

  return(all_control_ids)
}


get_treatment_col_name <- function(assignment_matrix, tweet_id) {
  # If we're using the real assignment vector, then just return "treatment"
  if (is.null(assignment_matrix)) {
    treatment_col_name <- "treatment"
  } else {
    # Otherwise, we need to return the id of the treatment tweet that is
    # being swapped in and treated as treatment here
    treatment_col_name <- assignment_matrix %>%
      dplyr::filter(treatment_tweet == tweet_id) %>%
      dplyr::select(-c(treatment_tweet)) %>%
      select(where(~ .x != 0)) %>%
      colnames()
  }

  return(treatment_col_name)
}

get_closest_control_ids <- function(X0, X1, num_controls) {
  # Standardize, if necessary
  colnames(X1) <- "treatment"

  # Calculate euclidean distance between treatment_likes and control_likes
  distances <- dist(cbind(X1, X0) %>% t(), method = "euclidean") %>%
    as.matrix() %>%
    as_tibble()

  distances$tweet_id <- colnames(distances)

  # Get the closest controls
  control_seqs <- distances %>%
    dplyr::arrange(treatment) %>%
    dplyr::select(tweet_id) %>%
    dplyr::slice(2:(num_controls + 1)) %>%
    dplyr::pull()

  return(control_seqs)
}

get_time_range_for_metric <- function(metric, matching_time_range) {
  for (tr in matching_time_range) {
    if (tr[[1]] == metric) {
      return(tr)
    }
  }
}

get_author_metrics <- function(tid, metric) {
  # Get an author metric, e.g. author_n_followers

  # Filter author metrics dataframe
  author_metrics %>%
    # Filter to the tweet ID requested
    dplyr::filter(
      tweet_id == !!tid
    ) %>%
    # Get the metric requested
    dplyr::pull(!!sym(metric))
}

get_author_sd <- function(author_metric) {
  # Get the standard deviation of an author metric in the treatment sample for this config, e.g. author_n_followers

  # Filter author metrics dataframe
  treatment_sds %>%
    # Filter to the tweet ID requested
    dplyr::filter(
      metric == !!author_metric
    ) %>%
    # Get the metric requested
    dplyr::pull("sd")
}

unload_metrics <- function(tweet_id, tweet_df, tweet_metadata, unloading_standardized = FALSE) {
  # Get the metrics for a tweet into a list
  # Can be used either to unload metrics for a tweet, or to unload the standard deviations for treatment tweets, such that they
  # are in the same order as the metrics for the treatment tweet

  # Create list to save into
  matching_dfs <- list()

  # Iterate through the metrics that this treatment tweet can use for matching
  for (metric in unlist(tweet_metadata$metrics_present_for_tweet)) {
    # Check if this is a tweet metric that changes over time
    if ((metric %in% tweet_metadata$tweet_metrics) | (metric %in% tweet_metadata$backup_tweet_metrics)) {
      # If so, get the start and end time that we'll be using for matching
      time_ranges <- get_time_range_for_metric(metric, tweet_metadata$matching_timestamps)[[2]] %>% unlist()

      # Check if we should be using a backup metric
      if (unloading_standardized & ((paste0("calculated_", metric) %in% tweet_metadata$tweet_metrics)) ){
        metric <- paste0("calculated_", metric)
      }

      # Filter to the requested start/end times
      matching_dfs[[metric]] <- tweet_df[[metric]][
        (tweet_df[["time_since_publication"]] %in% time_ranges)
      ]
    } else {
      # If this is an author-level metric, retrieve it
      if (unloading_standardized) {
        matching_dfs[[metric]] <- get_author_sd(metric)
      } else {
        matching_dfs[[metric]] <- get_author_metrics(tweet_id, metric)
      }
    }
  }
  return(matching_dfs)
}


get_standardization_vector <- function(tweet_id) {
  # Read the treatment tweet's metrics over time from disk,
  # so that we can map time-since-publication (which we match on, and use for making sure
  # metrics are in the right order) onto time-to-slap (which is what we standardize by)
  treatment_tweet <- read_tweet_data(tweet_id, config)

  # Pivot SDs
  sds <- treatment_tweet %>%
    dplyr::select("note_0_time_since_first_crh", "time_since_publication") %>%
    merge(treatment_sds, on = "note_0_time_since_first_crh") %>%
    tidyr::pivot_wider(
      id_cols = "time_since_publication",
      names_from = "metric",
      values_from = "sd"
    )

  tweet_metadata <- all_control_configs[[tweet_id]]

  sd_vec <- unload_metrics(tweet_id, sds, tweet_metadata, TRUE) %>%
    unlist() %>%
    as.numeric()

  return(sd_vec)
}


read_tweet_metrics <- function(fname, fdir, config_to_read, logger) {
  # Get the ID for this tweet based on its file name
  tweet_id <- sub(".json.gz", "", fname)

  # Read the matching parameters from disk (e.g. what metrics to match on)

  tryCatch(
    {
      tweet_metadata <- all_control_configs[[tweet_id]]
    },
    error = function(e) {
      log4r::error(logger, "Error processing file:", fname, "\n", e$message)
      return(list())
    }
  )

  # Read data the treatment tweet's metrics over time from disk
  treatment_tweet <- read_tweet_data(sub(".json.gz", "", fname), config_to_read)

  treatment_metrics <- unload_metrics(tweet_id, treatment_tweet, tweet_metadata)

  # Get the list of control tweet IDs that we'll be using
  control_ids <- unlist(tweet_metadata$control_tweet_ids)

  control_metrics <- list()
  for (control_id in control_ids) {
    control_tweet <- read_tweet_data(control_id, config_to_read)
    control_metrics[[control_id]] <- unload_metrics(control_id, control_tweet, tweet_metadata)
  }

  return(list(treatment = treatment_metrics, control = control_metrics))
}

remove_nas_and_degenerates <- function(X1_to_clean, X0_to_clean, tweet_fname) {
  # Filter out rows with NA values
  matching_variables_to_remove <- (apply(X0_to_clean, 1, function(x) any(is.na(x))) | apply(X1_to_clean, 1, function(x) any(is.na(x))))
  cleaned_X1 <- X1_to_clean[!matching_variables_to_remove, ] %>% as.matrix()
  cleaned_X0 <- X0_to_clean[!matching_variables_to_remove, ]

  if (sum(matching_variables_to_remove) > 0) {
    log4r::error(logger, paste0(
      "For ", tweet_fname, ", removed ", sum(matching_variables_to_remove),
      " metric time points from matching sequence due to NA values."
    ))
  }

  # Filter out degenerate rows
  matching_variables_to_remove <- apply(cleaned_X0, 1, function(x) sd(x) == 0)
  cleaned_X1 <- cleaned_X1[!matching_variables_to_remove, ] %>% as.matrix()
  cleaned_X0 <- cleaned_X0[!matching_variables_to_remove, ]

  if (sum(matching_variables_to_remove) > 0) {
    log4r::warn(logger, paste0(
      "For ", tweet_fname, ", removed ", sum(matching_variables_to_remove),
      " metric time points due to the variable being identical across all donors."
    ))
  }

  # Compute hashes for each post to identify duplicates
  post_hashes <- apply(cleaned_X0, 2, digest::digest)

  # Find unique hashes and group duplicate posts
  hash_groups <- split(seq_along(post_hashes), post_hashes)

  # Initialize vectors to keep track of posts to keep and drop
  posts_to_keep <- integer()
  posts_to_drop <- integer()

  # Initialize list to store logs for duplicates
  duplicate_logs <- list()

  # Set a seed for reproducibility
  set.seed(digest::digest2int(tweet_fname))

  # Get the column names of cleaned_X0
  col_names <- colnames(cleaned_X0)

  # Loop through each group of duplicates
  for (hash in names(hash_groups)) {
    indices <- hash_groups[[hash]]

    if (length(indices) > 1) {
      # Randomly select one post to keep
      index_to_keep <- sample(indices, 1)
      indices_to_drop <- setdiff(indices, index_to_keep)

      # Update the lists of posts to keep and drop
      posts_to_keep <- c(posts_to_keep, index_to_keep)
      posts_to_drop <- c(posts_to_drop, indices_to_drop)

      # Log the duplicates
      duplicate_logs[[hash]] <- list(
        duplicate_posts = indices,
        kept = index_to_keep,
        dropped = indices_to_drop
      )
    } else {
      # Only one post with this hash; keep it
      posts_to_keep <- c(posts_to_keep, indices)
    }
  }

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
      duplicate_post_names <- col_names[dup_info$duplicate_posts]
      kept_post_name <- col_names[dup_info$kept]
      dropped_post_names <- col_names[dup_info$dropped]

      log4r::warn(logger, paste0(
        "Duplicate posts with hash ", hash,  " for post ", tweet_id, ": ",
        paste(duplicate_post_names, collapse = ", "),
        ". Kept post ", kept_post_name,
        ", dropped posts ", paste(dropped_post_names, collapse = ", "), "."
      ))
    }
  }

  return(list(X1 = cleaned_X1, X0 = cleaned_X0))
}

## Main function for Weight Calculation ################################################################################
calc_pensynth_weights <- function(permutation = NULL) {
  # Read covariates to match on
  matching_metrics_vector <- unlist(config$matching_metrics)
  matching_metrics_entry <- paste0(matching_metrics_vector, collapse = ",")

  # set standardization
  standardize <- config$standardize_when_calculating_weights

  # Get lambda sequence
  lseq <- config$lambda

  # Get restricted_pool_size
  if (config$restrict_donor_pool) {
    restricted_pool_size <- config$restricted_pool_size
  } else {
    restricted_pool_size <- NULL
  }

  log4r::info(logger, paste0(
    "Calculating weights with penalization. ",
    "Standardized: ", standardize, ". Restricted pool size: ", restricted_pool_size
  ))

  if (!is.null(permutation)) {
    # If treatment assignment matrix has been shuffled, log this
    log4r::info(logger, paste0(
      "Calculating weights for shuffled treatment assignment matrix ", permutation
    ))

    # Load the shuffled assignment matrix
    assignment_matrix <- read_parquet(
      fs::path(
        permutation_dir,
        paste0("assignment_matrix_", permutation, ".parquet")
      ),
      as_data_frame = TRUE
    )

    # Get rownames
    assignment_matrix <- assignment_matrix %>%
      dplyr::rename(treatment_tweet = `__index_level_0__`)
  } else {
    # Create null assignment treatment matrix, as it isn't needed if it has not been shuffled
    assignment_matrix <- NULL
  }

  # Get the name for all the treatment tweet files
  all_tweet_ids <- names(all_control_configs)

  # Filter to exclude any config items that weren't needed/included in input data
  input_config <- config[!names(config) %in% c(
    "title",
    "weighting_strategy",
    "standardize_when_calculating_weights",
    "include_permutations",
    "val_backdate",
    "lambda",
    "restricted_pool_size",
    "restrict_donor_pool",
    "author_metrics",
    "tweet_metrics",
    "backup_tweet_metrics"
  )]


  tweets_to_calculate <- list()
  for (tweet_id in all_tweet_ids) {
    fname <- paste0(tweet_id, ".json.gz")

    tweet_data <- all_control_configs[[tweet_id]]
    if (tweet_data$use_tweet) {
      tweets_to_calculate <- append(tweets_to_calculate, fname)
    }
  }
  # Log number of tweets
  log4r::info(logger, paste0("Found ", length(tweets_to_calculate), " tweet files to process."))

  # Create progress bar
  pbar <- progressr::progressor(along = tweets_to_calculate)
  parallelization_start_time <- Sys.time()

  # Iterate through tweets
  foreach(
    tweet_fname = tweets_to_calculate,
    .packages = pkgs,
    .export = c(
      "output_dir", "train_backdate", "val_backdate", "logger",
      "get_closest_control_ids",
      "control_dir", "merge_dir", "swappable_trt_dir", "permutation_dir", "standardize_X",
      "get_control_ids", "get_treatment_col_name", "author_metrics", "get_time_range_for_metric", "get_author_metrics",
      "unload_metrics", "read_tweet_metrics", "read_from_table", "parallelization_start_time",
      "all_control_configs", "read_control_configs"
    ),
    .inorder = FALSE,
    .errorhandling = "pass"
  ) %dopar% {
    tweet_id <- sub(".json.gz", "", tweet_fname)

    output_subdir <- fs::path(
      output_dir,
      ifelse(is.null(permutation), "", paste0("assignment_", permutation))
    )

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

      # Find out if weights have already been calculated using the same parameters
      tbd <- train_backdate
      vbd <- val_backdate
      weights_for_this_run <- previous_weights %>%
        dplyr::filter(optimizer == "clarabel") %>%
        dplyr::filter(train_backdate == tbd) %>%
        dplyr::filter(val_backdate == vbd) %>%
        dplyr::filter(donor_pool_restricted == !is.null(restricted_pool_size)) %>%
        dplyr::filter(standardized == standardize) %>%
        dplyr::filter(dev == config$dev) %>%
        dplyr::filter(time_freq == config$time_freq) %>%
        dplyr::filter(max_date == config$max_date) %>%
        dplyr::filter(use_backup_tweets == config$use_backup_tweets) %>%
        dplyr::filter(use_bookmark_tweets == config$use_bookmark_tweets) %>%
        dplyr::filter(volatile_tweet_filtering == config$volatile_tweet_filtering) %>%
        dplyr::filter(pre_break_min_time == config$pre_break_min_time) %>%
        dplyr::filter(pre_break_max_time == config$pre_break_max_time) %>%
        dplyr::filter(post_break_min_time == config$post_break_min_time) %>%
        dplyr::filter(replace_calculated_when_missing == config$replace_calculated_when_missing) %>%
        dplyr::filter(missing_metric_action == config$missing_metric_action) %>%
        dplyr::filter(lambda %in% lseq) %>%
        dplyr::filter(matching_metrics == matching_metrics_entry)

      # Filter by pool size
      if (!is.null(restricted_pool_size)) {
        weights_for_this_run <- weights_for_this_run %>%
          dplyr::filter(num_donors <= restricted_pool_size)
      }


      # If they have already been calculated, skip
      if (nrow(weights_for_this_run) >= length(lseq)) {
        log4r::info(logger, paste0("Weights for ", tweet_fname, " already calculated. Skipping."))
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
      tweet_data <- read_tweet_metrics(
        fname = tweet_fname,
        fdir = control_dir,
        config_to_read = input_config,
        logger = logger
      )
      if (length(tweet_data) == 0) {
        should_continue <- FALSE
      }

      if (should_continue) {
        # Read in the alternate treatments
        if (!is.null(permutation)) {
          swap_data <- read_tweet_metrics(
            tweet_fname,
            swappable_trt_dir,
            input_config
          )
        }

        # Get all control ids
        all_control_ids <- names(tweet_data$control)

        n_control <- length(all_control_ids)


        if (!is.null(permutation)) {
          # Now, for the permuations, join the observed controls and observed swappable treatments together
          tweet_data <- tweet_data %>%
            dplyr::inner_join(swap_df)

          # Do the same for the covariates
          for (cov in matching_metrics_vector) {
            covariates[[cov]] <- covariates[[cov]] %>%
              dplyr::inner_join(swap_trt_covariates[[cov]])
          }
        }


        # Get time between cutoff and CRH slap, to use as target
        log4r::info(logger, paste0("Starting on ", tweet_fname, " which has ", n_control, " potential controls"))


        # Select treatment sequences for training period
        X1 <- tweet_data$treatment %>%
          unlist() %>%
          as.matrix()

        # Select control seq for training period
        X0 <- sapply(tweet_data$control, function(x) {
          x %>%
            unlist() %>%
            as.matrix()
        }) %>%
          bind_cols() %>%
          as.matrix()

        # Standardize, if necessary
        if (standardize) {
          standardized <- standardize_X(X1, X0, sd = get_standardization_vector(tweet_id))
          X1 <- standardized$X1
          X0 <- standardized$X0
        }

        cleaned <- remove_nas_and_degenerates(X1, X0, tweet_fname)

        X1 <- cleaned$X1
        X0 <- cleaned$X0

        # If we are using a restricted pool, get the closest control ids
        # Otherwise, we will use all control ids
        if (!is.null(restricted_pool_size)) {
          control_ids <- get_closest_control_ids(X0, X1, restricted_pool_size)
          log4r::info(logger, paste0("For ", tweet_fname, ", restricting donor pool to ", restricted_pool_size, " closest controls."))
          X0 <- X0 %>%
            tibble::as_tibble() %>%
            dplyr::select(all_of(control_ids)) %>%
            as.matrix()

          cleaned <- remove_nas_and_degenerates(X1, X0, tweet_fname)
          X1 <- cleaned$X1
          X0 <- cleaned$X0
        } else {
          control_ids <- all_control_ids
        }



        # Assign equal weight to each point in sequence
        if (config$weighting_strategy == "unweighted") {
          v <- rep(1, nrow(X0))
        } else {
          stop(paste0("Unknown weighting strategy: ", config$weighting_strategy))
        }

        # Run penalized synthetic control
        # estimate lambda using pre-intervention timeseries MSE
        start_time <- Sys.time()

        log4r::info(logger, "Starting optimization for tweet ", tweet_fname, ".")



        if (train_backdate != val_backdate) {
          validation <- tweet_df %>%
            dplyr::filter(
              treatment_note_0_time_since_first_crh <
                lubridate::duration(val_backdate, "hours")
            ) %>%
            filter(
              treatment_note_0_time_since_first_crh >= lubridate::duration(train_backdate, "hours")
            )

          # Select donor sequences for validation period
          Z0 <- validation %>%
            dplyr::select(!!(control_ids)) %>%
            as.matrix()

          # Select treatment seq for validtion period
          Z1 <- validation %>%
            dplyr::select(
              !!get_treatment_col_name(assignment_matrix, gsub(".parquet", "", tweet_fname)),
            ) %>%
            as.matrix()

          # Start optimization
          res <- pensynth::cv_pensynth(
            X0 = X0, X1 = X1, Z0 = Z0, Z1 = Z1, v = v, standardize = FALSE, # We don't standardize in pensynth, since we already do it here
            return_solver_info = TRUE,
            lseq = lseq,
            opt_pars = clarabel::clarabel_control(
              max_iter = 1000,
              verbose = FALSE,
              tol_feas = 1e-09,
              tol_gap_abs = 1e-09,
              tol_gap_rel = 0
            )
          )
        } else {
          # Start optimization
          res <- pensynth::pensynth(
            X0 = X0, X1 = X1, v = v, standardize = FALSE, # We don't standardize in pensynth, since we already do it here
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
        }


        # End optimization
        end_time <- Sys.time()
        log4r::info(logger, paste0("Completed optimization for tweet ", tweet_fname, ". Took: ", lubridate::as.duration(end_time - start_time)))

        # Print best lambda
        log4r::info(logger, paste0("Best lambda: ", res$l_opt))

        # Get weights from object
        new_weights <- res$w_path
        rownames(new_weights) <- colnames(X0)
        new_weights <- new_weights %>%
          t() %>%
          as_tibble()

        new_results <- tibble::tibble(
          weight_id = uuid::UUIDgenerate(n = nrow(new_weights)),
          optimizer = "clarabel",
          train_backdate = train_backdate,
          val_backdate = val_backdate,
          donor_pool_restricted = !is.null(restricted_pool_size),
          standardized = standardize,
          dev = config$dev,
          time_freq = config$time_freq,
          max_date = config$max_date,
          use_backup_tweets = config$use_backup_tweets,
          use_bookmark_tweets = config$use_bookmark_tweets,
          volatile_tweet_filtering = config$volatile_tweet_filtering,
          pre_break_min_time = config$pre_break_min_time,
          pre_break_max_time = config$pre_break_max_time,
          post_break_min_time = config$post_break_min_time,
          replace_calculated_when_missing = config$replace_calculated_when_missing,
          missing_metric_action = config$missing_metric_action,
          matching_metrics = rep(matching_metrics_entry, nrow(new_weights)),
          num_donors = length(control_ids),
          tweet_id = ifelse(is.null(permutation),
            gsub(".json.gz", "", tweet_fname),
            get_treatment_col_name(assignment_matrix, gsub("json.gz", "", tweet_fname))
          ),
          lambda = res$lseq,
          validation_mse = res$mse_path,
          run_time = res$solve_time,
          solver_status = res$status,
          solver_n_iter = res$iter
        ) %>%
          dplyr::bind_cols(new_weights)


        # Save results
        if (nrow(previous_weights) > 0) {
          all_results <- dplyr::bind_rows(previous_weights, new_results)
        } else {
          all_results <- new_results
        }

        arrow::write_parquet(all_results, output_path)

        pbar(
          paste0(
            "tweet_fname=",
            tweet_fname,
            "; Time Elapsed=",
            lubridate::as.duration(
              Sys.time() - parallelization_start_time
            )
          )
        )
      }
    }
  }
}


### Perform Calculation ################################################################################################
calc_pensynth_weights(
  permutation = NULL
)


if (config$include_permutations) {
  # Run weight calculation
  for (permutation in 0:999) {
    calc_pensynth_weights(
      permutation = permutation
    )
  }
}

#
# # Format this file
# styler::style_file("src/pipeline/f_calc_weights.R")
