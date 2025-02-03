# Get data_root environment var
local_data_root <- Sys.getenv("BW_LOCAL_DATA_ROOT")
shared_data_root <- Sys.getenv("BW_SHARED_DATA_ROOT")

# Check if package is installed: if so, load it; if not, install then load it.
check_and_install <- function(package, repos = "https://cloud.r-project.org") {
  if (!require(package, character.only = TRUE)) {
    install.packages(package, repos = repos)
    library(package, character.only = TRUE)
  }
}


convert_to_hours <- function(time_string) {
  # Convert a time string to hours (e.g. '10h' -> 10 or '3600 seconds' -> 1)

  # Convert the time string to lowercase for consistency
  time_string <- tolower(time_string)

  # Define the conversion factors for different time units to hours
  conversion_factors <- c(
    "s" = 1 / 3600,
    "sec" = 1 / 3600,
    "second" = 1 / 3600,
    "seconds" = 1 / 3600,
    "m" = 1 / 60,
    "min" = 1 / 60,
    "minute" = 1 / 60,
    "minutes" = 1 / 60,
    "h" = 1,
    "hour" = 1,
    "hours" = 1,
    "d" = 24,
    "day" = 24,
    "days" = 24
  )

  # Extract the numeric value and the unit from the time string
  matches <- regmatches(time_string, regexec("([0-9.]+)\\s*(\\w+)", time_string))
  if (length(matches[[1]]) != 3) {
    stop(paste0("Unknown time format in time_string: ", time_string))
  }

  value <- as.numeric(matches[[1]][2])
  unit <- matches[[1]][3]

  # Check if the unit is in the conversion factors
  if (!unit %in% names(conversion_factors)) {
    stop("Unknown time unit")
  }

  # Convert to hours
  hours <- value * conversion_factors[[unit]]
  return(hours)
}



read_from_table <- function(fp, config) {
  if (tools::file_ext(fp) == "gz") {
    table <- jsonlite::read_json(fp)

    config_matches <- list()
    for (entry in table) {
      is_match <- TRUE
      if ("max_date" %in% names(entry)){
        entry$max_date <- lubridate::as_datetime(entry$max_date)
      }
      for (key in names(config)) {
        these_match <- identical(config[[key]], entry[[key]])
        is_match <- is_match & these_match
      }
      if (is_match) {
        config_matches <- append(config_matches, list(entry))
      }
    }

    return(config_matches)

  # Read parquet
  } else if (tools::file_ext(fp) == "parquet"){
    table <- arrow::read_parquet(fp)

    for (config_key in names(config)){
      if ("POSIXct" %in% class(config[[config_key]])){
        table <- table %>%
          dplyr::mutate(across(all_of(config_key), ~ as.numeric(.))) %>%
          dplyr::filter(!!sym(config_key) == as.numeric(config[[config_key]])) %>%
          dplyr::select(-all_of(config_key))
      } else {
      table <- table %>%
        dplyr::filter(!!sym(config_key) == config[[config_key]]) %>%
        select(-all_of(config_key))
      }
    }

    return(table)
  } else {
    stop(paste0("Unknown file extension when reading file ", fp))
  }
}
