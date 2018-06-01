#' Quickly analyse a data frame using h2o.ai
#'
#' Uses h2o.ai to quickly run a random forest, gradient boosting machine and
#'   deep learning model on an input data frame, and returns information on the
#'   performance of the models.
#'
#' @param data input data frame to be analysed
#' @param response_variable name (string) of column in the data frame that
#'   contains the response / dependent variable. If missing, defaults to "y".
#' @param analysis_mode one of either "light" (default) or "full". This determines
#'   the options that will be passed to the internal models. Light mode aims to
#'   reduce run time at the expense of accuracy, full mode is more heavyweight.
#'   Your mileage may vary.
#' @return a list, each item of which is another list containing the h2o model
#'   objects, and the h2o.performance objects extracted from those objects,
#'   respectively.
#' @export
#' @examples
#' data <- datasets::iris
#' quick_result <- quick_analyse(data = data, response_variable = "Species")
#' full_result <- quick_analyse(data = data, response_variable = "Species", analysis_mode = "full")
quick_analyse <- function(data, response_variable = "y", analysis_mode = c("light", "full")) {

  # ------ Setup -------
  # The internal functions require h2o and caret to run
  require(h2o)
  require(caret)

  # Function for extracting metrics from the models
  get_metrics <- function(model) {

    model_id <- model@model_id
    algorithm <- model@algorithm
    logloss <- h2o::h2o.logloss(model)
    rmse <- h2o::h2o.rmse(model)

    return(data.frame(list(model_id = model_id,
                           algorithm = algorithm,
                           logloss = logloss,
                           rmse = rmse)))
  }

  # Check if the data argument is a data frame, and if not, try and coerce it
  if (!is.data.frame(data)) {
    stop("The 'data' argument must be a data frame")
  }

  # Create empty lists to hold models and key metrics.
  #   In the return value, models and metrics will be nested inside result
  models <- list()
  metrics <- NULL

  # Determine analysis mode
  mode <- match.arg(analysis_mode)
  message(paste("Analysis mode: ", mode))

  # Define variables and response
  x <- setdiff(names(data), response_variable)
  message(paste("x:", x))
  y <- response_variable
  message(paste("y:", y))

  # Determine (or define) problem as binary class., multiclass, regression
  # TBC


  # Create empty list to hold h2o model params
  h2o_params <- list()


  # Checking mode and setting options appropriately
  if (mode == "full") {

    # No limit on threads or memory
    h2o_params["nthreads"] <- -1
    h2o_params["max_mem_size"] <- "100G"

    # 5-fold CV
    h2o_params["nfolds"] <- 5

    # Train / test split (no validation)
    split <- caret::createDataPartition(y = data[[y]], p = 0.75, list = FALSE)

    # Create data frames
    train <- data[split, ]
    test <- data[-split, ]
    rm(split)


  } else if (mode == "light") {

    h2o_params["nthreads"] <- 3
    h2o_params["max_mem_size"] <- "6G"

    # No cross-validation
    h2o_params["nfolds"] <- 0

    # Train/test/validation split (60:20:20)
    split <- caret::createDataPartition(y = data[[y]], p = 0.6, list = FALSE)

    # Initial train/test split
    train <- data[split, ]
    test <- data[-split, ]
    rm(split)

    # Further split test into test and validation
    valid <- caret::createDataPartition(y = test[[y]], p = 0.5, list = FALSE)
    test <- test[valid, ]
    validation <- test[-valid, ]
    rm(valid)

  }

  # Start h2o cluster and remove any previous instances
  h2o::h2o.init(nthreads = h2o_params$nthreads,
           max_mem_size = h2o_params$max_mem_size)

  h2o::h2o.removeAll()

  # Create train and test frames
  message("Creating training frame")
  h2o_params["training_frame"] <- list(h2o::as.h2o(train))
  message("Creating test frame")
  h2o_params["test_frame"] <- list(h2o::as.h2o(test))

  # Only use a validation frame for light mode
  if (mode == "full") {
    h2o_params["validation_frame"] <- list(NULL)
  } else if (mode == "light") {
    message("Creating validation frame")
    h2o_params["validation_frame"] <- list(h2o::as.h2o(validation))
  }


  # Random forest
  message("Running random forest")
  rf0 <- h2o::h2o.randomForest(model_id = "rf0",
                               training_frame = h2o_params$training_frame,
                               validation_frame = h2o_params$validation_frame,
                               x = x,
                               y = y,
                               nfolds = h2o_params$nfolds,
                               seed = 1234)

  # Add the model output to the return list
  models["rf0"] <- list(rf0)

  # Add the first row of metrics to the metrics data frame
  metrics <- get_metrics(rf0)


  # GBM
  message("Running GBM")
  gbm0 <- h2o::h2o.gbm(model_id = "gbm0",
                       training_frame = h2o_params$training_frame,
                       validation_frame = h2o_params$validation_frame,
                       x = x,
                       y = y,
                       nfolds = h2o_params$nfolds,
                       seed = 1234)

  # Store model in model list
  models["gbm0"] <- list(gbm0)

  # Store model metrics in metrics data frame
  metrics <- rbind(metrics, get_metrics(gbm0))

  # Deep learning
  message("Running deep learning")
  dl0 <- h2o::h2o.deeplearning(model_id = "dl0",
                               training_frame = h2o_params$training_frame,
                               validation_frame = h2o_params$validation_frame,
                               x = x,
                               y = y,
                               nfolds = h2o_params$nfolds,
                               seed = 1234)

  models["dl0"] <- list(dl0)

  # Store model metrics in metrics data frame
  metrics <- rbind(metrics, get_metrics(dl0))

  # Return the list of models and metrics
  result <- list(metrics = metrics, models = models)
  return(result)

}
