library(tidyverse)
library(corrr)
library(rstatix)

library(tidymodels)
library(finetune)
library(future)
library(purrr)
library(furrr)
#library(textrecipes)
library(themis)
#library(embed)
#library(tailor)
library(furrr)

library(bonsai)
library(lightgbm)
#library(xgboost)
library(ranger)
#library(betareg)
library(readr)
library(janitor)
library(lubridate)

library(memoise)
dk_cach <- memoise::cache_filesystem('./cache')
### download to local
```{r}
competition_name <- 'playground-series-s5e4'
data_path <- file.path('../input',competition_name)
target_name <- 'listening_time_minutes'
# system(paste0('kaggle competitions  download -c ', competition_name))
# unzip(paste0(competition_name,'.zip'),exdir=file.path('../input',competition_name))
# file.remove(paste0(competition_name,'.zip'))
```


### loading data

```{r}
train <-
  readr::read_csv(file.path(data_path, "train.csv"),
                  show_col_types = FALSE
  ) |>
  janitor::clean_names()
test <-
  readr::read_csv(file.path(data_path, "test.csv"),
                  show_col_types = FALSE
  ) |>
  janitor::clean_names()
submission <- readr::read_csv(file.path(data_path, "sample_submission.csv"), show_col_types = FALSE)
```

### get_outlier
```{r}



internal_get_outliers <- function(od_target){
  library(applicable)
  library(isotree)
  remove_cols <- c('id','listening_time_minutes')
  od_tr <- od_target |> select(-any_of(remove_cols))
  #od_te <- od_target |>  select(-any_of(remove_cols))
  if_mod <- apd_isolation(od_tr, ntrees = 100, nthreads = 4)
  od_score <- score(if_mod, od_tr) |>select('score_pctl')
  return(od_score)
}

get_mice_df <- function(df,dataset_id=1){
  library(mice)
  imp <- mice(df, m = 5, maxit = 5, method='rf',seed = 123,printFlag=F)
  fit <- with(imp, lm(episode_length_minutes~genre+
                        host_popularity_percentage+
                        guest_popularity_percentage+
                        publication_day+
                        publication_time+
                        number_of_ads+
                        episode_sentiment ))
  complete_df <- complete(imp, dataset_id)
  return(complete_df)
}

get_outliers <- memoise::memoise(internal_get_outliers,cache=dk_cach)

### data nest

tr<- train |>
  mutate(episode_id = parse_number(episode_title))|>
  #select(-episode_title)|>
  (\(x) bind_cols(x,get_outliers(od_target = x)))()|>
  nest(.by = podcast_name) |>slice(1)

te <- test |>
  mutate(episode_id = parse_number(episode_title))|>
  select(-episode_title)|>
  nest(.by = podcast_name)


tmp_df_raw <- tr|>unnest(data )|>select(-id, -podcast_name, -episode_title
                                    )
tmp_df_mean <-
    recipe(listening_time_minutes~., data=tmp_df_raw) |>
    step_impute_mean(all_numeric_predictors())|>
    prep() |>
  juice()

tmp_df_mice<- tmp_df_raw|>get_mice_df(dataset_id = 4)

get_glance_lm <- function(df,isplot=F){
  library(ggfortify)
  mod <- lm(listening_time_minutes~., df)
  mod|>glance()|>print()

  if(isplot==T){
    autoplot(mod)
  }
}
c(1:4)|> map_dfr(\(x) get_mice_df(tmp_df_raw, x) |>get_glance_lm())
