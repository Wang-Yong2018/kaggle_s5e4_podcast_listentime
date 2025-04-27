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
library(mgcv)

library(readr)
library(janitor)
library(lubridate)

library(memoise)
dk_cach <- memoise::cache_filesystem('./cache')
### download to local
competition_name <- 'playground-series-s5e4'
data_path <- file.path('../input',competition_name)
target_name <- 'listening_time_minutes'
# system(paste0('kaggle competitions  download -c ', competition_name))
# unzip(paste0(competition_name,'.zip'),exdir=file.path('../input',competition_name))
# file.remove(paste0(competition_name,'.zip'))


### loading data

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



internal_get_outliers <- function(od_target){
  library(applicable)
  library(isotree)
  remove_cols <- c('id','listening_time_minutes')
  od_tr <- od_target |> dplyr::select(-any_of(remove_cols))
  #od_te <- od_target |>  select(-any_of(remove_cols))
  if_mod <- apd_isolation(od_tr, ntrees = 100, nthreads = 4)
  od_score <- score(if_mod, od_tr) |>dplyr::select('score_pctl')
  return(od_score)
}

internal_get_mice_df <- function(df,dataset_id=1,with_y=FALSE){
  library(mice)
  y <-df|>pluck('listening_time_minutes')

  if (with_y ==FALSE){
    df <- df|>dplyr::select(-listening_time_minutes)
  }
  imp <- mice(df, m = 5, maxit = 5, method='rf',seed = 123,printFlag=F)
  fit <- with(imp, lm(episode_length_minutes~genre+
                        host_popularity_percentage+
                        guest_popularity_percentage+
                        publication_day+
                        publication_time+
                        number_of_ads+
                        episode_sentiment ))
  complete_df <- complete(imp, dataset_id)

  complete_df['listening_time_minutes'] <- y # to assign y value again for case of without y missing value fill

  return(complete_df)
}

get_mice_df <- memoise::memoise(internal_get_mice_df,cache=dk_cach)
get_outliers <- memoise::memoise(internal_get_outliers,cache=dk_cach)

### data nest

tr<- train |>
  mutate(episode_id = parse_number(episode_title))|>
  #select(-episode_title)|>
  (\(x) bind_cols(x,get_outliers(od_target = x)))()|>
  nest(.by = podcast_name) |>slice(1)

te <- test |>
  mutate(episode_id = parse_number(episode_title))|>
  dplyr::select(-episode_title)|>
  nest(.by = podcast_name)


tmp_df_raw <- tr|>unnest(data )|>dplyr::select(-id, -podcast_name, -episode_title
                                    )
tmp_df_mean <-
    recipe(listening_time_minutes~., data=tmp_df_raw) |>
    step_impute_mean(all_numeric_predictors())|>
    prep() |>
  juice()
tmp_df_median <-
  recipe(listening_time_minutes~., data=tmp_df_raw) |>
  step_impute_median(all_numeric_predictors())|>
  prep() |>
  juice()

tmp_df_knn5 <-
  recipe(listening_time_minutes~., data=tmp_df_raw) |>
  step_impute_knn(all_numeric_predictors(),
                  neighbors =5,
                  impute_with = vars(all_predictors()))|>
  prep() |>
  juice()

    tmp_df_linear <-
      recipe(listening_time_minutes~., data=tmp_df_raw) |>
      # update_role('id',new_role = 'ID')|>
      step_novel(all_nominal_predictors()) |>
      step_unknown(all_nominal_predictors()) |>
      step_other(all_nominal_predictors()) |>
      step_dummy(all_nominal_predictors()) |> # One-hot encode categorical vars
      step_interact(terms = ~ starts_with("publication_day"):starts_with('publication_time'))|>
      step_interact(terms = ~ starts_with("genre"):starts_with('episode_sentiment'))|>
      step_zv(all_predictors()) %>%               # 移除零方差变量
      step_corr(all_numeric_predictors(), threshold = 0.9) %>% # 消除高相关性
       step_impute_linear(all_numeric_predictors(),
                          impute_with =vars(starts_with("genre"),
                                            host_popularity_percentage,
                                            starts_with("publication_"),
                                            starts_with("episode_sentiment"),
                                            episode_id,
                                            number_of_ads))|>
      prep()|>juice()


tmp_df_mice_woy<- tmp_df_raw|>get_mice_df(dataset_id = 4,with_y = FALSE)
tmp_df_mice_wy<- tmp_df_raw|>get_mice_df(dataset_id = 4,with_y = TRUE)

get_glance_lm <- function(df,degree=5, isplot=F){
  library(ggfortify)
  lm_eng <-linear_reg() |>
    set_mode("regression") |>
    set_engine("lm")
  poisson_glm<- poisson_reg() %>%
    set_engine("glm",
               family = poisson(link = "log")  # 泊松分布+对数链接
    ) %>%
    set_mode("regression")

  rcp <- recipe(listening_time_minutes~., data = df)|>
    step_normalize(all_numeric_predictors(), -all_outcomes()) %>%  # 标准化数值变量
    step_ns(episode_length_minutes, host_popularity_percentage, guest_popularity_percentage,deg_free = degree)|>
    step_dummy(all_nominal_predictors())|>
    step_zv(all_predictors())|>
    step_corr(all_predictors())

  #lm_eng <- parsnip::linear_reg()+set_mode('regression') + set_engine('lm')
  fit_wf <- workflow() |>
    add_recipe(rcp)|>
    add_model(lm_eng)|>
    fit(df)

  mod <- fit_wf|>extract_fit_engine()


  if(isplot==T){
    autoplot(mod)
  }
  return(mod|>glance())
}
df_list <- list(m_woy=tmp_df_mice_woy,
                m_wy = tmp_df_mice_wy,
                mean= tmp_df_mean,
                median = tmp_df_median,
                knn5=tmp_df_knn5,
                linear=tmp_df_linear

)

df_list |>
  future_map_dfr(\(x) get_glance_lm(x|>mutate_if(is.numeric, ~as.integer(round(.x))),degree =10),.id = 'source') |>
  arrange(AIC)


get_glance_gam <- function(df,isplot=F){

  rcp <- recipe(listening_time_minutes~., data = df)|>
    step_normalize(all_numeric(), -all_outcomes()) %>%  # 标准化数值变量
    step_dummy(all_nominal_predictors())|>
    step_zv(all_predictors())|>
    step_corr(all_predictors())
  # 3. 模型定义

#
#     gen_additive_mod(engine="mgcv") |># add selective inference
#     set_mode("regression") |>
#     set_engine("gam", # or gam, bam allows for larger datasets
#                method = "fREML", #recommended parameter for fitting the model
#                discrete=TRUE,
# #               nthreads = parallel::detectCores() #if you will run this on a cluster, detect the cores.
#     )
    gam_model <-
      gen_additive_mod() %>%
      set_engine("mgcv") %>%
      set_mode("regression")

  # gam_model <-
  #   gen_additive_mod() |>                # GAM 模型
  #   set_engine("mgcv", formula = listening_time_minutes~.)|>   # 指定平滑项
  #   set_mode("regression")

  # 4. 工作流整合
  gam_workflow <- workflow() %>%
    add_recipe(rcp) %>%
    add_model(gam_model)|>
    fit(data = df)
#
#   gam_workflow <- workflow() %>%
#     add_recipe(gam_recipe) %>%
#     add_model(gam_model)
#     gam_formula <- listening_time_minutes ~
#       s(episode_length_minutes, bs = "cr", k = 10) +
#       s(host_popularity_percentage, bs = "tp") +
#       s(guest_popularity_percentage, bs = "cr", k = 5) +
#       te(number_of_ads, genre, k = c(3, 5)) +  # 仅保留关键交互
#       s(score_pctl, bs = "cr") +
#       s(episode_sentiment, bs = "re")  # 情感作为随机效应


    # gam_model <- gam(
    #   formula = listening_time_minutes~
    #     s(episode_length_minutes)+
    #     s(host_popularity_percentage)+
    #     s(guest_popularity_percentage)+
    #     number_of_ads
    #     #te(genre, publication_day,publication_time, episode_sentiment,episode_id)
    #     ,
    #   data = df,
    #   method = "REML",     # 限制性最大似然，推荐参数估计方法[7](@ref)
    #   family = gaussian(), # 连续型响应变量
    #   select = TRUE        # 启用自动平滑项选择[7](@ref)
    #    )
  gam_model <- gam_workflow |> extract_fit_engine()
  gam_model |> glance() |> print()
  return(gam_model)
}

get_glance_gam(df=tmp_df_median)
## TODO 1. study if the mice package support train /test mode. that is build a model / answer: should not. logic is cycling
####
