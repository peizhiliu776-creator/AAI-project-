# 使用说明

本项目基于 CitiBike CSV 数据，按照每小时训练站点需求，并通过 Streamlit 网页展示历史分析、未来预测、热点地图、热力图、天气影响和站点或者区域趋势，用的python版本是Python 3.13.7

## 各个文件夹的作用

`data/`

存放原始CSV 数据。如果要添加新的同样格式的数据，把 CSV 或解压后的文件放到这里即可

`app/`

Streamlit 网页应用代码。里面的程序是 `app/streamlit_app.py`

`src/`

存放主要训练以及功能模块，包括数据读取、清洗、特征工程、建模、预测、地图和可视化逻辑

`scripts/`

存放可直接运行的脚本文件，例如完整 pipeline、NOAA 天气下载、App 数据文件生成等

`processed/`

存放 pipeline 生成的数据，例如清洗后的骑行记录、站点小时需求表、区域小时需求表、App 加载用的 parquet （缓存）文件等。这个文件夹可以删除，重新运行 pipeline 后会再次生成

`models/`

存放训练好的模型文件，例如最佳需求预测模型、member/casual 专用模型和模型配置文件。这个文件夹可以删除，重新训练后会再生成

`outputs/`

存放输出结果，例如模型评估指标、预测结果、图表、NOAA 天气 CSV、特征重要性、混淆矩阵等。这个文件夹可以删除，重新运行后会再生成

`.idea/`

PyCharm 的项目配置文件夹，只和本地 IDE 有关，不影响算法本身

## 各个程序的作用

`scripts/run_pipeline.py`

主运行程序。它会读取原始 CSV，清洗数据，生成特征，聚合站点和区域小时需求，训练模型，并保存评估结果、模型文件和网页所需要的数据

`scripts/download_noaa_weather.py`

从 NOAA Climate Data Online / NCEI 接口下载的纽约天气数据，生成 `outputs/external/noaa_nyc_daily_weather.csv`。天气字段包括温度、降水、降雪、风速等

`scripts/build_app_artifacts.py`

重新生成 App 加载用的数据文件，不重新训练模型。适合模型已经训练好，如果网页加载数据需要更新时可以用它

`app/streamlit_app.py`

Streamlit 网页应用入口。运行后可以查看 Historical Analysis、Future Demand Prediction、Weather Impact、Station/Region Trend Viewer 等页面

`src/data_loader.py`

查找和读取 Citi Bike CSV 文件。支持全数据集读取，或者抽样读取，看想训练的数据集数量决定，并跳过无效的系统文件

`src/preprocess.py`

清洗原始骑行记录，统一列名，解析开始和结束时间，计算骑行时长和距离，过滤无效时间、无效经纬度和异常骑行记录

`src/feature_engineering.py`

生成建模需要的特征，包括时间、节假日、周末和工作日、季节、站点、区域网格、滞后需求、滚动统计、天气强度等

`src/weather.py` 和 `src/weather_features.py`

处理天气数据，并把天气数值转换成天气类别，例如无雨、小雨、中雨、大雨、无雪、小雪、大雪、低温、适中、高温、风力等级等

`src/holidays.py`

生成 NYC 假期表，并在特征中标记是否假日、假日名称等信息

`src/modeling.py`

训练和比较机器学习模型，计算 MAE、RMSE、R2、WAPE、SMAPE 等指标，保存最佳模型、模型对比结果、特征重要性和混淆矩阵

`src/forecast.py`

根据用户在 App 中选择的未来日期、小时、站点或者区域、rider type、节假日、季节、天气场景，生成未来需求预测和趋势预测

`src/map_utils.py`

生成地图、站点信息、热力图、热点地图和流向线图

`src/visualization.py`

生成图表，例如历史趋势、天气影响、模型对比等

`src/geocoding.py`

根据站点经纬度查询真实街道地址，并缓存到 `outputs/external/station_addresses.csv`

`src/pipeline.py`

把数据处理、特征工程、建模、输出保存串成完整流程，方便二次使用

## 机器学习算法

这个测算当前站点需求的任务本质上是回归任务。预测目标是某个站点在某个小时的预计骑行需求量，所以主要看误差指标，而不是只看普通分类任务里的 accuracy

`random_forest`

随机森林能处理非线性关系，对站点、小时、天气、节假日等表格特征比较稳健，不需要太多特征缩放

`xgboost`

XGBoost 是更强的梯度提升树实现，通常在表格型预测任务上表现较好。当前主力 `xgboost` 使用更低的 learning rate、更多树和稍深的树来提高对站点、小时、区域、天气和历史需求模式的拟合能力

`xgboost_smooth`

这是更稳健的 XGBoost 候选，用更浅的树、更强的正则化和更高的 min_child_weight，目标是减少过拟合，让 MAE 和 WAPE 更稳定

`xgboost_tweedie`

这是面向非负需求量的 XGBoost 候选，适合很多低需求记录、少量高需求记录数据的的偏态分布，并且直接用原始 demand 训练，不再对目标值做 log 转换

现在默认保留四个模型进行对比：`random_forest`、`xgboost`、`xgboost_smooth`、`xgboost_tweedie`。组合模型 `ensemble_mae` 和 `ensemble_balanced` 已取消，最终只会从这些单模型候选里选择。模型训练后仍会做预测校准，如果验证集上发现某个模型存在系统性偏高或偏低，程序会自动在线性校准和单调非线性校准中选择更好的方式，让 App 里的预测也使用校准后的结果。最终模型默认按加权综合排名选择，更重视 MAE、WAPE、R2 和 RMSE，同时保留 SMAPE 作为辅助指标。最终结果会保存到：

```text
outputs/best_model_summary.csv
outputs/model_selection_ranking.csv
outputs/evaluation_metrics.csv
models/best_station_demand_model.joblib
```
整体运行流程为：

读取 Citi Bike CSV

清洗数据

构建站点信息

聚合 station hour和 demand

加入时间、天气、节假日、区域、历史需求特征

按时间切分 训练集，矫正集，测试集

算法训练

每个模型做 calibration 和 residual adjustment

在 test set 上计算指标

按 combined ranking 选择最佳模型

保存模型和 App 需要的结果文件

App 加载最佳模型进行未来需求预测

如果 rider type 选择 `member` 或 `casual`，pipeline 默认还会训练对应人群的模型，App 预测时会优先使用对应人群的模型

## 判断指标是什么意思

`MAE`

平均绝对误差：比如 MAE = 1.35，表示平均每个“站点每小时”的预测误差约 1.35 次骑行。越低越好

`RMSE`

均方根误差：它会更重地惩罚大误差，所以可以用来看模型是否经常出现很大的错判。越低越好

`R2`

决定系数：表示模型解释需求变化的能力。越接近 1 越好，接近 0 说明模型和只用平均值差不多，负数说明效果很差

`WAPE`

相对误差：总绝对误差除以总真实需求量，越低越好

`SMAPE`

对称百分比误差：它比普通 MAPE 更适合低需求站点，因为很多站点小时需求可能接近 0。越低越好

`MAPE_nonzero`

只在真实需求大于 0 的记录上计算百分比误差，避免真实值为 0 时无法计算的问题。越低越好

`hotspot accuracy`

把高需求站点小时标记成 hotspot 后，判断热点/非热点是否正确。它是辅助指标，不是需求预测的主指标

`precision`

预测为热点的记录里，有多少是真的热点。越高表示误报越少

`recall`

真实热点里，有多少被模型找出来了。越高表示漏掉热点越少

`F1`

准确度和 recall 的综合指标。越高越好

`confusion matrix`

混淆矩阵，用来检查热点分类具体错在哪里

## 如何运行
在控制台或者终端复制下面指令并粘贴即可
先进入项目根目录：

```powershell
cd "C:\Users\szzddx\Desktop\AAI project"
```

安装需求和依赖：

```powershell
pip install -r requirements.txt
```

启用 XGBoost 模型：

```powershell
pip install xgboost
```

如果已经有天气文件 `outputs/external/noaa_nyc_daily_weather.csv`，可以直接运行 pipeline。

如果需要重新下载 NOAA 天气数据：

```powershell
$env:NOAA_TOKEN="NOAA_TOKEN"
python scripts/download_noaa_weather.py --start 2025-08-01 --end 2026-03-31
```
根据需求可以调整时间范围，token 可以用我的：CWxPrSwuzDQqbcgAOnRvxEalDiyrjGjs


推荐运行完整 pipeline：

```powershell
python scripts/run_pipeline.py --weather outputs/external/noaa_nyc_daily_weather.csv
```

这个命令会使用项目 `data/` 文件夹里有的CSV，清洗数据，生成站点和区域小时需求数据，合并天气、节假日、周末和工作日、季节等特征，训练并比较多个机器学习模型，保存最佳模型和评估结果，并生成 App 需要的输出文件

注意：这个默认命令会读取全部原始 CSV，并生成全量地图和历史分析数据；但模型训练默认最多使用最近 `500,000` 行 station-hour 数据。这样做是为了避免本地训练时间太长或内存不足

如果只是想正常运行软件和得到一个可用模型，可以用默认命令：

```powershell
python scripts/run_pipeline.py --weather outputs/external/noaa_nyc_daily_weather.csv
```

如果想让模型使用更大的训练样本，同时仍然比较稳定，推荐使用 `1,000,000` 行：

```powershell
python scripts/run_pipeline.py --weather outputs/external/noaa_nyc_daily_weather.csv --max-train-rows 1000000
```

如果电脑内存和时间都比较充足，可以尝试 `1,500,000` 行：

```powershell
python scripts/run_pipeline.py --weather outputs/external/noaa_nyc_daily_weather.csv --max-train-rows 1500000
```

如果要尝试使用全部站点小时训练数据：

```powershell
python scripts/run_pipeline.py --weather outputs/external/noaa_nyc_daily_weather.csv --max-train-rows 0
```

这个可能很慢，也可能占用大量内存。（我已经爆了一次内存，而且app加载会很慢）

App 为了避免每次打开网页都加载几百万行历史记录，会非常慢，会单独生成一个预测上下文文件 `processed/app_station_context.parquet`。默认每个站点/人群保留最近 `168` 小时上下文。

如果想让 App 使用更多历史上下文，可以在完整 pipeline 里指定：

```powershell
python scripts/run_pipeline.py --weather outputs/external/noaa_nyc_daily_weather.csv --max-train-rows 1000000 --app-recent-hours 720
```

如果模型和 processed 数据已经生成好了，只想重新生成 App 使用的上下文文件，可以运行：

```powershell
python scripts/build_app_artifacts.py --recent-hours 720
```

常用选择：

`--app-recent-hours 168`：默认值，速度最快，内存压力小

`--app-recent-hours 336`：保留最近 14 天上下文

`--app-recent-hours 720`：保留最近 30 天上下文，预测和趋势页信息更充分，但 App 加载会更慢

`--app-recent-hours 0`：App 上下文保留全部 station-hour 行，不推荐，可能会导致内存不足

如果只想总共抽样约 100,000 条原始骑行记录：

```powershell
python scripts/run_pipeline.py --weather outputs/external/noaa_nyc_daily_weather.csv --total-sample-rows 100000
```

注意：这种方式会影响处理后的地图覆盖范围，因为整条 pipeline 都基于抽样数据。

如果只想快速测试流程：

```powershell
python scripts/run_pipeline.py --weather outputs/external/noaa_nyc_daily_weather.csv --fast-models
```

如果想跳过 member/casual 单独模型：

```powershell
python scripts/run_pipeline.py --weather outputs/external/noaa_nyc_daily_weather.csv --no-rider-models
```

如果模型已经训练过，只想重新生成网页加载用的聚合数据：

```powershell
python scripts/build_app_artifacts.py
```

要打开网页的话，在终端输入：

```powershell
python -m streamlit run app/streamlit_app.py
```
运行后浏览器会打开本地网页

## 主要输出文件

`processed/clean_trips.parquet`

清洗后的骑行明细

`processed/station_hour_demand.parquet`

站点小时级需求建模数据

`processed/area_hour_demand.parquet`

区域和网格小时级需求数据

`processed/station_metadata.csv`

站点 ID、站点名称、经纬度、区域等信息

`models/best_station_demand_model.joblib`

App 默认使用的最佳站点需求预测模型

`models/best_station_demand_model_member.joblib`

member rider 专用模型，如果训练时没有跳过 rider-specific models

`models/best_station_demand_model_casual.joblib`

casual rider 专用模型，如果训练时没有跳过 rider-specific models

`outputs/evaluation_metrics.csv`

各模型评估结果

`outputs/best_model_summary.csv`

最终选中的最佳模型和关键指标

`outputs/model_selection_ranking.csv`

综合排名计算结果

`outputs/figures/`

模型对比图、特征重要性、混淆矩阵、预测效果图等

`outputs/predictions/`

最新预测结果和热点结果

