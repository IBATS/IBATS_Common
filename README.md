# IBATSCommon

[![Build Status](https://travis-ci.org/IBATS/IBATSCommon.svg?branch=master)](https://travis-ci.org/IBATS/IBATSCommon)
[![GitHub issues](https://img.shields.io/github/issues/IBATS/IBATSCommon.svg)](https://github.com/IBATS/IBATSCommon/issues)
[![GitHub forks](https://img.shields.io/github/forks/IBATS/IBATSCommon.svg)](https://github.com/IBATS/IBATSCommon/network)
[![GitHub stars](https://img.shields.io/github/stars/IBATS/IBATSCommon.svg)](https://github.com/IBATS/IBATSCommon/stargazers) 
[![GitHub license](https://img.shields.io/github/license/IBATS/IBATSCommon.svg)](https://github.com/IBATS/IBATSCommon/blob/master/LICENSE) 
[![HitCount](http://hits.dwyl.io/IBATS/https://github.com/DataIntegrationAlliance/IBATSCommon.svg)](http://hits.dwyl.io/DataIntegrationAlliance/https://github.com/IBATS/IBATSCommon)
[![Twitter](https://img.shields.io/twitter/url/https/github.com/IBATS/IBATSCommon.svg?style=social)](https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2FIBATS%2FIBATSCommon) 

IBATS（Integration Backtest Analysis Trade System）的公共模块，所有Feeder, Trader均集成这个模块，并使用其提供的一些公共工具

#### 修改历史：
- 2020-10-11  v0.20.4
>feat: add pair plots

- 2020-10-10  v0.20.3
>feat: add adf_coint_test

- 2020-10-09  v0.20.2
> add ts_plot for ACP PACF analysis plot
> remove diff(1) on add_factor_of_price

- 2020-10-06  v0.20.0
> add factor_analysis with factor_analyzer module
> add diff factor on get_factor function

- 2020-09-30  v0.19.7
> 对非平稳因子，如果sdt<10 使用diff(1) 否则使用 pct_change

- 2020-09-22  v0.19.6
> pct_change 因子中 inf 值转化为当前序列的最大值或最小值（取决于是正无穷，还是负无穷）

- 2020-09-22  v0.19.6
> 增加 add_pct_change_columns=True 在 add_factor_of_price，实现对非平稳序列增加相应的 pct_change 列

- 2020-09-11  v0.18.7
> 增强 market2 中的成交价格形成机制，支持通过 DealPriceScheme 参数化控制 

- 2020-09-05  v0.18.6
> 增加 market2.py 市场模拟工具

- 2020-02-09
> remove keras
> add dependence tensorflow>=2.1.0. keras is replaced by tensorflow.keras 

- 2020-01-29
> market2 在 market 基础上进行修改：调整 action 数值，为了更加适应 one_hot 模式，调整 long=0, short=1 close=2, keep=3。 \
> flag转化为 one hot 模式。

- 2020-01-22
> remove " and is_windows_os()" on if condition of plot_or_show function.

- 在此之前的历史就不记录了

#### 编译命令
```bash
python setup.py build bdist_wheel
twine upload dist/*
```