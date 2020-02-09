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

- 2020-02-09
> remove keras
> add dependence tensorflow>=2.1.0. keras is replaced by tensorflow.keras 

- 2020-01-29
> market2 在 market 基础上进行修改：调整 action 数值，为了更加适应 one_hot 模式，调整 long=0, short=1 close=2, keep=3。 \
> flag转化为 one hot 模式。

- 2020-01-22
> remove " and is_windows_os()" on if condition of plot_or_show function.

- 在此之前的历史就不记录了
