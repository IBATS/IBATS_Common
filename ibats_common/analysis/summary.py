#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 19-5-9 上午9:53
@File    : summary.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import logging
import os
from collections import defaultdict

import docx
import numpy as np
import pandas as pd
from scipy.stats import anderson, normaltest

from ibats_common.analysis import get_report_folder_path
from ibats_common.analysis.corr import corr
from ibats_common.analysis.plot import drawdown_plot, plot_rr_df, wave_hist
from ibats_common.analysis.plot_db import get_rr_with_md

logger = logging.getLogger(__name__)


def summary_md(df: pd.DataFrame, percentiles=[0.2, 1 / 3, 0.5, 2 / 3, 0.8], risk_free=0.03,
               figure_4_each_col=False,
               col_transfer_dic: (dict, None) = None,
               stat_col_name_list=None,
               drawdown_col_name_list=None,
               ):
    """
    汇总展示数据分析结果，同时以 dict 形式返回各项指标分析结果
    第一个返回值，df的各项分析结果
    第二个返回值，各个列的各项分析结果
    :param df:
    :param percentiles:分为数信息
    :param risk_free:无风险收益率
    :param figure_4_each_col:hist图使用，每一列显示单独一张图片
    :param col_transfer_dic:列转换方法
    :param stat_col_name_list:对哪些列的数据执行统计
    :return:
    """
    columns = list(df.columns)
    logger.info('data columns: %s', columns)
    ret_dic = {}
    each_col_dic = defaultdict(dict)

    logger.info('Description:')
    df.describe(percentiles=percentiles)
    corr_df = corr(df)
    ret_dic['corr'] = corr_df
    logger.info('Correlation Coefficient:\n%s', corr_df)
    # ffn_stats = ffn.calc_stats(df)
    # histgram 分布图
    n_bins_dic = wave_hist(df, figure_4_each_col=figure_4_each_col, col_transfer_dic=col_transfer_dic)
    ret_dic['wave_hist'] = n_bins_dic
    # 回撤曲线
    drawdown_df = drawdown_plot(df, col_name_list=drawdown_col_name_list)
    ret_dic['drawdown'] = drawdown_df
    # 单列分析
    stat_df = (df if stat_col_name_list is None else df[stat_col_name_list])
    for col_name, data in stat_df.items():
        data = data.dropna()
        data = data[~np.isinf(data)]
        logger.info("=" * 50 + ' %s ' + "=" * 50, col_name)
        result = normaltest(data)
        each_col_dic[col_name]['normaltest'] = result
        logger.info('%s %s', col_name, result)
        result = anderson(data)
        logger.info('%s %s', col_name, result)
        each_col_dic[col_name]['anderson'] = result
        stats = data.calc_perf_stats()
        # stats = ffn_stats[col_name]
        # 设置无风险收益率
        stats.set_riskfree_rate(risk_free)
        stats.display()

    return ret_dic, each_col_dic


def summary_rr(df: pd.DataFrame, risk_free=0.03,
               figure_4_each_col=False,
               col_transfer_dic: (dict, None) = None,
               stat_col_name_list=None,
               drawdown_col_name_list=None,
               enable_show_plot=True,
               enable_save_plot=False,
               name=None
               ):
    """
    汇总展示数据分析结果，同时以 dict 形式返回各项指标分析结果
    第一个返回值，df的各项分析结果
    第二个返回值，各个列的各项分析结果
    :param df:
    :param risk_free:无风险收益率
    :param figure_4_each_col:hist图使用，每一列显示单独一张图片
    :param col_transfer_dic:列转换方法
    :param stat_col_name_list:对哪些列的数据执行统计
    :param drawdown_col_name_list:对哪些列的数据执行统计
    :param enable_show_plot: 展示plot
    :param enable_save_plot: 保存文件
    :param name:
    :return:
    """
    columns = list(df.columns)
    logger.info('data columns: %s', columns)
    ret_dic, each_col_dic, file_path_dic = {}, defaultdict(dict), {}
    # 获取统计数据
    stats = df.calc_stats()
    stats.set_riskfree_rate(risk_free)
    enable_kwargs_dic = {"enable_save_plot": enable_save_plot, "enable_show_plot": enable_show_plot, "name": name}
    file_path = plot_rr_df(df, **enable_kwargs_dic)
    if enable_save_plot:
        file_path_dic['rr'] = file_path

    # histgram 分布图
    n_bins_dic, file_path = wave_hist(df, figure_4_each_col=figure_4_each_col, col_transfer_dic=col_transfer_dic,
                           **enable_kwargs_dic)
    ret_dic['hist'] = n_bins_dic
    if enable_save_plot:
        file_path_dic['hist'] = file_path

    # 回撤曲线
    drawdown_df, file_path = drawdown_plot(df, perf_stats=stats, col_name_list=drawdown_col_name_list,
                                           **enable_kwargs_dic)
    if enable_save_plot:
        file_path_dic['drawdown'] = file_path

    ret_dic['drawdown'] = drawdown_df
    # 单列分析
    stat_df = (df if stat_col_name_list is None else df[stat_col_name_list])
    for col_name, data in stat_df.items():
        data = data.dropna()
        data = data[~np.isinf(data)]
        logger.info("=" * 50 + ' %s ' + "=" * 50, col_name)
        result = normaltest(data)
        each_col_dic[col_name]['normaltest'] = result
        logger.info('%s %s', col_name, result)
        result = anderson(data)
        logger.info('%s %s', col_name, result)
        each_col_dic[col_name]['anderson'] = result
        stats = data.calc_perf_stats()
        # stats = ffn_stats[col_name]
        # 设置无风险收益率
        stats.set_riskfree_rate(risk_free)
        stats.display()

    return ret_dic, each_col_dic, file_path_dic


def _test_summary_md():
    from ibats_common.example.data import load_data

    df = load_data('RB.csv').set_index('trade_date').drop('instrument_type', axis=1)
    df.index = pd.DatetimeIndex(df.index)
    col_transfer_dic = {
        'return': ['open', 'high', 'low', 'close', 'volume']
    }
    summary_md(df, drawdown_col_name_list=['close'], figure_4_each_col=False, stat_col_name_list=['close'],
               col_transfer_dic=col_transfer_dic)


def summary_stg(stg_run_id=None):
    from ibats_common.backend.mess import get_latest_stg_run_id
    from ibats_common.analysis.plot_db import show_order, show_cash_and_margin, show_rr_with_md
    if stg_run_id is None:
        stg_run_id = get_latest_stg_run_id()

    show_order(stg_run_id, module_name_replacement_if_main='ibats_common.example.ma_cross_stg')
    df = show_cash_and_margin(stg_run_id)
    sum_df, symbol_rr_dic = show_rr_with_md(stg_run_id,
                                            module_name_replacement_if_main='ibats_common.example.ma_cross_stg')
    summary_rr(sum_df, figure_4_each_col=True, col_transfer_dic={'return': sum_df.columns})
    # for symbol, rr_df in symbol_rr_dic.items():
    #     col_transfer_dic = {'return': rr_df.columns}
    #     summary_rr(rr_df, figure_4_each_col=True, col_transfer_dic=col_transfer_dic)


def _test_summary_stg():
    stg_run_id = None
    summary_stg(stg_run_id)


def summary_stg_2_docx(stg_run_id=None, module_name_replacement_if_main='ibats_common.example.ma_cross_stg'):
    """
    生成策略分析报告
    :param stg_run_id:
    :param module_name_replacement_if_main:
    :return:
    """
    # rr_plot_file_path = os.path.join(cache_folder_path, 'rr_plot.png')
    stg_run_id, sum_df, symbol_rr_dic = get_rr_with_md(
        stg_run_id,
        module_name_replacement_if_main=module_name_replacement_if_main
    )
    ret_dic, each_col_dic, file_path_dic = summary_rr(sum_df, enable_save_plot=True)

    # show_order(stg_run_id, module_name_replacement_if_main='ibats_common.example.ma_cross_stg')
    # df = show_cash_and_margin(stg_run_id)

    # 生成 docx 万恶将所需变量
    heading_title = f'策略分析报告[{stg_run_id}]'

    # 生成 docx 文件
    document = docx.Document()
    # 设置默认字体
    document.styles['Normal'].font.name = '微软雅黑'
    document.styles['Normal']._element.rPr.rFonts.set(docx.oxml.ns.qn('w:eastAsia'), '微软雅黑')

    # 创建自定义段落样式(第一个参数为样式名, 第二个参数为样式类型, 1为段落样式, 2为字符样式, 3为表格样式)
    UserStyle1 = document.styles.add_style('UserStyle1', 1)
    # 设置字体尺寸
    UserStyle1.font.size = docx.shared.Pt(40)
    # 设置字体颜色
    UserStyle1.font.color.rgb = docx.shared.RGBColor(0xff, 0xde, 0x00)
    # 居中文本
    UserStyle1.paragraph_format.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
    # 设置中文字体
    UserStyle1.font.name = '微软雅黑'
    UserStyle1._element.rPr.rFonts.set(docx.oxml.ns.qn('w:eastAsia'), '微软雅黑')

    # 加入不同等级的标题
    document.add_heading(heading_title, 0)
    document.add_heading(u'策略回测收益曲线', 1)
    # 增加图片（此处使用相对位置）
    document.add_picture(file_path_dic['rr'])  # , width=docx.shared.Inches(1.25)
    document.add_heading(u'策略回撤曲线', 2)
    document.add_picture(file_path_dic['drawdown'])  # , width=docx.shared.Inches(1.25)

    # 添加文本
    paragraph = document.add_paragraph(u'添加了文本')
    # 设置字号
    run = paragraph.add_run(u'设置字号')
    run.font.size = docx.shared.Pt(24)

    # 保存文件
    file_name = f"{stg_run_id} {np.random.randint(10000)}.docx"
    file_path = os.path.join(get_report_folder_path(), file_name)
    document.save(file_path)
    return file_path


def _test_summary_stg_2_docx(auto_open_file=True):
    import subprocess
    stg_run_id = None
    file_path = summary_stg_2_docx(stg_run_id)
    if auto_open_file:
        # subprocess.Popen(file_path)
        subprocess.call(["xdg-open", file_path])


if __name__ == "__main__":
    # _test_summary_md()
    # _test_summary_stg()
    _test_summary_stg_2_docx()
