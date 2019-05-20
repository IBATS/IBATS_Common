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
import datetime
import json

import docx
import numpy as np
import pandas as pd
from ibats_utils.mess import open_file_with_system_app, date_2_str, datetime_2_str
from scipy.stats import anderson, normaltest

from ibats_common.analysis import get_report_folder_path
from ibats_common.analysis.plot import drawdown_plot, plot_rr_df, wave_hist, plot_scatter_matrix, plot_corr, clean_cache
from ibats_common.analysis.plot_db import get_rr_with_md, show_trade, show_cash_and_margin
from ibats_common.backend.mess import get_stg_run_info
from ibats_common.common import RunMode, CalcMode

logger = logging.getLogger(__name__)
STR_FORMAT_DATETIME_4_FILE_NAME = '%Y-%m-%d %H_%M_%S'
FORMAT_2_PERCENT = lambda x: f"{x * 100: .2f}%"
FORMAT_2_FLOAT2 = r"{0:.2f}"


def summary_md(df: pd.DataFrame, percentiles=[0.2, 1 / 3, 0.5, 2 / 3, 0.8],
               risk_free=0.03,
               stat_col_name_list=None,
               enable_show_plot=True,
               enable_save_plot=False,
               name=None,
               func_kwargs_dic={},
               **kwargs,
               ):
    """
    汇总展示数据分析结果，同时以 dict 形式返回各项指标分析结果
    第一个返回值，df的各项分析结果
    第二个返回值，各个列的各项分析结果
    第三个返回值，相关文件输出路径或路径列表（当 enable_save_plot=True）
    :param df:
    :param risk_free:无风险收益率
    :param stat_col_name_list:对哪些列的数据执行统计
    :param enable_show_plot: 展示plot
    :param enable_save_plot: 保存文件
    :param name:
    :param func_kwargs_dic:
    :param kwargs:
    :return:
    """
    columns = list(df.columns)
    logger.info('data columns: %s', columns)
    ret_dic, each_col_dic, file_path_dic = {}, defaultdict(dict), {}

    logger.info('Description:')
    df.describe(percentiles=percentiles)

    quantile_df = df.quantile(percentiles)
    ret_dic['quantile_df'] = quantile_df

    # 获取统计数据
    stats = df.calc_stats()
    stats.set_riskfree_rate(risk_free)
    ret_dic['stats'] = stats
    enable_kwargs_dic = {"enable_save_plot": enable_save_plot, "enable_show_plot": enable_show_plot, "name": name}

    # scatter_matrix
    # diagonal，必须且只能在{'hist', 'kde'}中选择1个，
    # 'hist'表示直方图(Histogram plot),'kde'表示核密度估计(Kernel Density Estimation)
    # 该参数是scatter_matrix函数的关键参数
    file_path = plot_scatter_matrix(df, diagonal='kde', **enable_kwargs_dic)
    if enable_save_plot:
        file_path_dic['scatter_matrix'] = file_path

    # stats.plot_correlation()
    file_path = plot_corr(df, **enable_kwargs_dic)
    if enable_save_plot:
        file_path_dic['correlation'] = file_path

    # return rate plot 图
    func_kwargs = func_kwargs_dic.setdefault('rr', {})
    file_path = plot_rr_df(df, **func_kwargs, **enable_kwargs_dic)
    if enable_save_plot:
        file_path_dic['rr'] = file_path

    # histgram 分布图
    func_kwargs = func_kwargs_dic.setdefault('hist', {})
    n_bins_dic, file_path = wave_hist(df, **func_kwargs, **enable_kwargs_dic)
    ret_dic['hist'] = n_bins_dic
    if enable_save_plot:
        file_path_dic['hist'] = file_path

    # 回撤图
    func_kwargs = func_kwargs_dic.setdefault('drawdown', {})
    drawdown_df, file_path = drawdown_plot(df, perf_stats=stats, **func_kwargs, **enable_kwargs_dic)
    ret_dic['drawdown'] = drawdown_df
    if enable_save_plot:
        file_path_dic['drawdown'] = file_path

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


def summary_rr(df: pd.DataFrame, risk_free=0.03,
               figure_4_each_col=False,
               col_transfer_dic: (dict, None) = None,
               stat_col_name_list=None,
               col_name_list=None,
               enable_show_plot=True,
               enable_save_plot=False,
               name=None,
               **kwargs
               ):
    """
    汇总展示数据分析结果，同时以 dict 形式返回各项指标分析结果
    第一个返回值，df的各项分析结果
    第二个返回值，各个列的各项分析结果
    第三个返回值，相关文件输出路径或路径列表（当 enable_save_plot=True）
    :param df:
    :param risk_free:无风险收益率
    :param figure_4_each_col:hist图使用，每一列显示单独一张图片
    :param col_transfer_dic:列转换方法
    :param stat_col_name_list:对哪些列的数据执行统计
    :param col_name_list:对哪些列的数据执行统计
    :param enable_show_plot: 展示plot
    :param enable_save_plot: 保存文件
    :param name:
    :param kwargs:
    :return:
    """
    columns = list(df.columns)
    logger.info('data columns: %s', columns)
    ret_dic, each_col_dic, file_path_dic = {}, defaultdict(dict), {}

    # 获取统计数据
    stats = df.calc_stats()
    stats.set_riskfree_rate(risk_free)
    ret_dic['stats'] = stats
    enable_kwargs_dic = {"enable_save_plot": enable_save_plot, "enable_show_plot": enable_show_plot, "name": name}

    # scatter_matrix
    # diagonal，必须且只能在{'hist', 'kde'}中选择1个，
    # 'hist'表示直方图(Histogram plot),'kde'表示核密度估计(Kernel Density Estimation)
    # 该参数是scatter_matrix函数的关键参数
    file_path = plot_scatter_matrix(df, diagonal='kde', **enable_kwargs_dic)
    if enable_save_plot:
        file_path_dic['scatter_matrix'] = file_path

    # stats.plot_correlation()
    file_path = plot_corr(df, **enable_kwargs_dic)
    if enable_save_plot:
        file_path_dic['correlation'] = file_path

    # return rate plot 图
    file_path = plot_rr_df(df, **enable_kwargs_dic)
    if enable_save_plot:
        file_path_dic['rr'] = file_path

    # histgram 分布图
    n_bins_dic, file_path = wave_hist(df, figure_4_each_col=figure_4_each_col, col_transfer_dic=col_transfer_dic,
                                      **enable_kwargs_dic)
    ret_dic['hist'] = n_bins_dic
    if enable_save_plot:
        file_path_dic['hist'] = file_path

    # 回撤图
    drawdown_df, file_path = drawdown_plot(df, perf_stats=stats, col_name_list=col_name_list,
                                           **enable_kwargs_dic)
    ret_dic['drawdown'] = drawdown_df
    if enable_save_plot:
        file_path_dic['drawdown'] = file_path

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


def df_2_table(doc, df, format_by_index=None, format_by_col=None):
    row_num, col_num = df.shape
    t = doc.add_table(row_num + 1, col_num + 1)

    # Highlight all cells limegreen (RGB 32CD32) if cell contains text "0.5"
    from docx.oxml.ns import nsdecls
    from docx.oxml import parse_xml
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    # write head
    col_name_list = list(df.columns)
    for j in range(col_num):
        # t.cell(0, j).text = df.columns[j]
        # paragraph = t.cell(0, j).add_paragraph()
        paragraph = t.cell(0, j + 1).paragraphs[0]
        paragraph.add_run(col_name_list[j]).bold = True
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # write head bg color
    for j in range(col_num + 1):
        # t.cell(0, j).text = df.columns[j]
        t.cell(0, j)._tc.get_or_add_tcPr().append(
            parse_xml(r'<w:shd {} w:fill="00A2E8"/>'.format(nsdecls('w'))))

    # format table style to be a grid
    t.style = 'TableGrid'

    # populate the table with the dataframe
    for i in range(row_num):
        index = df.index[i]
        paragraph = t.cell(i + 1, 0).paragraphs[0]
        paragraph.add_run(date_2_str(index)).bold = True
        paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
        if format_by_index is not None and index in format_by_index:
            formater = format_by_index[index]
        else:
            formater = None

        for j in range(col_num):
            if formater is None and format_by_col is not None and col_name_list[j] in format_by_col:
                formater = format_by_col[col_name_list[j]]

            content = df.values[i, j]
            if formater is None:
                text = str(content)
            elif isinstance(formater, str):
                text = str.format(formater, content)
            elif callable(formater):
                text = formater(content)
            else:
                raise ValueError('%s: %s 无效', index, formater)

            paragraph = t.cell(i + 1, j + 1).paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            paragraph.add_run(text)

    for i in range(1, row_num + 1):
        for j in range(col_num + 1):
            if i % 2 == 0:
                t.cell(i, j)._tc.get_or_add_tcPr().append(
                    parse_xml(r'<w:shd {} w:fill="A3D9EA"/>'.format(nsdecls('w'))))


def _test_summary_md():
    from ibats_common.example.data import load_data

    df = load_data('RB.csv').set_index('trade_date').drop('instrument_type', axis=1)
    df.index = pd.DatetimeIndex(df.index)
    col_transfer_dic = {
        'return': ['open', 'high', 'low', 'close', 'volume']
    }
    ret_dic, each_col_dic, file_path_dic = summary_md(
        df, enable_show_plot=True, enable_save_plot=False,
        func_kwargs_dic={
            "hist": {
                "figure_4_each_col": False,
                "col_transfer_dic": col_transfer_dic,
            },
            "drawdown": {
                "col_name_list": ['close'],
                "stat_col_name_list": ['close'],
            }
        })


def summary_stg(stg_run_id=None):
    from ibats_common.analysis.plot_db import show_order, show_cash_and_margin, show_rr_with_md
    info = get_stg_run_info(stg_run_id)
    stg_run_id = info.stg_run_id

    data_dict, file_path = show_order(stg_run_id)
    df = show_cash_and_margin(stg_run_id)
    sum_df, symbol_rr_dic, save_file_path_dic = show_rr_with_md(stg_run_id)
    summary_rr(sum_df, figure_4_each_col=True, col_transfer_dic={'return': sum_df.columns})
    # for symbol, rr_df in symbol_rr_dic.items():
    #     col_transfer_dic = {'return': rr_df.columns}
    #     summary_rr(rr_df, figure_4_each_col=True, col_transfer_dic=col_transfer_dic)


def _test_summary_stg():
    stg_run_id = None
    summary_stg(stg_run_id)


def stats_df_2_docx_table(stats_df, document):
    """
    将 stats_df 统计信息以表格形式写入 docx 文件中
    :param stats_df:
    :param document:
    :return:
    """

    format_by_index = {
        "total_return": FORMAT_2_PERCENT,
        "cagr": FORMAT_2_PERCENT,
        "mtd": FORMAT_2_PERCENT,
        "three_month": FORMAT_2_PERCENT,
        "six_month": FORMAT_2_PERCENT,
        "ytd": FORMAT_2_PERCENT,
        "one_year": FORMAT_2_PERCENT,
        "three_year": FORMAT_2_PERCENT,
        "five_year": FORMAT_2_PERCENT,
        "ten_year": FORMAT_2_PERCENT,
        "best_day": FORMAT_2_PERCENT,
        "worst_day": FORMAT_2_PERCENT,
        "best_month": FORMAT_2_PERCENT,
        "worst_month": FORMAT_2_PERCENT,
        "best_year": FORMAT_2_PERCENT,
        "worst_year": FORMAT_2_PERCENT,
        "avg_drawdown": FORMAT_2_PERCENT,
        "avg_up_month": FORMAT_2_PERCENT,
        "win_year_perc": FORMAT_2_PERCENT,
        "twelve_month_win_perc": FORMAT_2_PERCENT,
        "incep": FORMAT_2_PERCENT,
        "rf": FORMAT_2_FLOAT2,
        "max_drawdown": FORMAT_2_PERCENT,
        "calmar": FORMAT_2_FLOAT2,
        "daily_sharpe": FORMAT_2_FLOAT2,
        "daily_sortino": FORMAT_2_FLOAT2,
        "daily_mean": FORMAT_2_FLOAT2,
        "daily_vol": FORMAT_2_FLOAT2,
        "daily_skew": FORMAT_2_FLOAT2,
        "daily_kurt": FORMAT_2_FLOAT2,
        "monthly_sharpe": FORMAT_2_FLOAT2,
        "monthly_sortino": FORMAT_2_FLOAT2,
        "monthly_mean": FORMAT_2_FLOAT2,
        "monthly_vol": FORMAT_2_FLOAT2,
        "monthly_skew": FORMAT_2_FLOAT2,
        "monthly_kurt": FORMAT_2_FLOAT2,
        "yearly_sharpe": FORMAT_2_FLOAT2,
        "yearly_sortino": FORMAT_2_FLOAT2,
        "yearly_mean": FORMAT_2_FLOAT2,
        "yearly_vol": FORMAT_2_FLOAT2,
        "yearly_skew": FORMAT_2_FLOAT2,
        "yearly_kurt": FORMAT_2_FLOAT2,
        "avg_drawdown_days": FORMAT_2_FLOAT2,
        "avg_down_month": FORMAT_2_FLOAT2,
        "start": date_2_str,
        "end": date_2_str,
    }
    df_2_table(document, stats_df, format_by_index=format_by_index)


def summary_stg_2_docx(stg_run_id=None, enable_save_plot=True, enable_show_plot=False, enable_clean_cache=True):
    """
    生成策略分析报告
    :param stg_run_id:
    :param enable_save_plot:
    :param enable_show_plot:
    :param enable_clean_cache:
    :return:
    """
    info = get_stg_run_info(stg_run_id)
    stg_run_id = info.stg_run_id
    run_mode = RunMode(info.run_mode)
    kwargs = {"enable_show_plot": enable_show_plot, "enable_save_plot": enable_save_plot, "run_mode": run_mode}
    if run_mode == RunMode.Backtest_FixPercent:
        sum_df, symbol_rr_dic = get_rr_with_md(stg_run_id, compound_rr=True)
    else:
        sum_df, symbol_rr_dic = get_rr_with_md(stg_run_id, compound_rr=False)

    ret_dic, each_col_dic, file_path_dic = summary_rr(sum_df, **kwargs)

    _, file_path = show_trade(stg_run_id, **kwargs)
    file_path_dic['trade'] = file_path
    df, file_path = show_cash_and_margin(stg_run_id, **kwargs)
    file_path_dic['cash_and_margin'] = file_path

    # 生成 docx 文档将所需变量
    heading_title = f'策略分析报告[{stg_run_id}] ' \
                    f'{date_2_str(min(sum_df.index))} - {date_2_str(max(sum_df.index))} ({sum_df.shape[0]} days)'

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

    # 文件内容
    document.add_heading(heading_title, 0).alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
    document.add_paragraph('')
    document.add_paragraph('')
    heading_count = 1
    document.add_heading(f'{heading_count}、策略回测收益曲线', 1)
    # 增加图片（此处使用相对位置）
    document.add_picture(file_path_dic['rr'])  # , width=docx.shared.Inches(1.25)
    heading_count += 1
    # 添加分页符
    document.add_page_break()

    document.add_heading(f'{heading_count}、策略回撤曲线', 1)
    document.add_picture(file_path_dic['drawdown'])
    heading_count += 1
    document.add_page_break()

    if run_mode != RunMode.Backtest_FixPercent:
        document.add_heading(f'{heading_count}、现金与仓位堆叠图', 1)
        document.add_picture(file_path_dic['cash_and_margin'])
        heading_count += 1
        document.add_page_break()

    document.add_heading(f'{heading_count}、散点图矩阵图（Scatter Matrix）', 1)
    document.add_picture(file_path_dic['scatter_matrix'])
    heading_count += 1
    document.add_page_break()

    document.add_heading(f'{heading_count}、相关性矩阵图（Correlation）', 1)
    document.add_picture(file_path_dic['correlation'])
    heading_count += 1
    document.add_page_break()

    document.add_heading(f'{heading_count}、绩效统计数据（Porformance stat）', 1)
    stats_df = ret_dic['stats'].stats
    stats_df_2_docx_table(stats_df, document)
    heading_count += 1
    document.add_page_break()

    # 交易记录
    document.add_heading(f'{heading_count}、买卖点记录', 1)
    document.add_picture(file_path_dic['trade'])
    heading_count += 1
    document.add_page_break()

    # 保存文件
    try:
        calc_mode_str = CalcMode(json.loads(info.trade_agent_params_list)[0]['calc_mode']).name + " "
    except:
        calc_mode_str = " "

    run_mode_str = run_mode.name + " "
    file_name = f"{stg_run_id} {run_mode_str}{calc_mode_str}" \
                f"{date_2_str(min(sum_df.index))} - {date_2_str(max(sum_df.index))} ({sum_df.shape[0]} days) " \
                f"{datetime_2_str(datetime.datetime.now(), STR_FORMAT_DATETIME_4_FILE_NAME)}.docx"
    file_path = os.path.join(get_report_folder_path(), file_name)
    document.save(file_path)
    if enable_clean_cache:
        clean_cache()

    return file_path


def _test_summary_stg_2_docx(auto_open_file=True):
    stg_run_id = 1
    file_path = summary_stg_2_docx(stg_run_id, enable_clean_cache=True)
    if auto_open_file:
        open_file_with_system_app(file_path)


def summary_md_2_docx(df: pd.DataFrame, percentiles=[0.2, 1 / 3, 0.5, 2 / 3, 0.8],
                      risk_free=0.03,
                      stat_col_name_list=None,
                      enable_show_plot=True,
                      enable_save_plot=False,
                      name=None,
                      func_kwargs_dic={},
                      enable_clean_cache=True,
                      ):
    """
    汇总展示数据分析结果，同时以 dict 形式返回各项指标分析结果
    第一个返回值，df的各项分析结果
    第二个返回值，各个列的各项分析结果
    :param df:
    :param percentiles:分为数信息
    :param risk_free:无风险收益率
    :param stat_col_name_list:对哪些列的数据执行统计
    :param enable_show_plot:显示plot
    :param enable_save_plot:保存plot
    :param name:
    :param func_kwargs_dic:
    :param enable_clean_cache:
    :return:
    """
    ret_dic, each_col_dic, file_path_dic = summary_md(
        df, percentiles=percentiles, risk_free=risk_free, stat_col_name_list=stat_col_name_list,
        enable_show_plot=enable_show_plot, enable_save_plot=enable_save_plot, name=name,
        func_kwargs_dic=func_kwargs_dic)

    # 生成 docx 文档将所需变量
    heading_title = f'数据分析报告 {date_2_str(min(df.index))} - {date_2_str(max(df.index))} ({df.shape[0]} days)'

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

    # 文件内容
    document.add_heading(heading_title, 0).alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
    document.add_paragraph('')
    document.add_paragraph('')
    heading_count = 1
    if 'rr' in file_path_dic:
        document.add_heading(f'{heading_count}、行情曲线', 1)
        # 增加图片（此处使用相对位置）
        document.add_picture(file_path_dic['rr'])  # , width=docx.shared.Inches(1.25)
        heading_count += 1
        # 添加分页符
        document.add_page_break()

    if 'quantile_df' in ret_dic:
        document.add_heading(f'{heading_count}、分位数信息（Quantile）', 1)
        df = ret_dic['quantile_df']
        format_by_col = {_: FORMAT_2_FLOAT2 for _ in df.columns}
        df_2_table(document, df, format_by_col=format_by_col)
        heading_count += 1
        document.add_page_break()

    if 'hist' in file_path_dic:
        document.add_heading(f'{heading_count}、Histgram 分布图', 1)
        document.add_picture(file_path_dic['hist'])
        heading_count += 1
        document.add_page_break()

    if 'drawdown' in file_path_dic:
        document.add_heading(f'{heading_count}、行情回撤曲线', 1)
        document.add_picture(file_path_dic['drawdown'])
        heading_count += 1
        document.add_page_break()

    if 'scatter_matrix' in file_path_dic:
        document.add_heading(f'{heading_count}、散点图矩阵图（Scatter Matrix）', 1)
        document.add_picture(file_path_dic['scatter_matrix'])
        heading_count += 1
        document.add_page_break()

    if 'correlation' in file_path_dic:
        document.add_heading(f'{heading_count}、相关性矩阵图（Correlation）', 1)
        document.add_picture(file_path_dic['correlation'])
        heading_count += 1
        document.add_page_break()

    if 'stats' in file_path_dic:
        document.add_heading(f'{heading_count}、绩效统计数据（Porformance stat）', 1)
        stats_df = ret_dic['stats'].stats
        stats_df_2_docx_table(stats_df, document)
        heading_count += 1
        document.add_page_break()

    # 保存文件
    file_name = f"MD {date_2_str(min(df.index))} - {date_2_str(max(df.index))} ({df.shape[0]} days) " \
                f"{datetime_2_str(datetime.datetime.now(), STR_FORMAT_DATETIME_4_FILE_NAME)}.docx"
    file_path = os.path.join(get_report_folder_path(), file_name)
    document.save(file_path)
    if enable_clean_cache:
        clean_cache()

    return file_path


def _test_summary_md_2_docx(auto_open_file=True):
    from ibats_common.example.data import load_data

    df = load_data('RB.csv').set_index('trade_date').drop('instrument_type', axis=1)
    df.index = pd.DatetimeIndex(df.index)
    col_transfer_dic = {
        'return': ['open', 'high', 'low', 'close', 'volume']
    }
    file_path = summary_md_2_docx(
        df, enable_show_plot=False, enable_save_plot=True, stat_col_name_list=['close'],
        func_kwargs_dic={
            "hist": {
                "figure_4_each_col": False,
                "col_transfer_dic": col_transfer_dic,
            },
            "drawdown": {
                "col_name_list": ['close'],
            },
            "rr": {
                "col_name_list": ['close'],
            },
        })
    if auto_open_file:
        open_file_with_system_app(file_path)


if __name__ == "__main__":
    # _test_summary_md()
    # _test_summary_stg()
    # _test_summary_stg_2_docx()
    _test_summary_md_2_docx()
