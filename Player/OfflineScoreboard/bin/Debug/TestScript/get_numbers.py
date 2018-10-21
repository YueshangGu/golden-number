# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# 为了“开箱即用”，本脚本没有依赖除了Python库以外的组件。
# 添加自己的代码时，可以自由地引用如numpy这样的组件以方便编程。

import bisect
import random
import sys
from collections import Counter

import numpy as np
from scipy.optimize import curve_fit


def LineToNums(line, type=float):
    """将输入的行按间隔符分割，并转换为某种数字类型的可迭代对象（默认为float类型）"""
    return (type(cell) for cell in line.split('\t'))


history = []
metaLine = sys.stdin.readline()
lineNum, columnNum = LineToNums(metaLine, int)
for line in map(lambda _: sys.stdin.readline(), range(lineNum)):
    gnum, *nums = LineToNums(line)
    history.append((gnum, nums))


def Mean(iter, len):
    """用于计算均值的帮主函数"""
    return sum(iter) / len


MAX_QUE_SIZE = 120
start = 20
WEIGHT_DECAY = 0.8


def update_index(cnt_que, index):
    return [each + 1 if each >= index else each for each in cnt_que]


def update(pred_golden, cnt_que):
    for each in history[start:]:
        new = each[0]
        index = bisect.bisect(pred_golden, new)
        if index == 0:  # 边界特殊处理
            if pred_golden[0] - new > pred_golden[1] - pred_golden[0]:  # 新增待定黄金点
                pred_golden.insert(index, new)
                if len(cnt_que) == MAX_QUE_SIZE:  # 维持队列长度
                    cnt_que.pop()
                cnt_que = update_index(cnt_que, index)  # 更新cnt_que里的index
                cnt_que.insert(0, index)
            else:
                pred_golden[index] = (1 - WEIGHT_DECAY) * new + WEIGHT_DECAY * pred_golden[index]
                if len(cnt_que) == MAX_QUE_SIZE:
                    cnt_que.pop()
                cnt_que.insert(0, index)
        elif index == len(pred_golden):  # 边界特殊处理
            if new - pred_golden[index - 1] > pred_golden[index - 1] - pred_golden[index - 2]:  # 新增待定黄金点
                pred_golden.insert(index, new)
                if len(cnt_que) == MAX_QUE_SIZE:
                    cnt_que.pop()
                cnt_que = update_index(cnt_que, index)
                cnt_que.insert(0, index)
            else:
                pred_golden[index - 1] = (1 - WEIGHT_DECAY) * new + WEIGHT_DECAY * pred_golden[
                    index - 1]
                if len(cnt_que) == MAX_QUE_SIZE:
                    cnt_que.pop()
                cnt_que.insert(0, index - 1)
        elif pred_golden[index] - new < (pred_golden[index] - pred_golden[index - 1]) / 3.0:
            pred_golden[index] = (1 - WEIGHT_DECAY) * new + WEIGHT_DECAY * pred_golden[index]
            if len(cnt_que) == MAX_QUE_SIZE:
                cnt_que.pop()
            cnt_que.insert(0, index)
        elif new - pred_golden[index - 1] < (pred_golden[index] - pred_golden[index - 1]) / 3.0:
            pred_golden[index - 1] = (1 - WEIGHT_DECAY) * new + WEIGHT_DECAY * pred_golden[index - 1]
            if len(cnt_que) == MAX_QUE_SIZE:
                cnt_que.pop()
            cnt_que.insert(0, index - 1)
        else:  # 新增待定黄金点
            pred_golden.insert(index, new)
            if len(cnt_que) == MAX_QUE_SIZE:
                cnt_que.pop()
            cnt_que = update_index(cnt_que, index)
            cnt_que.insert(0, index)
    return pred_golden, cnt_que


def func(x, a, b, c):  # 前start不采用拟合函数的方法
    return a * np.exp(-b * x) + c


def gauss(pred_golden, index):  # 输出结果随机化
    mean = pred_golden[index]
    if index == 0:
        var = (pred_golden[index + 1] - pred_golden[index]) / 6.0
    elif index == len(pred_golden) - 1:
        var = (pred_golden[index] - pred_golden[index - 1]) / 6.0
    else:
        var = min(pred_golden[index + 1] - pred_golden[index], pred_golden[index] - pred_golden[index - 1]) / 6.0
    return random.gauss(mean, var)


def main():
    if len(history) == 0:
        candidate1 = 50 * 0.618 - abs(random.gauss(0, 5))
        candidate2 = random.gauss(50 * 0.618 * 0.618, 5)
    elif len(history) <= 3:
        candidate1 = Mean(map(lambda h: h[0], history[-5:]), min(len(history), 5))
        candidate2 = candidate1 * 0.618
    elif len(history) < start:
        xs = range(1, len(history) + 1)
        ys = [each[0] for each in history]
        popt, pcov = curve_fit(func, xs, ys)
        a, b, c = popt
        candidate1 = func(len(history) + 1, a, b, c)
        candidate2 = Mean(map(lambda h: h[0], history[-5:]), min(len(history), 5)) * 0.618
    else:
        pred_golden = [history[0][0]]
        cnt_que = [0]
        for each in history[1:start]:
            index = bisect.bisect(pred_golden, each[0])
            pred_golden.insert(index, each[0])
            cnt_que = update_index(cnt_que, index)
            cnt_que.insert(0, index)
        pred_golden, cnt_que = update(pred_golden, cnt_que)
        counts = Counter(cnt_que)

        top_two = counts.most_common(2)
        candidates = []
        for each in top_two:
            candidates.append(gauss(pred_golden, each[0]))
        if len(candidates) < 2:
            candidate1 = candidates[0]
            candidate2 = candidates[0] * 0.618
        else:
            candidate1 = candidates[0]
            candidate2 = candidates[1]

    if random.random() > 0.5:
        tmp = candidate1
        candidate1 = candidate2
        candidate2 = tmp
    print("%.13f\t%.13f" % (candidate1, candidate2))


if __name__ == '__main__':
    main()