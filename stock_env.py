from __future__ import annotations

from typing import SupportsFloat, Any

import gymnasium as gym
import pandas as pd
from gymnasium.core import ObsType, ActType, RenderFrame
from gymnasium.vector.utils import spaces


class StockEnv(gym.core.Env):

    def __init__(self):
        """
        构造方法
        """
        # 读取股票数据
        self.stock_data = pd.read_csv('stock_data.csv').iloc[::-1]
        # 初始时刻的index
        self.stock_index = 0
        # 仓位
        # 初始时刻的仓位
        # position_index：股票仓位的index，用以记录买入价
        # position_stock：股票仓位
        self.stock_position = {'position_index': 0, 'position_stock': 0}

        # 动作空间
        self.action_space = spaces.Discrete(n=11, start=-5)
        # 观察空间
        self.observation_space = self.stock_data
        # 随机数种子
        self.np_random = 0

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.stock_index = self.stock_index + 1

        # 新的观测值
        observation = self.stock_data.iloc[self.stock_index]

        # 计算奖励
        # 假如都以收盘价进行计算
        # 原市值
        pre_value = self.stock_data.iloc[self.stock_position['position_index']]['close'] * self.stock_position[
            'position_stock']
        # 更新股票仓位
        position_index_val = self.stock_index
        self.stock_position['position_index'] = position_index_val
        position_stock_val = self.stock_position['position_stock'] + action
        self.stock_position['position_stock'] = position_stock_val
        # 新市值
        new_value = self.stock_data.iloc[self.stock_position['position_index']]['close'] * self.stock_position[
            'position_stock']
        # 奖励
        reward = new_value - pre_value

        # terminated
        terminated = False
        if (self.stock_data.shape[0] - 1) == self.stock_index:
            terminated = True

        # truncated
        # 可以设置爆仓的话，truncated为True，在本文永远是False
        truncated = False

        # info
        info = {}

        return observation, reward, terminated, truncated, info

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        # 初始时刻的index
        self.stock_index = 0
        # 仓位
        self.stock_position = {'position_index': 0, 'position_stock': 0}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        value = self.stock_data.iloc[self.stock_position['position_index']]['close'] * self.stock_position[
            'position_stock']
        print('value ', value)

    def close(self):
        pass
