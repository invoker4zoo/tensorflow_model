# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechart: whatshowlove
@ software: PyCharm
@ file: agent.py
@ time: $17-8-16 下午2:21
"""
import gym
from gym import Env
from random import random


class SampleQLearnAgent(object):
    def __init__(self, env):
        self.env = env
        ## {s0:[],s1:[]}
        ## 对于格子世界，每一个状态为格子位置编码
        ## 对于open ai gym ,每一个状态为一个状态矩阵
        self.Q = {}
        self._init_agent()
        self.state = None

    def _init_agent(self):
        """

        :return:
        """
        pass

    def trans_state_to_key(self, state):
        """
        :param state: 网格位置 int
        :return:
        """
        return str(state)

    def check_state_in_Q(self, s):
        """
        :return:
        """
        return s in self.Q.keys()