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
from gridworld import *


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
        """
        self.state = self.env.reset()
        s_name = self.trans_state_to_key(self.state)
        self._insert_state_in_Q(s_name, randomized=False)

    def trans_state_to_key(self, state):
        """
        :param state: 网格位置 int
        :return:
        """
        return str(state)

    def check_state_in_Q(self, s):
        """
        check if state in Q table
        :return:
        """
        return s in self.Q.keys()

    def _init_state_value(self, s_name, randomized=True):
        """
        使用状态名字s_name初始化状态Q-table
        :param s_name:
        :param randomized:
        :return:
        """
        if not self.check_state_in_Q(s_name):
            self.Q[s_name] = {}
            for action in range(self.env.action_space.n):
                default_value = random()/10 if randomized else 0.0
                self.Q[s_name][action] = default_value
        else:
            pass

    def _insert_state_in_Q(self, s, randomized=True):
        """
        if state not in Q, insert s to Q-table
        :param s:
        :param randomized:
        :return:
        """
        if not self.check_state_in_Q(s):
            self._init_state_value(s, randomized)
        else:
            pass

    def get_Q_value(self, s, a):
        """
        :param s: state
        :param a: action
        :return: Q(s,a)
        """
        self._insert_state_in_Q(s, randomized=True)
        return self.Q[s][a]

    def set_Q_value(self, s, a, value):
        """
        set Q(s,a)
        :param s: state
        :param a: action
        :param value: Q(s,a)
        """
        self._insert_state_in_Q(s, randomized=True)
        self.Q[s][a] = value

    def current_policy(self, s, episode_num, use_epsilon=True):
        """

        :param s: current state
        :param episode_num:episode 次数
        :param use_epsilon: 是否使用探索策略
        :return: action
        """
        epsilon = 1.00/(episode_num+1)
        Q_s = self.Q[s]
        rand_value = random()
        action = None
        if use_epsilon and rand_value < epsilon:
            action = self.env.action_space.sample()
        else:
            act_str = max(Q_s, key=Q_s.get)
            action = int(act_str)
        return action

    def action_from_policy(self, s, episode_num, use_epsilon=True):
        return self.current_policy(s=s, episode_num=episode_num,use_epsilon=use_epsilon)

    def act(self,a):
        """
        env do with action a
        :param a:
        :return: state, reward, isdone, info
        """
        return self.env.step(a)

    def learning(self, gamma, alpha, max_episode_num):
        """

        :param gamma:
        :param alpha:
        :param max_episonde_num:
        :return:
        """
        num_episode = 0


        while num_episode<max_episode_num:
            self.state = self.env.reset()
            s0 = self.trans_state_to_key(self.state)
            self.env.render()
            a0 = self.action_from_policy(s0, num_episode,use_epsilon=True)
            # every episode step counter
            step_in_episode = 0
            # game world status
            is_done = False
            while not is_done:
                s1, r1, is_done, info = self.act(a0)
                self.env.render()
                s1 = self.trans_state_to_key(s1)
                self._insert_state_in_Q(s1, randomized=True)
                a1 = self.action_from_policy(s1,num_episode,use_epsilon=True)
                q_old = self.get_Q_value(s0, a0)
                q_prime = self.get_Q_value(s1, a1)
                step_decrase = r1 + gamma * q_prime - q_old
                q_new = q_old + alpha * step_decrase
                self.set_Q_value(s0, a0, q_new)
                s0, a0 = s1, a1
                step_in_episode += 1

            print "Episode {0} takes {1} steps".format(num_episode, step_in_episode)
            num_episode += 1

        # saving Q-table

        return




def main():

    # env = gym.make("WindyGridWorld-v0")
    # directory = "/home/qiang/workspace/reinforce/python/monitor"
    #
    # env = gym.wrappers.Monitor(env, directory, force=True)
    env = CliffWalk()
    agent = SampleQLearnAgent(env)
    env.reset()
    print("Learning...")
    agent.learning(gamma=0.9,
                   alpha=0.1,
                   max_episode_num=500)


if __name__ == "__main__":
    main()