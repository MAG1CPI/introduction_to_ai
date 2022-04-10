# -*- coding: utf-8 -*-

import random
from copy import deepcopy
from cmath import sqrt, log
from board import Board
from game import Game


class RandomPlayer:
    '''
    随机落子玩家
    '''

    def __init__(self, color: 'str') -> None:
        '''
        初始化
        :param color: 持子颜色
        '''
        self.color = color

    def random_choice(self, board: 'Board') -> 'str':
        '''
        从合法落子位置中随机选一个落子位置
        :param board: 棋盘
        :return: 随机合法落子位置, e.g. 'A1' 
        '''
        action_list = list(board.get_legal_actions(self.color))

        if len(action_list) == 0:
            return None
        else:
            return random.choice(action_list)

    def get_move(self, board: 'Board') -> 'str':
        '''
        根据当前棋盘状态获取落子位置
        :param board: 棋盘
        :return: action 落子位置, e.g. 'A1'
        '''
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))

        action = self.random_choice(board)

        return action


class HumanPlayer:
    '''
    人类玩家
    '''

    def __init__(self, color: 'str') -> None:
        '''
        初始化
        :param color: 持子颜色
        '''
        self.color = color

    def get_move(self, board: 'Board') -> 'str':
        '''
        根据当前棋盘输入人类合法落子位置
        :param board: 棋盘
        :return: 人类下棋落子位置
        '''
        if self.color == "X":
            player = "黑棋"
        else:
            player = "白棋"

        # 人类玩家输入落子位置，如果输入 'Q', 则返回 'Q'并结束比赛。
        # 如果人类玩家输入棋盘位置，e.g. 'A1'，
        # 首先判断输入是否正确，然后再判断是否符合黑白棋规则的落子位置
        while True:
            action = input(
                "请'{}-{}'方输入一个合法的坐标(e.g. 'D3'，若不想进行，请务必输入'Q'结束游戏。): ".format(player, self.color))

            if action == "Q" or action == 'q':
                return "Q"
            else:
                row, col = action[1].upper(), action[0].upper()

                if row in '12345678' and col in 'ABCDEFGH':
                    # 检查人类输入是否为符合规则的可落子位置
                    if action in board.get_legal_actions(self.color):
                        return action
                else:
                    print("你的输入不合法，请重新输入!")


class UCTNode:
    '''
    UCT算法节点
    '''

    def __init__(self, color: 'str', parent: 'UCTNode', state: 'Board', action: 'str' = None) -> None:
        '''
        初始化
        :param color: 节点颜色('X'/'O')
        :param parent: 节点父母
        :param state: 节点状态
        :param action: 转移至该节点的动作(e.g.,'C3')
        '''
        # 访问次数和奖励
        self.visitTimes = int(0)
        self.reward = int(0)
        # 节点颜色
        self.color = color
        # 节点父母和孩子
        self.parent = parent
        self.children = []
        # 节点所代表的状态和到达该状态的动作
        self.state = state
        self.action = action

    def IsFull(self) -> bool:
        '''
        判断节点孩子是否已满
        :return: 满(true)/未满(false)
        '''
        actionList = list(self.state.get_legal_actions(self.color))
        return len(self.children) == len(actionList)

    def AddChild(self, color: 'str', state: 'Board', action: 'str') -> None:
        '''
        为节点增加一个孩子
        :param color: 孩子颜色('X'/'O')
        :param state: 孩子状态
        :param action: 转移到孩子的动作(e.g.,'C3')
        '''
        childNode = UCTNode(color, self, state, action)
        self.children.append(childNode)


class AIPlayer:
    '''
    AI玩家
    '''

    def __init__(self, color: 'str') -> None:
        '''
        初始化
        :param color: 持子颜色
        '''
        self.color = color
        # 超参
        self.SCALAR = 1.414
        # 最大执行次数
        self.COUNT = 50
        # 当前搜索树根
        self.node = None

    def get_move(self, board: 'Board') -> 'str':
        '''
        根据当前棋盘状态获取最佳落子位置
        :param board: 棋盘
        :return: action 最佳落子位置, e.g., 'C3'
        '''
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))
        # 如果搜索树存在, 将self.node重新指向该树上对应的节点, 然后根据该树进行搜索
        if self.node != None:
            for child in self.node.children:
                for gchild in child.children:
                    if board == gchild.state:
                        self.node = gchild
                        action = self.UCTSearch(self.node)
                        return action
        # 如果搜索树不存在, 将新建一个树进行搜索
        state = deepcopy(board)
        self.node = UCTNode(self.color, None, state)
        action = self.UCTSearch(self.node)
        return action

    def UCTSearch(self, node: 'UCTNode') -> 'str':
        '''
        UCT算法获取最优落子位置
        :param node: 当前节点
        :return: 最优动作, e.g., 'C3'
        '''
        # 重复进行self.COUNT次选择拓展模拟回溯。
        for i in range(self.COUNT):
            # 选择+拓展
            expandNode = self.SelectPolicy(node)
            # 模拟+回溯 -> 3次 防止模拟结果过于极端, 影响后续选择
            for j in range(3):
                reward = self.SimulatePolicy(expandNode)
                self.BackPropagate(expandNode, reward)
        # 获得并返回最优动作
        bestChild = self.UCB1(node, 0)
        return bestChild.action

    def SelectPolicy(self, node: 'UCTNode') -> 'UCTNode':
        '''
        选择 + 拓展: 找到叶子节点或有未被拓展节点, 然后拓展
        :param node: 选择的起始节点
        :return: 拓展的节点
        '''
        # 只要 不是终局 & 孩子已满 就一直向下选择
        while not self.TerminalTest(node.state) and node.IsFull():
            # 没有孩子, 但不是终局 -> 出现了'过', 此时即将结束 -> 退出
            if len(node.children) == 0:
                break
            node = self.UCB1(node, self.SCALAR)
        # 拓展
        return self.Expand(node)

    def Expand(self, node: 'UCTNode') -> 'UCTNode':
        '''
        拓展: 选择将要拓展的节点并返回
        :param node: 选择的节点
        :return: 拓展节点
        '''
        # case 1: 终局, 即叶子节点 -> 无需拓展, 返回节点自己
        if self.TerminalTest(node.state):
            return node
        # 所有可行动作中随机选择一个未被执行的动作
        actionList = list(node.state.get_legal_actions(node.color))
        # case 2: 当前颜色没有可行动作, 即“过” -> 不拓展, 返回节点自己
        if len(actionList) == 0:
            return node
        # case 3：当前节点有可行动作 -> 拓展, 返回一个未被拓展的孩子节点
        triedAction = [child.action for child in node.children]
        actionList = list(set(actionList) - set(triedAction))
        action = random.choice(actionList)
        # 为该节点添加一个孩子并返回
        newColor = 'O'if node.color == 'X' else 'X'
        newState = deepcopy(node.state)
        newState._move(action, node.color)
        node.AddChild(newColor, newState, action)
        return node.children[-1]

    def UCB1(self, node: 'UCTNode', SCALAR: 'float') -> 'UCTNode':
        '''
        选择最优子节点
        :param node: 选择要拓展后继的节点
        :param SCALAR: UCT算法超参数
        :return: 最优子节点
        '''
        # 当前节点没有孩子 -> 返回当前节点
        if len(node.children) == 0:
            return node

        bestChild = []
        bestUCB = -float('inf')
        # 找到所有最大UCB值的节点, 并从中选择一个并返回
        for child in node.children:
            # 有节点未被访问过
            if child.visitTimes == 0:
                return child
            tempUCB = float(child.reward)/child.visitTimes + \
                SCALAR * sqrt(2 * log(node.visitTimes)/child.visitTimes).real
            if bestUCB < tempUCB:
                bestChild = [child]
                bestUCB = tempUCB
            elif bestUCB == tempUCB:
                bestChild.append(child)

        return random.choice(bestChild)

    def SimulatePolicy(self, node: 'UCTNode') -> 'int':
        '''
        模拟: 模拟拓展搜索树直到找到一个终止节点
        :param node: 拓展的节点
        :return: 奖励
        '''
        board = deepcopy(node.state)
        color = node.color
        # 模拟拓展搜索树直到找到终止节点
        while not self.TerminalTest(board):
            actionList = list(board.get_legal_actions(color))
            # 当前颜色玩家有可行动位置就行动, 否则跳过行动, 即'过'
            if len(actionList) != 0:
                action = random.choice(actionList)
                board._move(action, color)
            # 切换玩家颜色
            color = 'O'if color == 'X' else 'X'
        # 获取模拟结果
        winner, score = board.get_winner()
        reward = 100 + score
        # 计算并返回模拟奖励
        if winner == 2:
            reward = 50
        elif 'XO'[winner] != self.color:
            reward = -reward
        return reward

    def BackPropagate(self, node: 'UCTNode', reward: 'int') -> None:
        '''
        反向传播: 回溯并更新路径
        :param node: 开始回溯的节点
        :param reward: 通过模拟获得的期望值
        '''
        while node is not None:
            node.visitTimes += 1
            # 同色减, 异色加
            if node.color == self.color:
                node.reward -= reward
            else:
                node.reward += reward
            node = node.parent

    def TerminalTest(self, state: 'Board') -> 'bool':
        '''
        终局判断
        :param state: 当前棋盘状态
        :return: 已结束(true)/未结束(false)
        '''
        black = list(state.get_legal_actions('X'))
        white = list(state.get_legal_actions('O'))
        return len(black) == 0 and len(white) == 0


if __name__ == '__main__':
    # 设置玩家
    black_player = AIPlayer("X")
    white_player = RandomPlayer("O")
    # 游戏初始化，第一个玩家是黑棋，第二个玩家是白棋
    game = Game(black_player, white_player)
    # 开始下棋
    game.run()
