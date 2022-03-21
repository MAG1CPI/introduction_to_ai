import copy                  # 深度拷贝方法
from board import Chessboard


# 基于棋盘类的搜索策略
class Game:
    # 初始化
    def __init__(self, show=True):
        self.chessBoard = Chessboard(show)
        self.solves = []

    # 重置游戏
    def gameInit(self, show=True):
        self.chessBoard.boardInit(show)

    # 递归放置皇后
    def setQueens(self, row, r=[]):
        # 遍历每一列
        for i in range(8):
            # 重置
            self.chessBoard.boardInit(False)
            for k in range(row):
                self.chessBoard.setQueen(k, r[k], False)
            # 合法则设置皇后
            if self.chessBoard.isLegal(row, i):
                self.chessBoard.setQueen(row, i, False)
                r.append(i)
                # 当前已经是第8行，则添加答案
                if row == 7:
                    self.solves.append(copy.deepcopy(r))
                else:
                    self.setQueens(row + 1, r)
                r.pop()

    # 搜索
    def run(self):
        self.setQueens(0)

    # 结果显示
    def showResults(self, result):
        self.chessBoard.boardInit(False)
        for i, item in enumerate(result):
            if item >= 0:
                self.chessBoard.setQueen(i, item, False)

        self.chessBoard.printChessboard(False)

    # 获取结果
    def get_results(self):
        self.run()
        return self.solves


def main():
    game = Game(False)
    solutions = game.get_results()

    print('There are {} results.'.format(len(solutions)))
    '''
    for i in range(len(solutions)):
        game.showResults(solutions[i])
    '''


main()
