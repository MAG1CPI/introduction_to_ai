import numpy as np

# 棋盘类
class Chessboard:
    # 初始化
    def __init__(self, show = True):
        self.boardInit(show)
    
    # 重置棋盘
    def boardInit(self, show = True):
        # 棋盘 矩阵
        self.chessboardMatrix = np.zeros((8,8),dtype=np.int8)
        # 禁止落子位置 矩阵
        self.unableMatrix = np.zeros((8,8),dtype=np.int8)
        # 皇后位置 矩阵
        self.queenMatrix = np.zeros((8,8),dtype=np.int8)
        # 棋盘绘制 矩阵
        printMatrix = np.zeros((9,9),dtype=np.int8)              
        for i in range(8):
            printMatrix[0][i+1] = i
            printMatrix[i+1][0] = i
        self.printMatrix = printMatrix
        # 打印
        if show:
            self.printChessboard()

    # 棋盘符号转换
    def trans(self, x):
        return {0:'-',-1:'x',1:'o'}.get(x)

    # 落子是否在棋盘上
    def isOnChessboard(self, x,y):
        return (x*(7-x) >=0 and y*(7-y) >=0)
    
    # 棋盘上落子合法性
    def isLegal(self, x, y):
        return (self.unableMatrix +1)[x][y]

    # 落子
    def setQueen(self, x, y, show = True):
        if self.isLegal(x, y):
            # 设置 皇后位置矩阵
            self.queenMatrix[x][y] =1
            # 设置 禁止落子位置矩阵
            for i in range(8):
                self.unableMatrix[x][i] =-1
                self.unableMatrix[i][y] =-1
            for i in range(-7,8):
                if self.isOnChessboard(x+i,y+i):
                    self.unableMatrix[x+i][y+i] =-1
                if self.isOnChessboard(x+i,y-i):
                    self.unableMatrix[x+i][y-i] =-1
            # 设置 棋盘矩阵
            self.chessboardMatrix = self.unableMatrix +2*self.queenMatrix
            # 设置 棋盘绘制矩阵
            self.printMatrix[1:9,1:9] = self.chessboardMatrix
            # 打印
            if show:
                self.printChessboard()
            return True
        else:
            print('落子失败')
            return False
    
    # 绘制棋盘
    def printChessboard(self, showALL = True):
        # 同时显示皇后和禁止落子位 或 只显示皇后
        if showALL:
            Board = self.printMatrix
        else:
            Board = self.printMatrix
            Board[1:9,1:9] = self.queenMatrix
            
        for i in range(9):
            for j in range(9):
                if i+j==0:
                    print('  ', end='')
                elif (i==0 and j!=8) or j==0 :
                    print(str(Board[i][j])+' ', end='')
                elif i==0 and j==8:
                    print(str(Board[i][j])+' ')
                elif j!=8:
                    print(self.trans(Board[i][j])+' ', end='')
                else:
                    print(self.trans(Board[i][j])+' ')
    
    
    ##############################玩家互动##############################
    # 互动函数
    def play(self):
        while True:
            action = input("请输入一个合法的坐标(e.g. '2-3'，若想重新开始，请打出'init'，若想退出，请打出'Q'。): ")

            if action == 'init':
                self.boardInit()
                continue
            if action == 'Q' or action == 'q' or action == 'quit' or action == 'QUIT':
                break

            if '-' in action:
                x,y =action.split('-')
            elif ',' in action:
                x,y =action.split(',')
            else:
                x = action[0]
                y = action[-1]
            
            if x.isdigit() and y.isdigit():
                self.setQueen(int(x),int(y))

                if self.isWin():
                    print('Win!')
                    self.printMatrix[1:9,1:9] = self.queenMatrix
                    self.printChessboard()
                    action = input("输入'Y'进入下一局,输入其它任意值退出: ")
                    if action == 'y' or action == 'Y':
                        self.boardInit()
                        self.Play()
                    else:
                        break
                elif self.isLose():
                    print('Lose!')
                    self.printMatrix[1:9,1:9] = self.queenMatrix
                    self.printChessboard()
                    action = input("输入'Y'进入下一局,输入其它任意值退出: ")
                    if action == 'y' or action == 'Y':
                        self.boardInit()
                        self.play()
                    else:
                        break
            else:
                print('请输入合法坐标或指定命令.')

    # 胜利条件判断
    def isWin(self):
        if sum(sum(self.queenMatrix)) ==8:
            return True
        else:
            return False

    # 失败条件判断
    def isLose(self):
        if not self.isWin() and sum(sum(self.unableMatrix)) ==-64:
            return True
        else:
            return False
    ###################################################################