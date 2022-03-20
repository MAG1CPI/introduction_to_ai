# 逻辑编程库
from kanren import run, eq, membero, var, conde
from kanren.core import lall


#########################辅助函数###########################
# 在列表list中，c的右边为r
def right(c, r, list):
    return membero((c, r), zip(list, list[1:]))

# 在列表list中，a与b相邻
def Next(a, b, list):
    return conde([right(a, b, list)], [right(b, a, list)])
############################################################


# 推理智能体
class Agent:
    # 初始化
    def __init__(self):
        # 实例集
        self.units = var()
        # 规则集
        self.rules_zebraproblem = None
        # 解
        self.solutions = None

    # 定义逻辑规则
    def define_rules(self):
        self.rules_zebraproblem = lall(
            # 实例集定义，五个unit成员
            (eq,        (var(), var(), var(), var(), var()), self.units),

            # 实例属性，各个unit房子又包含五个成员属性: (国家，工作，饮料，宠物，颜色)
            # 实例的属性
            (membero, ('英国人', var(), var(), var(), '红色'), self.units),
            (membero, ('西班牙人', var(), var(), '狗', var()), self.units),
            (membero, ('日本人', '油漆工', var(), var(), var()), self.units),
            (membero, ('意大利人', var(), '茶', var(), var()), self.units),
            (membero, (var(), '摄影师', var(), '蜗牛', var()), self.units),
            (membero, (var(), '外交官', var(), var(), '黄色'), self.units),
            (membero, (var(), var(), '咖啡', var(), '绿色'), self.units),
            (membero, (var(), '小提琴家', '橘子汁', var(), var()), self.units),

            # 实例在解中的位置
            (eq,    (('挪威人', var(), var(), var(), var()), var(), var(), var(), var()), self.units),
            (eq,    (var(), var(), (var(), var(), '牛奶', var(), var()), var(), var()), self.units),

            # 实例与实例直接位置的关系
            (right, (var(), var(), var(), var(), '绿色'), (var(), var(), var(), var(), '白色'), self.units),
            (Next,  ('挪威人', var(), var(), var(), var()), (var(), var(), var(), var(), '蓝色'), self.units),
            (Next,  (var(), var(), var(), '狐狸', var()), (var(), '医生', var(), var(), var()), self.units),
            (Next,  (var(), var(), var(), '马', var()), (var(), '外交官', var(), var(), var()), self.units),

            # 补充解集
            (membero, (var(), var(), var(), '斑马', var()), self.units),
            (membero, (var(), var(), '矿泉水', var(), var()), self.units)
        )

    # 规则求解器
    def solve(self):
        self.define_rules()
        self.solutions = run(0, self.units, self.rules_zebraproblem)
        return self.solutions


def main():
    agent = Agent()
    solutions = agent.solve()

    print("答案如下：")
    print([house for house in solutions[0] if '斑马' in house][0][4],
          "房子里的人养斑马", sep="")
    print([house for house in solutions[0] if '矿泉水' in house][0][4],
          "房子里的人喜欢喝矿泉水", sep="")
    print("解如下：")
    for i in range(5):
        print(solutions[0][i])


main()
