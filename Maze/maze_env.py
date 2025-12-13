class MazeEnvironment:
    def __init__(self, maze):
        self.maze = maze
        self.height = len(maze)
        self.width = len(maze[0])
        self.start = None
        self.goal = None
        self.agent_pos = None

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos[0]*self.width+self.agent_pos[1]

    def step(self, action):
        new_pos = self.get_new_position(self.agent_pos, action)
        reward, done = self.get_reward_done(new_pos)
        self.agent_pos = new_pos
        return new_pos[0]*self.width+new_pos[1], reward, done

    def get_new_position(self, pos, action):
        if pos is None:
            return None

        row, col = pos
        if action == 0:
            row -= 1
        elif action == 1:
            row += 1
        elif action == 2:
            col -= 1
        elif action == 3:
            col += 1

        # Check if new position is out of bounds
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return pos

        if self.maze[row][col] == 1:  # Hit an obstacle
            return pos

        return row, col

    def get_reward_done(self, pos):
        if pos is None:
            return -1, True

        row, col = pos
        if self.maze[row][col] == 1:  # Hit an obstacle
            return -1, False
        elif self.maze[row][col] == 2:  # Reached the start point
            return -1, False
        elif self.maze[row][col] == 3:  # Reached the goal
            return 0, True
        else:
            return -1, False  # Moving to an empty cell

    def render(self):
        print("迷宫示例图：")
        for row in range(self.height):
            for col in range(self.width):
                print(self.maze[row][col], end=' ')
            print()

    def new_render(self):
        print("智能体所处迷宫中的位置(A)：")
        for row in range(self.height):
            for col in range(self.width):
                if (row, col) == self.agent_pos:
                    print('A', end=' ')
                else:
                    print(self.maze[row][col], end=' ')
            print()


# 迷宫地图示例
# maze_map = [
#     [0, 0, 1, 0],
#     [0, 1, 0, 0],
#     [0, 2, 3, 0],
#     [0, 0, 0, 0],
# ]

# maze_map = [[2, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
#             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
#             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
#             [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#             [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
#             [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#             [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
#             [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
#             [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 3],
#             ]
#
# # 创建迷宫环境
# maze_env = MazeEnvironment(maze_map)
#
# # 设置起始位置和目标位置
# maze_env.start = (0, 0)
# maze_env.goal = (12, 12)
#
# # 重置环境，并获取起始位置
# start_position = maze_env.reset()
#
# # 渲染迷宫地图
# maze_env.render()
# maze_env.new_render()
#
# # 运行一些步骤
# actions = ['left', 'right', 'down', 'down']
# for action in actions:
#     new_position, reward, done = maze_env.step(action)
#     print(done)
#     maze_env.new_render()
#     if done:
#         break
