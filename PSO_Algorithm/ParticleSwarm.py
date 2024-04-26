import random
import math
from GNN.search_space import HybridSearchSpace
from GNN.gnn_model_manager import GNNModelManager
from PSO_Algorithm.Particle import Particle


class ParticleSwarm:
    def __init__(self, args, x_min, x_max, max_vel):
        self.args = args
        self.x_min = x_min
        self.x_max = x_max
        self.max_vel = max_vel  # 粒子最大速度
        self.best_fitness_value = float('-Inf')
        self.best_position = [0.0 for i in range(self.args.particle_dim)]  # 种群最优位置
        self.fitness_val_list = []  # 每次迭代最优适应值
        self.hybrid_search_space = HybridSearchSpace(self.args.num_gnn_layers)
        self.gnn_manager = GNNModelManager(self.args)
        self.load_training_data()
        # 对种群进行初始化
        self.Particle_list = [Particle(self.x_min, self.x_max, self.max_vel, self.args.particle_dim) for i in
                              range(self.args.particle_num)]
        for part in self.Particle_list:
            value, test_value = self.get_fitness(part.get_pos())
            part.set_fitness_value(value)

    def load_training_data(self):
        self.gnn_manager.load_data(self.args.dataset)
        # dataset statistics
        print(self.gnn_manager.data)

    def set_bestFitnessValue(self, value):
        self.best_fitness_value = value

    def get_bestFitnessValue(self):
        return self.best_fitness_value

    def set_bestPosition(self, i, value):
        self.best_position[i] = value

    def get_bestPosition(self):
        return self.best_position

    def get_fitness(self, particle):  # 适应函数

        net_genes = self.hybrid_search_space.get_net_instance_pso(particle)
        param_genes = self.hybrid_search_space.get_param_instance_pso(particle)
        val_acc, test_acc = self.gnn_manager.train(net_genes, param_genes)
        return val_acc, test_acc

    # 更新速度
    def update_vel(self, part):
        for i in range(self.args.particle_dim):
            ww = self.args.W * part.get_vel()[i]
            vel_value = self.args.c1 * random.random() * (part.get_best_pos()[i] - part.get_pos()[i]) + self.args.c2 * random.random() * (self.get_bestPosition()[i] - part.get_pos()[i])
            if vel_value > self.max_vel[i]:
                vel_value = self.max_vel[i]
            elif vel_value < -self.max_vel[i]:
                vel_value = -self.max_vel[i]
            part.set_vel(i, vel_value)

    # 更新位置
    def update_pos(self, part):
        for i in range(self.args.particle_dim):
            # pos_value = part.get_pos()[i] + part.get_vel()[i]
            if (part.get_vel()[i] > (self.x_max[i] / 1000)) & (part.get_vel()[i] <= 1):
                pos_value = part.get_pos()[i] + 1
            elif (part.get_vel()[i] >= -1) & (part.get_vel()[i] < (-self.x_max[i] / 1000)):
                pos_value = part.get_pos()[i] - 1
            elif (part.get_vel()[i] > 1) & (part.get_vel()[i] <= self.max_vel[i]):
                pos_value = part.get_pos()[i] + math.ceil(part.get_vel()[i])
            elif (part.get_vel()[i] < -1) & (part.get_vel()[i] >= -self.max_vel[i]):
                pos_value = part.get_pos()[i] - math.floor(part.get_vel()[i])
            else:
                pos_value = part.get_pos()[i]

            if pos_value > self.x_max[i]:
                pos_value = pos_value % (self.x_max[i] + 1)
            elif pos_value < self.x_min[i]:
                pos_value = pos_value + self.x_max[i] + 1

            part.set_pos(i, pos_value)
        value, test_value = self.get_fitness(part.get_pos())
        if value > part.get_fitness_value():
            part.set_fitness_value(value)
            for i in range(self.args.particle_dim):
                part.set_best_pos(i, part.get_pos()[i])
        if value > self.get_bestFitnessValue():
            self.set_bestFitnessValue(value)
            for i in range(self.args.particle_dim):
                self.set_bestPosition(i, part.get_pos()[i])

    def update(self):
        # self.load_training_data()

        for i in range(self.args.iterations):
            for part in self.Particle_list:
                self.update_vel(part)  # 更新速度
                self.update_pos(part)  # 更新位置
            self.fitness_val_list.append(self.get_bestFitnessValue())  # 每次迭代完把当前的最优适应度存到列表
            print('第' + str(i + 1) + '轮iteration的fitness:' + str(self.get_bestFitnessValue()))
            print('第' + str(i + 1) + '轮iteration的pos:' + str(self.get_bestPosition()))
            with open("pso_best.txt", "w") as f:
                f.write('第' + str(i + 1) + '轮iteration的fitness:' + str(self.get_bestFitnessValue()) + '\n')
                f.write('第' + str(i + 1) + '轮iteration的参数组合'
                                           ':' + str(self.get_bestPosition()) + '\n')
        return self.fitness_val_list, self.get_bestPosition()
