import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PSO_Algorithm.Particle import Particle

from PSO_Algorithm.search_space import SearchSpace
from args import reset_pso_parameters
from models import initialModel, aggregator
from norms import Gnorm
from utils import load_data, accuracy, normalize_adj, normalization
from utils import print_args


class ChaoticParticleSwarm:
    def __init__(self, args, x_min, x_max, max_vel):
        self.args = args
        self.x_min = [x_min for i in range(self.args.particle_dim)]
        self.x_max = [x_max for i in range(self.args.particle_dim)]
        self.max_vel = [x_max for i in range(self.args.particle_dim)]  # 粒子最大速度
        self.best_fitness_value = float('-Inf')
        self.best_position = [0 for i in range(self.args.particle_dim)]  # 种群最优位置
        self.fitness_val_list = []  # 每次迭代最优适应值
        self.genGlobalBest = [[0] * self.args.particle_dim for i in range(self.args.iterations)]  # 每一轮迭代的全局最优位置
        self.hybrid_search_space = SearchSpace()

        # 加载数据集
        self.adj, self.features, self.labels, self.idx_train, self.idx_val, self.idx_test, _ = load_data(
            dataset=self.args.dataset)
        self.n_nodes, self.feat_dim = self.features.shape
        self.adj_norm = normalize_adj(self.adj)
        self.labels = torch.LongTensor(self.labels)

        # seed 初始化
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.args.cuda:
            torch.cuda.manual_seed(self.args.seed)

        # feature 初始化
        # particle = normalization(particle)
        x = self.features
        self.xlist = []
        gnorm = Gnorm(self.args.graph_normalization)

        for i in range(self.args.propagation_layers):
            x = torch.spmm(self.adj_norm, x).detach_()
            # x = torch.load("./propagation/ogbn-arxiv/hop" + str(i+1) + ".pth")
            x = gnorm.norm(x)
            # print(y.add_(x))
            # y.add(x * particle[i + 1])
            self.xlist.append(x)


        # 对种群进行初始化
        self.Particle_list = [Particle(self.x_min, self.x_max, self.max_vel, self.args.particle_dim) for i in
                              range(self.args.particle_num)]
        for part in self.Particle_list:
            fit = self.get_fitness(part.get_pos())
            part.set_fitness_value(fit)
            for i in range(self.args.particle_dim):
                part.set_best_pos(i, part.get_pos()[i])
            if fit > self.best_fitness_value:
                self.best_fitness_value = fit
                self.best_position = part.get_pos()




    def get_fitness(self, particle):  # 适应函数


        # transformation_selection, propagation_selection = self.hybrid_search_space.get_instance_pso(particle)
        # reset_pso_parameters(self.args, transformation_selection, propagation_selection)
        np.set_printoptions(precision=3)
        particle = [float('{:.2f}'.format(i)) for i in particle]
        print('============================\n particle={}'.format(particle))

        # particle = normalization(particle)
        y = []
        y.append(self.features)
        gnorm = Gnorm(self.args.graph_normalization)


        for i in range(self.args.propagation_layers):
            # x = torch.spmm(self.adj_norm, x).detach_()
            # # x = torch.load("./propagation/ogbn-arxiv/hop" + str(i+1) + ".pth")
            # x = gnorm.norm(x)
            xfeature = (self.features * particle[i] + (1 - particle[i]) * self.xlist[i])
            y.append(xfeature)

        y = torch.stack(y, dim=0)
        features = aggregator(self.args.aggregator_type, y).detach_()

        model = initialModel(skip_connection=self.args.skip_connections, nfeat=features.shape[1],
                             nhid=self.args.dim_hidden,
                             nclass=self.labels.max().item() + 1,
                             nlayer=self.args.num_layers,
                             act=self.args.activation_function,
                             dropout=self.args.dropout)

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.args.lr, weight_decay=self.args.weight_decay)

        device = torch.device(f"cuda:{self.args.device}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        features = features.to(device)
        labels = self.labels.to(device)
        t_total = time.time()
        best_val = 0
        best_test = 0
        for epoch in range(self.args.epochs):
            model.train()
            optimizer.zero_grad()
            output = model(features)
            loss_train = F.nll_loss(output[self.idx_train], labels[self.idx_train])
            acc_train = accuracy(output[self.idx_train], labels[self.idx_train])
            loss_train.backward()
            optimizer.step()

            model.eval()
            output = model(features)

            acc_val = accuracy(output[self.idx_val], labels[self.idx_val])
            acc_test = accuracy(output[self.idx_test], labels[self.idx_test])
            if acc_val > best_val:
                best_val = acc_val
                best_test = acc_test

        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print(f'Best val: {best_val:.4f}, best test: {best_test:.4f}')

        return best_test

    # 更新速度
    def update_vel(self, part):
        for i in range(self.args.particle_dim):
            ww = self.args.W * part.get_vel()[i]
            vel_value = ww + self.args.c1 * random.random() * (
                    part.get_best_pos()[i] - part.get_pos()[i]) + self.args.c2 * random.random() * (
                                self.best_position[i] - part.get_pos()[i])
            if vel_value > self.max_vel[i]:
                vel_value = self.max_vel[i]
            elif vel_value < -self.max_vel[i]:
                vel_value = -self.max_vel[i]
            part.set_vel(i, vel_value)

    # 更新位置
    def update_pos(self, part):
        for i in range(self.args.particle_dim):

            pos_value = part.get_pos()[i] + part.get_vel()[i]

            if pos_value > self.x_max[i]:
                pos_value = pos_value % self.x_max[i]
            elif pos_value < self.x_min[i]:
                pos_value = pos_value + self.x_max[i]

            part.set_pos(i, pos_value)

        fitness = self.get_fitness(part.get_pos())
        if fitness > part.get_fitness_value():
            part.set_fitness_value(fitness)
            for i in range(self.args.particle_dim):
                part.set_best_pos(i, part.get_pos()[i])
        if fitness > self.best_fitness_value:
            self.best_fitness_value = fitness
            self.best_position = part.get_pos()

    def update(self):
        # self.load_training_data()

        avgofdimension = [0.0 for i in range(self.args.particle_dim)]
        pm = 0.45
        exnon = 0.01
        x = 0.6
        pdavg = [0.0 for i in range(self.args.particle_dim)]
        variance = [0.0 for i in range(self.args.particle_dim)]

        for i in range(self.args.iterations):

            avgofdistance = 0.0  # 计算δ(pBest,gBest)  Eq.10
            for part in self.Particle_list:
                dist = 0
                for dim in range(self.args.particle_dim):
                    dist += math.pow((self.best_position[dim] - part.get_best_pos()[dim]), 2)
                avgofdistance += math.sqrt(dist)
            avgofdistance /= self.args.particle_num

            # for dim in range(self.args.particle_dim):  # 计算每一轮迭代的全局最优位置
            #     self.genGlobalBest[i][dim] = self.best_position[dim]

            if avgofdistance < 2:  # δd 设为 0.6

                # gBest 扰动维度距离计算
                for dim in range(self.args.particle_dim):  # 计算δ(gBest(particle/d)) Eq.12
                    avgofdimension[dim] = 0.0
                    for j in range(i + 1):
                        avgofdimension[dim] += math.pow((self.genGlobalBest[j][dim] - self.genGlobalBest[i][dim]), 2)
                    avgofdimension[dim] = math.sqrt(avgofdimension[dim] / (i + 1))

                # gBest 扰动判断和操作
                for dim in range(self.args.particle_dim):
                    if avgofdimension[dim] < 0.6:  # εδ(gbest)∈[0,10^-10]
                        x = 4 * x * (1 - x)
                        r6 = random.random()
                        r1 = 32767
                        if exnon < pm - r6:
                            timesequence = int(x * r1) % (self.x_max[dim])
                            self.best_position[dim] = (self.best_position[dim] + timesequence) % (self.x_max[dim])
                            print('第' + str(dim) + '维进行gBest扰动操作')
                        elif r6 > pm:
                            self.best_position[dim] = self.best_position[dim] - random.randint(0, self.x_max[dim])
                            if self.best_position[dim] < self.x_min[dim]:
                                self.best_position[dim] = self.best_position[dim] + self.x_max[dim]
                            print('第' + str(dim) + '维进行gBest扰动操作')

                # pBest 扰动维度距离计算

                # 计算每维位置的平均值
                for dim in range(self.args.particle_dim):
                    pdavg[dim] = 0.0
                    for part in self.Particle_list:
                        pdavg[dim] += part.get_best_pos()[dim]
                    pdavg[dim] /= self.args.particle_num

                # 计算每个位置的方差
                for dim in range(self.args.particle_dim):
                    variance[dim] = 0.0
                    for part in self.Particle_list:
                        variance[dim] += math.pow((part.get_best_pos()[dim] - pdavg[dim]), 2)
                    variance[dim] = math.sqrt(variance[dim] / self.args.particle_num)

                # pBest扰动
                for dim in range(self.args.particle_dim):
                    if variance[dim] < 0.9:  # εδ(pbest)∈[0,10^-10]
                        for part in self.Particle_list:
                            r3 = random.random()
                            if exnon < pm - r3:
                                part.set_best_pos(dim,
                                                  (part.get_best_pos()[dim] + random.randint(0, self.x_max[dim])) % (
                                                      self.x_max[dim]))
                                print(str(part) + '粒子的第' + str(dim) + '维进行pBest扰动操作')
                            elif r3 > pm:
                                part.set_best_pos(dim, part.get_best_pos()[dim] + self.x_max[dim])
                                print(str(part) + '粒子的第' + str(dim) + '维进行pBest扰动操作')

            for part in self.Particle_list:
                self.update_vel(part)  # 更新速度
                self.update_pos(part)  # 更新位置
            self.fitness_val_list.append(self.best_fitness_value)  # 每次迭代完把当前的最优适应度存到列表
            print(f'fitness:{self.best_fitness_value:.4f}')
            print('第' + str(i + 1) + '轮iteration的pos:' + str(self.best_position))
            # print('第' + str(i + 1) + '轮iteration的params:' + str(self.best_params))
            #             # print('第' + str(i + 1) + '轮iteration的time:' + str(self.best_time))
            #             # print('第' + str(i + 1) + '轮iteration的acc:' + str(self.best_acc))
        return self.fitness_val_list, self.best_position
