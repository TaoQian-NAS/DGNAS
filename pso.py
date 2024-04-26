import numpy as np

from PSO_Algorithm.ChaoticPSO import ChaoticParticleSwarm
from args import initialize


def make_print_to_file(path='./', dataset=""):
    import os
    import sys
    import datetime

    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = datetime.datetime.now().strftime('day' + '%Y_%m_%d_' + dataset + "_full_3")
    sys.stdout = Logger(fileName + '.log', path=path)


if __name__ == "__main__":
    args = initialize()
    make_print_to_file(path='./logs/', dataset=args.dataset)

    x_min = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    x_max = np.array([4, 3, 7, 5, 3, 11, 3, 6, 4, 4])
    m_vel = np.array([2, 2, 3, 4, 2, 3, 2, 3, 2, 2])

    pso = ChaoticParticleSwarm(args, x_min, x_max, m_vel)
    pso.update()
