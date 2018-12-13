import sys
import gym
import math
import numpy as np

class Node():

    def __init__(self, action, reward, parent, **kwargs):
        """
        initialize node class with
        - action: the action that lead to node
        - reward: value of node
        - parent: parent node
        - state (optional): current environment state
        """
        self.visits = 0
        self.reward = reward
        self.action = action
        self.parent = parent
        self.state = kwargs['state']
        self.childs = []

    def value(self):
        """
        mean value of node
        """
        return self.reward / max(self.visits, 1)

    def get_ucb(self, weight = 1):
        """
        compute ucb1 value of node
        """
        if self.parent == None: ucb1 = self.reward / max(self.visits, 1)
        else: ucb1 = self.reward / max(self.visits, 1) + weight * math.sqrt(2*math.log(self.parent.visits)/ max(self.visits, 1))
        return ucb1

class Interface():

    def get_child(self, node):
        """
        return a tuple (end, child)
        end is a boolean value set to True if end of an episode
        child is a child from node which is not expanded yet
        """
        raise NotImplementedError( "Should have implemented this" )

    def run(self, node, depth):
        """
        unroll simulation from
        - node: the start node
        - with depth: the rollout depth
        it shall return a tuple(mean, max)
        mean is mean value of nodes in the rollout
        max is max value of nodes in the rollout
        """
        raise NotImplementedError( "Should have implemented this" )

class Mcts():

    def __init__(self, interface, root, rollout_depth = 5, max_step = 200, mixmax_weight = 0.):
        """
        initialize mcts:
        - interface is a template for simulation
        - root is the starting node of the tree
        - rollout_depth is the depth of Monte Carlo rollouts
        - max_step is the maximum number of node expansion
        - mixmax_weight is a weighting beetween mean value and max value encountered in rollouts
        """
        self.interface = interface
        self.root = root
        self.pending_nodes = [self.root]
        self.rollout_depth = rollout_depth
        self.max_step = max_step
        self.mixmax_weight = mixmax_weight

    def selection(self):
        """
        select a node according to ucb1 value
        """
        choosen = self.pending_nodes[0]
        ucb1 = choosen.get_ucb()
        for node in self.pending_nodes[1:]:
            if node.get_ucb() > ucb1:
                ucb1 = node.get_ucb()
                choosen = node
        return choosen

    def expansion(self, node):
        """
        expand a child from the selected node
        """
        end, child = self.interface.get_child(node, True)
        if not child: self.pending_nodes.remove(node)
        else:
            node.childs.append(child)
            if not end: self.pending_nodes.append(child)
        return end, child

    def rollouts(self, node):
        """
        simulation rollout starting from node
        """
        mean_reward, max_reward = simulation.run(node, self.rollout_depth)
        return (1-self.mixmax_weight)*mean_reward + self.mixmax_weight*max_reward

    def backup(self, mean_reward, node):
        """
        backup mean_reward computed in simulation
        into node and node's parents
        """
        node.reward += mean_reward
        node.visits += 1
        parent = node.parent
        while parent:
            parent.reward += mean_reward
            parent.visits += 1
            parent = parent.parent

    def buildTree(self):
        """
        build the tree from Monte Carlo rollouts
        """
        global_step = 0
        while global_step < self.max_step and self.pending_nodes:
            selected = self.selection()
            end, child = self.expansion(selected)
            if child and not end:
                global_step += 1
                mean_reward = self.rollouts(child)
                self.backup(mean_reward, child)
        print(global_step)

    def buildTree_Wrapper(self, _):
        """
        wrapper of buildTree method for multiprocessing
        """
        self.buildTree()
        return self.root

    def merge(self, nodeA, nodeB):
        """
        merge two trees starting from nodeA and nodeB
        merged tree starts from nodeA
        """
        nodeA.visits += nodeB.visits
        nodeA.reward += nodeB.reward
        same = False
        for childB in nodeB.childs:
            for childA in nodeA.childs:
                if childB.action == childA.action :
                    same = True
                    self.merge(childA, childB)
                    break
            if not same:
                childB.parent = nodeA
                nodeA.childs.append(childB)

    def bestPath(self, node):
        """
        find the best path starting from node
        """
        path = [node]
        while node.childs:
            best_value = sys.float_info.min
            best_child = node.childs[0]
            for child in node.childs[1:]:
                if child.value() > best_child.value(): best_child = child
            path.append(best_child)
            node = best_child
        return path

if __name__ == '__main__':

    class Simulation(Interface):
        """
        An example of simulation class with
        get_child and run method implemented
        """

        def __init__(self, env_fn):
            """
            initialize simultion with environment instance
            """
            self.env = env_fn().unwrapped
            self.init_obs = self.env.reset()
            self.actions = list(range(self.env.action_space.n))

        def get_child(self, node):
            """
            an implementation of method get_child from template: interface
            """
            unavailable_act = []
            for child in node.childs:
                unavailable_act.append(child.action)
            available_act = [a for a in self.actions if a not in unavailable_act]
            if not available_act: return False, None
            action = np.random.choice(available_act)
            self.env.restore_full_state(node.state)
            reward = 0
            for _ in range(4):
                _, r, end, _ = self.env.step(action)
                reward += r
            return  end, Node(action, reward, node, state = self.env.clone_full_state())

        def run(self, node, simu_depth):
            """
            an implementation of method run from template: interface
            """
            self.env.restore_full_state(node.state)
            count, reward, max_r = 0, 0, 0
            for _ in range(simu_depth):
                action = np.random.choice(self.actions)
                count += 1
                for _ in range(4):
                    _, r, end, _ = self.env.step(action)
                    reward += r
                    if r > max_r: max_r = r
                if end : break
            return reward/max(count, 1), max_r

        def play(self, path):
            """
            a method to display path computed by mcts
            """
            self.env.restore_full_state(path[0].state)
            end = False
            for node in path[1:]:
                for _ in range(4):
                    self.env.render()
                    _, _, end, _ = self.env.step(node.action)
            if end: self.env.reset()
            return end

    from multiprocessing import Pool
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Assault-ram-v0')
    args = parser.parse_args()
    simulation = Simulation(lambda : gym.make(args.env))
    root = Node(0, 0, None, state = simulation.env.clone_full_state())
    processes_nb = 4
    for _ in range(1000):
        mcts = Mcts(simulation, root)
        with Pool(processes=processes_nb) as pool:
            out = pool.map(mcts.buildTree_Wrapper, range(processes_nb))
        for i in range(1,processes_nb)
            mcts.merge(out[0], out[i])
        path = mcts.bestPath(out[0])
        print(path)
        end = simulation.play(path)
        if end: root = Node(0, 0, None, state = simulation.env.clone_full_state())
        else:
            index = min(7,len(path)-1)
            root = Node(0, 0, None, state = path[index].state)
