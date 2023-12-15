"""
Implement your AI here
Do not change the API signatures for __init__ or __call__
__call__ must return a valid action
"""
import random
import math
import time as t

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_reward = 0

    # def select(self):
        # Implement UCB (Upper Confidence Bound) selection strategy
        # def ucb(child):
        #     start = t.time()
        #     if child[1].visit_count == 0:
        #         return float('inf')  # Consider unvisited nodes as having infinite value
        #     b= child[1].total_reward / child[1].visit_count + math.sqrt( 2 * math.log(self.visit_count) / child[1].visit_count)
        #     end = t.time()
        #     print("ucb time : ", end - start)
        #     return b
    def select(self):
        def ucb_tuned(child):
            if child.visit_count == 0:
                return float('inf')
            exploration = math.sqrt(2 * math.log(self.visit_count) / child.visit_count)
            exploitation = min(1/4, self.compute_variance(child))
            return child.total_reward / child.visit_count + exploration * exploitation

        return max(self.children.values(), key=ucb_tuned)

    def compute_variance(self, node):
        mean_sq = sum(child.total_reward**2 for child in node.children.values()) / node.visit_count
        mean = (node.total_reward / node.visit_count)**2
        return mean_sq - mean + math.sqrt(2 * math.log(self.visit_count) / node.visit_count)


    def expand(self):
        # start = t.time()
        actions = self.state.valid_actions()
        for action in actions:
            new_state = self.state.perform(action)
            self.children[action] = MCTSNode(new_state, parent=self)
        # end = t.time()
        # print("expand time : ", end - start)

    def simulate(self):
        # start = t.time()
        state = self.state.copy()
        while not state.is_game_over():
            actions = state.valid_actions()
            action = random.choice(actions)
            state = state.perform(action)
        # end = t.time()
        # print("simulate time : ", end - start)
        return state.current_score()

    def backpropagate(self, reward):
        # start = t.time()
        node = self
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node = node.parent
        # end = t.time()
        # print("BackPropogate time : ", end - start)

def mcts_search(root_state, num_iterations):
    root_node = MCTSNode(root_state)

    # start = t.time()
    for _ in range(num_iterations):
        node = root_node

        # Selection phase
        while not node.state.is_game_over() and node.children:
            child_node = node.select()
            node = child_node

        # Expansion phase
        if not node.state.is_game_over():
            node.expand()

        # Simulation phase
        reward = node.simulate()

        # Backpropagation phase
        node.backpropagate(reward)

        # end = t.time()
        # if end - start >15:
        #     break

    # Return the best action from the root node after the search
    best_action, best_child_node = max(root_node.children.items(), key=lambda child: child[1].visit_count)
    return best_action

class Submission:
    def __init__(self, board_size, win_size):
        ### Add any additional initiation code here
        pass

    def __call__(self, state):

        ### Replace with your implementation
        # actions = state.valid_actions()
        # return actions[-1]

        action = mcts_search(state, 2000)
        return action
    
