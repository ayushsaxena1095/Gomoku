"""
Implement your AI here
Do not change the API signatures for __init__ or __call__
__call__ must return a valid action
"""
import random
import math
# enum for grid cell contents
EMPTY = 0
MIN = 1
MAX = 2

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state

        if self.state.is_game_over():
            self.is_terminal = True
        else:
            self.is_terminal = False

        self.is_fully_expanded = self.is_terminal
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_reward = 0

class MCTS():
    def __init__(self, board_size, win_size):
        self.board_size = board_size
        self.win_size = win_size

    def search(self, root_state, num_iterations):
        self.root = MCTSNode(root_state)

        for iteration in range(num_iterations):
            # Selection phase
            node = self.select(self.root)

            # Simulation phase
            reward = self.simulate(node.state)

            # Backpropagation phase
            self.backpropagate(node, reward)

        # Return the best action from the root node after the search
        best_action, best_child_node = max(self.root.children.items(), key=lambda child: child[1].visit_count)
        return best_action

    def select(self, node):
        while not node.is_terminal:
            if node.is_fully_expanded:
                action, node = self.get_best_move(node, 2)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        actions = node.state.valid_actions()
        for action in actions:
            if action not in node.children:
                new_state = node.state.perform(action)
                new_node = MCTSNode(new_state, node)
                node.children[action] = new_node

                if len(actions) == len(node.children):
                    node.is_fully_expanded = True

                return new_node

    def simulate(self, state):
        current_state = state.copy()  # Make a copy of the current state
        while not current_state.is_game_over():
            valid_actions = current_state.valid_actions()
            if not valid_actions:  # No valid actions available
                break
            # Randomly choose an action from the available actions
            action = random.choice(valid_actions)
            current_state = current_state.perform(action)  # Perform the action

        # Calculate the reward based on the terminal state
        # reward = self.calculate_reward(current_state)
        return 1 if state.current_player() == MAX else -1 if state.current_player() == MIN else 0

    def backpropagate(self, node, reward):
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node = node.parent

    def evaluate_move(self, move):
        # Calculate the distance from the move to the center of the board
        center_x = self.board_size // 2
        center_y = self.board_size // 2
        distance = abs(move[0] - center_x) + abs(move[1] - center_y)

        # Assign a higher score to moves closer to the center
        return 1 / (distance + 1)  # Adding 1 to avoid division by zero

    def get_best_move(self, node, exploration_constant):
        best_score = float('-inf')
        best_moves = []

        current_player = 1 if node.state.current_player() == MAX else -1
        log_visit_count = math.log(node.visit_count)

        for action, child_node in node.children.items():
            if child_node.visit_count == 0:
                move_score = float('inf')  # Consider unexplored nodes
            else:
                exploitation = child_node.total_reward / child_node.visit_count
                exploration = exploration_constant * math.sqrt(log_visit_count / child_node.visit_count)
                move_score = current_player * exploitation + exploration

            # Multiply the move_score by the heuristic evaluation
            move_score *= self.evaluate_move(action)

            if move_score > best_score:
                best_score = move_score
                best_moves = [(action, child_node)]
            elif move_score == best_score:
                best_moves.append((action, child_node))

        return random.choice(best_moves)

class Submission:
    def __init__(self, board_size, win_size):
        ### Add any additional initiation code here
        self.board_size = board_size
        self.win_size = win_size

    def __call__(self, state):
        obj = MCTS(self.board_size, self.win_size)
        action = obj.search(state, 500)
        return action



