# Importing required libraries and modules
import random
import math
from helper_functions import *

# enum for grid cell contents
EMPTY = 0
MIN = 1
MAX = 2

class MCTSNode:
    def __init__(self, state, parent=None):
        """
        Initializing an MCTS Node
        :param state: Current state of the game board, includes all information about current position of game
        :param parent: The parent node of this node in MCTS tree. Secting NONE for root node
        """
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
        """
        Initializing MCTS board
        :param board_size: Defines game size i.e. dimensions of the board
        :param win_size: Defines winning size i.e. number of consecutive pieces required to win the game
        This method sets up fundamental parameters for MCTS algorithm
        """
        self.board_size = board_size
        self.win_size = win_size

    def search(self, root_state, num_iterations):
        """
        Performing MCTS starting from root state for specified iterations
        :param root_state: Initial state from where search begins
        :param num_iterations: The number of iterations fro MCTS process. Each one includes selection, simulation and backpropagation phases
        :return: The best move to be taken as determined by MCTS
        """
        row, col = is_winning_move(root_state.board)
        if row != None and col != None:
            return (row, col)
        row, col = find_pattern(root_state.board)
        if row != None and col != None:
            return (row, col)

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
        # print("MCTS called")
        return best_action

    def select(self, node):
        """
        Selecting a node for expansion in MCTS
        :param node: The current node from which selection process begins
        :return: Returns either a new node if current node is expanded already and a child has been selected
        This method iteratively selects best child node based on a balance of exploration and exploitation
        """
        while not node.is_terminal:
            if node.is_fully_expanded:
                action, node = self.get_best_action(node, 2)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        """
        Expand will be called if the current state is not fully expanded
        :param node: State from the select method
        :return: new node which was not explored before
        """
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
        """
        Simulate the game till a game over state is reached
        :param state: State from the search method
        :return: Score: +1 if Max wins, -1 if Min wins, 0 if draw
        """
        current_state = state.copy()  # Make a copy of the current state to avoid changing the parent state
        while not current_state.is_game_over():
            valid_actions = current_state.valid_actions()
            if not valid_actions:
                break
            # Randomly choose an action from the available actions
            action = random.choice(valid_actions)
            current_state = current_state.perform(action)  # Perform the action

        return 1 if state.current_player() == MAX else -1 if state.current_player() == MIN else 0

    def backpropagate(self, node, reward):
        """
        Back propogate the scores and updates visit counts on the way
        :param node: Node from the search method
        :param reward: Calculate the reward by traversing the tree from child to root_node and updating the score
        :return:
        """
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node = node.parent

    def evaluate_move(self, move):
        """
        :param move: Action (row, col) from the get_best_action method
        :return: Score: Higher if close to distance
        """
        # Calculate the distance from the move to the center of the board
        center_x = self.board_size // 2
        center_y = self.board_size // 2
        distance = abs(move[0] - center_x) + abs(move[1] - center_y)

        # Assign a higher score to moves closer to the center
        return 1 / (distance + 1)  # Adding 1 to avoid division by zero

    def get_best_action(self, node, exploration_constant):
        """
        :param node: Node from the selection method
        :param exploration_constant: Constant to adjust exploration in the UCT Formula
        :return: Best move in the form of (row, col) and child_node
        """
        best_score = float('-inf')
        best_moves = []

        current_player = 1 if node.state.current_player() == MAX else -1
        log_visit_count = math.log(node.visit_count)

        for action, child_node in node.children.items():
            if child_node.visit_count == 0:
                move_score = float('inf')  # Considers unexplored nodes
            else:
                exploitation = child_node.total_reward / child_node.visit_count
                exploration = exploration_constant * math.sqrt(log_visit_count / child_node.visit_count)
                move_score = current_player * exploitation + exploration

            # Multiplying the move_score by the heuristic evaluation
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
        action = obj.search(state, 250)
        return action





