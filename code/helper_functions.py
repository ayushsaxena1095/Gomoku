import numpy as np

def is_winning_move(board):
    """
    :param board: board of the game representing the Empty, Min and Max player's move.
    :return: It will return row, col if a possible match is found in the current board else None, None
    """

    patterns = [
        {0: np.array([0, 1, 1, 1, 1])}, {4: np.array([1, 1, 1, 1, 0])},
        {1: np.array([1, 0, 1, 1, 1])}, {3: np.array([1, 1, 1, 0, 1])},
    ]

    new_board = board.copy()
    new_board[2] = new_board[2] * 2

    # Variable to track all the available wins
    moves = []

    # Horizontal Pattern Matching
    for idx, pattern in enumerate(patterns):
        key, value = next(iter(pattern.items()))
        row, col = helper_horizontal(new_board, value)
        if row != None and col != None and row >= 0 and col >= 0:
            row, col = row, col + key
            if board[0][row][col] == 1:
                moves.append((str(value), row, col))
            else:
                continue

    # Vertical Pattern Matching
    for idx, pattern in enumerate(patterns):
        key, value = next(iter(pattern.items()))
        row, col = helper_vertical(new_board, value)
        if row != None and col != None and row >= 0 and col >= 0:
            row, col = row + key, col
            if board[0][row][col] == 1:
                moves.append((str(value), row, col))
            else:
                continue

    # Diagonal Pattern Matching
    for idx, pattern in enumerate(patterns):
        key, value = next(iter(pattern.items()))
        row, col = helper_diagonal(new_board, value)
        if row != None and col != None and row >= 0 and col >= 0:
            row, col = row + key, col + key
            if board[0][row][col] == 1:
                moves.append((str(value), row, col))
            else:
                continue

    # Anti-Diagonal Matching4
    for idx, pattern in enumerate(patterns):
        key, value = next(iter(pattern.items()))
        row, col = helper_antidiagonal(new_board, value)
        if row != None and col != None and row >= 0 and col >= 0:
            row, col = row + key, col - key
            if board[0][row][col] == 1:
                moves.append((str(value), row, col))
            else:
                continue

    moves = prioritize_patterns(moves)
    if moves:
        x, y = moves[0][1], moves[0][2]
    else:
        x, y = None, None
    return (x, y)



def find_pattern(board):
    patterns = [
        {0: np.array([0, 1, 1])}, {2: np.array([1, 1, 0])}, {1: np.array([1, 0, 1])},
        {2: np.array([0, 1, 0, 1])}, {1: np.array([1, 0, 1, 0])}, {0: np.array([0, 1, 1, 0])},
        {0: np.array([0, 1, 1, 1])}, {3: np.array([1, 1, 1, 0])}, {2: np.array([1, 1, 0, 1])},
        {0: np.array([0, 1, 1, 1, 0])},
        {0: np.array([0, 1, 1, 1, 1])}, {4: np.array([1, 1, 1, 1, 0])}, {2: np.array([1, 1, 0, 1, 1])},
    ]

    # Changing the values in board[1] i.e. min player to 2 in order to find the pattern in max player
    new_board = board.copy()
    new_board[1] = new_board[1] * 2

    # Variable to track all the available blocks
    moves = []

    # Horizontal Pattern Matching
    for idx, pattern in enumerate(patterns):
        key, value = next(iter(pattern.items()))
        row, col = helper_horizontal(new_board, value)
        if row != None and col != None and row >= 0 and col >= 0:
            row, col = row, col + key
            if board[0][row][col] == 1:
                moves.append((str(value), row, col))
            else:
                continue

    # Vertical Pattern Matching
    for idx, pattern in enumerate(patterns):
        key, value = next(iter(pattern.items()))
        row, col = helper_vertical(new_board, value)
        if row != None and col != None and row >= 0 and col >= 0:
            row, col = row + key, col
            if board[0][row][col] == 1:
                moves.append((str(value), row, col))
            else:
                continue

    # Diagonal Pattern Matching
    for idx, pattern in enumerate(patterns):
        key, value = next(iter(pattern.items()))
        row, col = helper_diagonal(new_board, value)
        if row != None and col != None and row >= 0 and col >= 0:
            row, col = row + key, col + key
            if board[0][row][col] == 1:
                moves.append((str(value), row, col))
            else:
                continue

    # Anti-Diagonal Matching4
    for idx, pattern in enumerate(patterns):
        key, value = next(iter(pattern.items()))
        row, col = helper_antidiagonal(new_board, value)
        if row != None and col != None and row >= 0 and col >= 0:
            row, col = row + key, col - key
            if board[0][row][col] == 1:
                moves.append((str(value), row, col))
            else:
                continue

    moves = prioritize_patterns(moves)
    if moves:
        x, y = moves[0][1], moves[0][2]
    else:
        x, y = None, None
    return (x, y)

def helper_horizontal(state, pattern):
    """
    :param board: board of the game representing the Empty, Min and Max player's move.
    :return: row, col if a possible match is found in the current board else None, None
    """
    # Adding the positions of Min and Max player to give accurate representation of the currently occupied positions
    board = state[1] + state[2]
    rows, cols = board.shape
    pattern_length = len(pattern)

    for i in range(rows):
        for j in range(cols - pattern_length + 1):
            if np.array_equal(board[i, j:j + pattern_length], pattern):
                return (i, j)

    return (None, None)

def helper_vertical(state, pattern):
    """
    :param state: The current state of the board given by state.board
    :param pattern: The pattern we are looking for in the format of 1d numpy array
    :return: row, col if a possible match is found in the current board else None, None
    """
    board = state[1] + state[2]
    transposed_board = board.T  # Transpose the board to search vertically
    rows, cols = transposed_board.shape
    pattern_length = len(pattern)

    for i in range(rows - pattern_length + 1):
        for j in range(cols):
            if np.array_equal(board[i:i + pattern_length, j], pattern):
                return (i, j)
    return (None, None)

def helper_diagonal(state, pattern):
    """
    :param state: The current state of the board given by state.board
    :param pattern: The pattern we are looking for in the format of 1d numpy array
    :return: row, col if a possible match is found in the current board else None, None
    """
    board = state[1] + state[2]
    rows, cols = board.shape
    pattern_length = len(pattern)

    # Check diagonals from top-left to bottom-right
    for i in range(rows - pattern_length + 1):
        for j in range(cols - pattern_length + 1):
            diagonal = np.diag(board[i:i + pattern_length, j:j + pattern_length])
            if np.array_equal(diagonal, pattern):
                # print("Diagonal")
                # return ("Diagonal", i, j)
                return (i, j)
    return (None, None)

def helper_antidiagonal(state, pattern):
    """
    :param state: The current state of the board given by state.board
    :param pattern: The pattern we are looking for in the format of 1d numpy array
    :return: row, col if a possible match is found in the current board else None, None
    """
    board = state[1] + state[2]
    rows, cols = board.shape
    pattern_length = len(pattern)

    # Check diagonals from top-right to bottom-left
    for i in range(rows - pattern_length + 1):
        for j in range(cols - 1, pattern_length - 2, -1):
            diagonal = np.diag(np.fliplr(board)[i:i + pattern_length, cols - j - 1:cols - j + pattern_length - 1])
            if np.array_equal(diagonal, pattern):
                # print("Anti-diagonal")
                # return ("Anti-diagonal", i, j)
                return (i, j)

    return (None, None)

def helper_diagonal_1(state, pattern):
    """
    :param state: The current state of the board given by state.board
    :param pattern: The pattern we are looking for in the format of 1d numpy array
    :return: row, col if a possible match is found in the current board else None, None
    """
    board = state[1] + state[2]
    rows, cols = board.shape
    pattern_length = len(pattern)

    # Check diagonals from top-left to bottom-right
    for i in range(rows - pattern_length + 1):
        for j in range(cols - pattern_length + 1):
            diagonal = np.diag(board[i:i + pattern_length, j:j + pattern_length])
            if np.array_equal(diagonal, pattern):
                # print("Diagonal")
                return ("Diagonal", i, j)

    # Check diagonals from top-right to bottom-left
    for i in range(rows - pattern_length + 1):
        for j in range(cols - 1, pattern_length - 2, -1):
            diagonal = np.diag(np.fliplr(board)[i:i + pattern_length, cols - j - 1:cols - j + pattern_length - 1])
            if np.array_equal(diagonal, pattern):
                # print("Anti-diagonal")
                return ("Anti-diagonal", i, j)

    return (None, None, None)

def prioritize_patterns(matches):
    """
    :param state: A tuple containing the pattern in string format followed by the row and col where it is found.
    :return: A sorted tuple that has the pattern with highest priority in the beginning.
    """
    pattern_scores = {
        '[0 1 1]': 1,
        '[1 1 0]': 1,
        '[1 0 1]': 1,
        '[0 1 0 1]': 2,
        '[1 0 1 0]': 2,
        '[0 1 1 0]': 2,
        '[0 1 1 1]': 3,
        '[1 1 1 0]': 3,
        '[1, 1, 0, 1]': 4,
        '[0 1 1 1 0]': 4,
        '[1 1 1 1 0]': 4,
        '[0 1 1 1 1]': 4,
        '[1 1 0 1 1]': 4,
    }

    return sorted(matches, key=lambda x: pattern_scores.get(x[0], 0), reverse=True)


if __name__ == "__main__":
    test_case = 1
    if test_case == 1:
        # Testing the horizontal pattern
        board = np.array([
            [
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
            ],
            [
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
            ],
            [
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 1., 1., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 1., 1., 1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 1., 0., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
            ]
        ])
        board[0] = board[0] - board[2]
        patterns = [
            np.array([0, 1, 1]), np.array([1, 1, 0]),
            np.array([0, 1, 0, 1]), np.array([1, 0, 1, 0]),
            np.array([0, 1, 1, 1]), np.array([1, 1, 1, 0]),
        ]
        # Horizontal
        patterns = [
            {0: np.array([0, 1, 1])}, {2: np.array([1, 1, 0])},
            {2: np.array([0, 1, 0, 1])}, {1: np.array([1, 0, 1, 0])},
            {0: np.array([0, 1, 1, 1])}, {3: np.array([1, 1, 1, 0])},
            {0: np.array([0, 1, 1, 1, 0])},
            {0: np.array([0, 1, 1, 1, 1])}, {4: np.array([1, 1, 1, 1, 0])}
        ]

        board[0] = board[0] - board[2]
        moves = []
        for idx, pattern in enumerate(patterns):
            key, value = next(iter(pattern.items()))
            row, col = helper_horizontal(board, value)
            if row != None and col != None and row >= 0 and col >= 0:
                row, col = row, col+key
                if board[0][row][col] == 1:
                    moves.append((str(value), row, col))
                else:
                    continue

        print(prioritize_patterns(moves))

    elif test_case == 2:
        # Testing the vertical pattern
        board = np.array([
            [
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
            ],
            [
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
            ],
            [
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
                [0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
                [0., 0., 1., 0., 1., 0., 0., 1., 0., 0.],
                [0., 0., 1., 0., 1., 0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 1., 0., 0., 0., 1., 0.],
                [0., 0., 0., 1., 1., 0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 1., 0., 0., 0., 0., 0., 0.]
            ]
        ])
        board[0] = board[0] - board[2]
        # Vertical
        patterns = [
            {0: np.array([0, 1, 1])}, {2: np.array([1, 1, 0])},
            {2: np.array([0, 1, 0, 1])}, {1: np.array([1, 0, 1, 0])},
            {0: np.array([0, 1, 1, 1])}, {3: np.array([1, 1, 1, 0])},
            {0: np.array([0, 1, 1, 1, 0])},
            {0: np.array([0, 1, 1, 1, 1])}, {4: np.array([1, 1, 1, 1, 0])}
        ]

        board[0] = board[0] - board[2]
        moves = []
        for idx, pattern in enumerate(patterns):
            key, value = next(iter(pattern.items()))
            row, col = helper_vertical(board, value)
            if row != None and col != None and row >= 0 and col >= 0:
                row, col = row + key, col
                if board[0][row][col] == 1:
                    moves.append((str(value), row, col))
                else:
                    continue

        print(prioritize_patterns(moves))

    elif test_case == 3:
        # Testing the Diagonal pattern
        board = np.array([
            [
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
            ],
            [
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
            ],
            [
                [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
                [0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
                [0., 1., 0., 0., 1., 0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 1., 0., 0., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
            ]
        ])

        # Diagonal
        patterns = [
            {0: np.array([0, 1, 1])}, {2: np.array([1, 1, 0])},
            {2: np.array([0, 1, 0, 1])}, {1: np.array([1, 0, 1, 0])},
            {0: np.array([0, 1, 1, 1])}, {3: np.array([1, 1, 1, 0])},
            {0: np.array([0, 1, 1, 1, 0])},
            {0: np.array([0, 1, 1, 1, 1])}, {4: np.array([1, 1, 1, 1, 0])}
        ]

        board[0] = board[0] - board[2]
        moves = []
        for idx, pattern in enumerate(patterns):
            key, value = next(iter(pattern.items()))
            row, col = helper_diagonal(board, value)
            if row != None and col != None and row >= 0 and col >= 0:
                row, col = row + key, col+key
                if board[0][row][col] == 1:
                    moves.append((str(value), row, col))
                else:
                    continue

        print(prioritize_patterns(moves))

    elif test_case == 4:
        # Testing the Anti-Diagonal pattern
        board = np.array([
            [
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
            ],
            [
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
            ],
            [
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 1., 0., 0., 0., 1., 0., 0.],
                [0., 0., 1., 0., 0., 0., 1., 0., 0., 0.],
                [0., 1., 0., 0., 0., 1., 0., 0., 0., 0.],
                [1., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
            ]
        ])
        patterns = [
            {0: np.array([0, 1, 1])}, {2: np.array([1, 1, 0])},
            {2: np.array([0, 1, 0, 1])}, {1: np.array([1, 0, 1, 0])},
            {0: np.array([0, 1, 1, 1])}, {3: np.array([1, 1, 1, 0])},
            {0: np.array([0, 1, 1, 1, 0])},
            {0: np.array([0, 1, 1, 1, 1])}, {4: np.array([1, 1, 1, 1, 0])}
        ]

        board[0] = board[0] - board[2]
        moves = []
        for idx, pattern in enumerate(patterns):
            key, value = next(iter(pattern.items()))
            row, col = helper_antidiagonal(board, value)
            if row != None and col != None and row >= 0 and col >= 0:
                row, col = row + key, col - key
                if board[0][row][col] == 1:
                    moves.append((str(value), row, col))
                else:
                    continue

        print(prioritize_patterns(moves))

