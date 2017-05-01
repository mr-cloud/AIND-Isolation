"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import math


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # TODO: finish this function!
    if game.is_winner(player):
        return float('inf')
    if game.is_loser(player):
        return float('-inf')

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # 1st: Evaluated by sigmoid function (1 / 1 + e^-z, z = (#my_moves - #opponents_moves))
    # return 1 / (1 + math.exp(-float(own_moves - opp_moves)))

    # 2nd: Evaluated by (#my_moves - 2 * #opponents_moves)
    return float(own_moves - 2 * opp_moves)

    # 3rd: Evaluated by  (#my_moves - #opponents_moves)^3
    # return float((own_moves - opp_moves) ** 3)

    # 4th: Evaluated by w * (#my_moves - #opponents_moves),
    # w is the weight and is positively proportional to the number of moves we have taken up to now.
    # return float(own_moves - opp_moves) * game.move_count/(game.width * game.height)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)  This parameter should be ignored when iterative = True.

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).  When True, search_depth should
        be ignored and no limit to search depth.

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            DEPRECATED -- This argument will be removed in the next release

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        if not legal_moves:
            return (-1, -1)
        if game.move_count <= 1:
            return legal_moves[random.randint(0, len(legal_moves)-1)]

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            score = None
            move = None
            if self.method == 'minimax':
                if self.iterative:
                    for depth in range(game.width * game.height):
                        score, move = self.minimax(game, depth)
                else:
                    score, move = self.minimax(game, self.search_depth)
            elif self.method == 'alphabeta':
                if self.iterative:
                    for depth in range(game.width * game.height):
                        score, move = self.alphabeta(game, depth)
                else:
                    score, move = self.alphabeta(game, self.search_depth)
            else:
                move = legal_moves[random.randint(0, len(legal_moves) - 1)]

        except Timeout:
            # Handle any actions required at timeout, if necessary
            # Sample a move from the legal moves.
            # move = legal_moves[random.randint(0, len(legal_moves)-1)]
            pass
        # Return the best move from the last completed search iteration

        return move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
            Note: count from current state.

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: finish this function!
        if maximizing_player:
            if depth <= 0 or game.is_winner(self) or game.is_loser(self):
                return self.score(game, self), None
            else:
                return max([(self.minimax(game.forecast_move(move), depth-1, False)[0], move) for move in game.get_legal_moves(self)])
        else:
            if depth <= 0 or game.is_winner(self) or game.is_loser(self):
                return self.score(game, self), None
            else:
                return min([(self.minimax(game.forecast_move(move), depth-1, True)[0], move) for move in game.get_legal_moves(game.get_opponent(self))])


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: finish this function!
        if maximizing_player:
            if depth <= 0 or game.is_winner(self) or game.is_loser(self):
                return self.score(game, self), None
            else:
                val = float('-inf')
                best_move = None
                for move in game.get_legal_moves(self):
                    next_val, _ = self.alphabeta(game.forecast_move(move), depth - 1, alpha, beta, False)
                    if next_val > val:
                        val = next_val
                        best_move = move
                    if val >= beta:
                        return val, best_move
                    alpha = max(alpha, val)
                return val, best_move
        else:
            if depth <= 0 or game.is_winner(self) or game.is_loser(self):
                return self.score(game, self), None
            else:
                val = float('inf')
                best_move = None
                for move in game.get_legal_moves(game.get_opponent(self)):
                    next_val, _ = self.alphabeta(game.forecast_move(move), depth-1, alpha, beta, True)
                    if next_val < val:
                        val = next_val
                        best_move = move
                    if val <= alpha:
                        return val, None
                    beta = min(beta, val)
                return val, best_move

