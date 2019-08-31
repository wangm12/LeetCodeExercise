# CandyCrush

"""
This question is about implementing a basic elimination algorithm for Candy Crush.

Given a 2D integer array board representing the grid of candy, different positive integers board[i][j] represent different types of candies. A value of board[i][j] = 0 represents that the cell at position (i, j) is empty. The given board represents the state of the game following the player's move. Now, you need to restore the board to a stable state by crushing candies according to the following rules:

If three or more candies of the same type are adjacent vertically or horizontally, "crush" them all at the same time - these positions become empty.
After crushing all candies simultaneously, if an empty space on the board has candies on top of itself, then these candies will drop until they hit a candy or bottom at the same time. (No new candies will drop outside the top boundary.)
After the above steps, there may exist more candies that can be crushed. If so, you need to repeat the above steps.
If there does not exist more candies that can be crushed (ie. the board is stable), then return the current board.
You need to perform the above rules until the board becomes stable, then return the current board.

Note:

The length of board will be in the range [3, 50].
The length of board[i] will be in the range [3, 50].
Each board[i][j] will initially start as an integer in the range [1, 2000].
"""
 def candyCrush(M):
    while True:
        # 1, Check
        crush = set()
        for i in range(len(M)):
            for j in range(len(M[0])):
                if j > 1 and M[i][j] and M[i][j] == M[i][j - 1] == M[i][j - 2]:
                    crush |= {(i, j), (i, j - 1), (i, j - 2)}
                if i > 1 and M[i][j] and M[i][j] == M[i - 1][j] == M[i - 2][j]:
                    crush |= {(i, j), (i - 1, j), (i - 2, j)}

        # 2, Crush
        if not crush: break
        for i, j in crush: M[i][j] = 0

        # 3, Drop
        for j in range(len(M[0])):
            idx = len(M) - 1
            for i in reversed(range(len(M))):
                if M[i][j]: M[idx][j] = M[i][j]; idx -= 1
            for i in range(idx + 1): M[i][j] = 0
    return M
