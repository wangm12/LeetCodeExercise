# LeetCode 48 Rotate Image


# /*
#  * clockwise rotate
#  * first reverse up to down, then swap the symmetry
#  * 1 2 3     7 8 9     7 4 1
#  * 4 5 6  => 4 5 6  => 8 5 2
#  * 7 8 9     1 2 3     9 6 3
# */
def Clockwise(matrix):
    # check edge case
    if not matrix or not matrix[0]:
        return

    matrix.reverse()
    for i in range (len(matrix)):
        for j in range (i+1, len(matrix[0])):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    return matrix

# /*
#  * anticlockwise rotate
#  * first reverse left to right, then swap the symmetry
#  * 1 2 3     3 2 1     3 6 9
#  * 4 5 6  => 6 5 4  => 2 5 8
#  * 7 8 9     9 8 7     1 4 7
# */
def AntiClockwise(matrix):
    # check edge case
    if not matrix or not matrix[0]:
        return
    for r in matrix:
        r.reverse()
    for i in range (len(matrix)):
        for j in range (i+1, len(matrix[0])):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    return matrix

test = [[1,2,3],[4,5,6],[7,8,9]]
test2 = [[1,2,3],[4,5,6],[7,8,9]]
print Clockwise(test)
print AntiClockwise(test2)
