class Matrix(object):

    def __init__(self, x):
        self.element = x
        self.row = len(x)
        self.col = len(x[0])

    def dot(self, y):
        e = y.element
        y_r = y.row
        y_c = y.col

        assert self.row == y_r

        res = [[0] * y_c for _ in range(self.row)]

        for i in range(self.row):
            for j in range(y_c):
                v = 0
                for k in range(self.col):
                    v += self.element[i][k] * e[k][j]
                res[i][j] = v
        return Matrix(res)

    def __str__(self):
        res = ''
        for i in self.element:
            res += (str(i))
            res += '\n'
        return res

    def __repr__(self):
        return "This is a Matrix class"


x = Matrix([
    [1, 2],
    [3, 4],
])

y = Matrix([[5], [6]])

print(repr(x.dot(y)))
