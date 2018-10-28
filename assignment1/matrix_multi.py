class Matrix():
    def __init__(self):
        self.x = None
        self.y = None

    def have_multi(self):
        if self.x == [] or self.y == [] or len(self.x[0]) != len(self.y):
            raise ValueError("The two matrixs are not aligned.")

    def multi(self, x, y):
        self.x, self.y = x, y
        self.have_multi()
        cx = len(self.x[0])

        res = [[0] * len(self.y[0]) for _ in range(len(self.x))]

        for i in range(len(self.x)):
            for j in range(len(self.y[0])):
                v = 0
                for k in range(cx):
                    v += self.x[i][k] * self.y[k][j]
                res[i][j] = v
        return res


x = [
    [1, 2],
    [3, 4],
]

y = [[5], [6]]

print(Matrix().multi(x, y))

x = [
    [1, 2],
    [3, 4],
]

y = [[5, 6]]

print(Matrix().multi(x, y))

x = []

y = [[5, 6]]

print(Matrix().multi(x, y))
