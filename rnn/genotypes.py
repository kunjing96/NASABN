from collections import namedtuple

Genotype = namedtuple('Genotype', 'recurrent concat')

PRIMITIVES = [
    'none',
    'tanh',
    'relu',
    'sigmoid',
    'identity'
]
STEPS = 8
CONCAT = 8

ENAS = Genotype(
    recurrent = [
        ('tanh', 0),
        ('tanh', 1),
        ('relu', 1),
        ('tanh', 3),
        ('tanh', 3),
        ('relu', 3),
        ('relu', 4),
        ('relu', 7),
        ('relu', 8),
        ('relu', 8),
        ('relu', 8),
    ],
    concat = [2, 5, 6, 9, 10, 11]
)

DARTS_V1 = Genotype(recurrent=[('relu', 0), ('relu', 1), ('tanh', 2), ('relu', 3), ('relu', 4), ('identity', 1), ('relu', 5), ('relu', 1)], concat=range(1, 9))
DARTS_V2 = Genotype(recurrent=[('sigmoid', 0), ('relu', 1), ('relu', 1), ('identity', 1), ('tanh', 2), ('sigmoid', 5), ('tanh', 3), ('relu', 5)], concat=range(1, 9))

DARTS = DARTS_V2

DARTS_run1_0 = Genotype(recurrent=[('identity', 0), ('relu', 1), ('relu', 1), ('identity', 2), ('identity', 0), ('tanh', 2), ('sigmoid', 3), ('tanh', 7)], concat=range(1, 9))
DARTS_run1_1 = Genotype(recurrent=[('relu', 0), ('relu', 0), ('relu', 1), ('identity', 1), ('relu', 0), ('relu', 3), ('relu', 2), ('relu', 4)], concat=range(1, 9))
DARTS_run1_2 = Genotype(recurrent=[('relu', 0), ('relu', 0), ('tanh', 1), ('tanh', 2), ('relu', 0), ('relu', 2), ('relu', 1), ('relu', 1)], concat=range(1, 9))
DARTS_run1_3 = Genotype(recurrent=[('tanh', 0), ('tanh', 1), ('relu', 0), ('relu', 0), ('relu', 0), ('relu', 3), ('identity', 5), ('identity', 5)], concat=range(1, 9))
DARTS_run1_4 = Genotype(recurrent=[('tanh', 0), ('identity', 0), ('relu', 0), ('relu', 0), ('relu', 0), ('relu', 3), ('identity', 5), ('identity', 5)], concat=range(1, 9))
DARTS_run1_5 = Genotype(recurrent=[('tanh', 0), ('identity', 0), ('relu', 0), ('relu', 0), ('relu', 0), ('relu', 3), ('identity', 5), ('identity', 5)], concat=range(1, 9))

DARTS_run2_0 = Genotype(recurrent=[('identity', 0), ('relu', 1), ('tanh', 0), ('identity', 2), ('identity', 1), ('tanh', 0), ('relu', 2), ('identity', 6)], concat=range(1, 9))
DARTS_run2_1 = Genotype(recurrent=[('relu', 0), ('relu', 0), ('sigmoid', 0), ('relu', 3), ('tanh', 3), ('sigmoid', 3), ('relu', 2), ('relu', 2)], concat=range(1, 9))
DARTS_run2_2 = Genotype(recurrent=[('relu', 0), ('relu', 1), ('sigmoid', 0), ('relu', 2), ('tanh', 3), ('tanh', 0), ('relu', 2), ('identity', 5)], concat=range(1, 9))
DARTS_run2_3 = Genotype(recurrent=[('relu', 0), ('relu', 1), ('sigmoid', 0), ('tanh', 1), ('tanh', 3), ('tanh', 4), ('relu', 2), ('identity', 5)], concat=range(1, 9))
DARTS_run2_4 = Genotype(recurrent=[('relu', 0), ('relu', 1), ('identity', 2), ('identity', 1), ('sigmoid', 3), ('tanh', 4), ('identity', 3), ('identity', 5)], concat=range(1, 9))
DARTS_run2_5 = Genotype(recurrent=[('relu', 0), ('relu', 1), ('sigmoid', 0), ('identity', 1), ('sigmoid', 3), ('sigmoid', 0), ('relu', 2), ('identity', 5)], concat=range(1, 9))

DARTS_run3_0 = Genotype(recurrent=[('sigmoid', 0), ('relu', 0), ('identity', 2), ('sigmoid', 0), ('relu', 3), ('tanh', 5), ('tanh', 2), ('sigmoid', 3)], concat=range(1, 9))
DARTS_run3_1 = Genotype(recurrent=[('identity', 0), ('sigmoid', 1), ('sigmoid', 0), ('tanh', 3), ('sigmoid', 4), ('sigmoid', 4), ('identity', 3), ('sigmoid', 6)], concat=range(1, 9))
DARTS_run3_2 = Genotype(recurrent=[('identity', 0), ('sigmoid', 1), ('identity', 1), ('identity', 1), ('identity', 1), ('identity', 3), ('identity', 0), ('identity', 0)], concat=range(1, 9))
DARTS_run3_3 = Genotype(recurrent=[('identity', 0), ('sigmoid', 1), ('sigmoid', 2), ('relu', 1), ('identity', 0), ('sigmoid', 0), ('identity', 0), ('identity', 0)], concat=range(1, 9))
DARTS_run3_4 = Genotype(recurrent=[('identity', 0), ('sigmoid', 1), ('sigmoid', 2), ('relu', 1), ('identity', 0), ('sigmoid', 0), ('identity', 0), ('identity', 0)], concat=range(1, 9))
DARTS_run3_5 = Genotype(recurrent=[('identity', 0), ('sigmoid', 1), ('sigmoid', 2), ('relu', 1), ('identity', 0), ('sigmoid', 0), ('identity', 0), ('identity', 0)], concat=range(1, 9))


V1 = Genotype(recurrent=[('none', 0), ('tanh', 0), ('identity', 0), ('sigmoid', 3), ('relu', 3), ('none', 3), ('identity', 6), ('none', 5)], concat=range(1, 9))
V2 = Genotype(recurrent=[('relu', 0), ('identity', 0), ('relu', 2), ('identity', 1), ('relu', 2), ('none', 3), ('none', 2), ('identity', 7)], concat=range(1, 9))
V3 = Genotype(recurrent=[('tanh', 0), ('identity', 0), ('identity', 1), ('identity', 0), ('identity', 3), ('identity', 5), ('identity', 6), ('identity', 4)], concat=range(1, 9))