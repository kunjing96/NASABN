from collections import namedtuple

Genotype = namedtuple('Genotype', 'recurrent attention concat')

PRIMITIVES = [
    'none',
    'tanh',
    'relu',
    'sigmoid',
    'identity'
]
ATTENTIONS = [
    'att',
    'multiatt',
]
STEPS = 8
CONCAT = 8

N1 = Genotype(recurrent=[('sigmoid', 0), ('identity', 0), ('relu', 0), ('identity', 0), ('identity', 0)], attention='att', concat=range(1, 6))

