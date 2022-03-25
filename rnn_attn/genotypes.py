from collections import namedtuple

Genotype = namedtuple('Genotype', 'recurrent concat')

PRIMITIVES = [
    'none',
    'tanh',
    'relu',
    'sigmoid',
    'identity'
]
STEPS = 5
CONCAT = 5

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
Attn_V1 = Genotype(recurrent=[('identity', 0), ('identity', 1), ('identity', 1), ('relu', 1), ('identity', 2), ('relu', 2), ('identity', 2), ('identity', 1)], concat=range(1, 9))
Attn_V2 = Genotype(recurrent=[('tanh', 0), ('identity', 0), ('identity', 1), ('identity', 0), ('identity', 0), ('identity', 0), ('tanh', 4), ('identity', 6)], concat=range(1, 9))
Attn_V3 = Genotype(recurrent=[('identity', 0), ('relu', 0), ('identity', 2), ('identity', 3), ('identity', 4), ('identity', 4), ('identity', 4), ('identity', 4)], concat=range(1, 9))
Attn_V4 = Genotype(recurrent=[('sigmoid', 0), ('relu', 1), ('sigmoid', 0), ('sigmoid', 0), ('sigmoid', 0), ('sigmoid', 5), ('relu', 2), ('sigmoid', 0)], concat=range(1, 9))
Attn_V5 = Genotype(recurrent=[('relu', 0), ('relu', 1), ('identity', 1), ('identity', 3), ('relu', 4), ('relu', 4), ('relu', 3), ('relu', 0)], concat=range(1, 9))
Attn_V6 = Genotype(recurrent=[('relu', 0), ('tanh', 1), ('sigmoid', 1), ('sigmoid', 1), ('identity', 0), ('relu', 2), ('tanh', 5), ('sigmoid', 3)], concat=range(1, 9))
Attn_V7 = Genotype(recurrent=[('tanh', 0), ('identity', 0), ('relu', 0), ('identity', 0), ('tanh', 1), ('identity', 0), ('relu', 0), ('relu', 0)], concat=range(1, 9))
Attn_V8 = Genotype(recurrent=[('identity', 0), ('tanh', 0), ('identity', 1), ('identity', 1), ('identity', 1), ('relu', 1), ('identity', 4), ('identity', 4)], concat=range(1, 9))
Attn_V9 = Genotype(recurrent=[('identity', 0), ('relu', 1), ('identity', 2), ('identity', 1), ('identity', 4), ('identity', 5), ('identity', 4), ('identity', 3)], concat=range(1, 9))
Attn_V10 = Genotype(recurrent=[('identity', 0), ('relu', 0), ('relu', 0), ('identity', 1), ('identity', 4), ('identity', 5), ('identity', 4), ('identity', 3)], concat=range(1, 9))
Attn_V11 = Genotype(recurrent=[('identity', 0), ('identity', 0), ('identity', 1), ('identity', 1), ('relu', 3), ('relu', 3), ('relu', 4), ('relu', 6)], concat=range(1, 9))
Attn_V12 = Genotype(recurrent=[('identity', 0), ('identity', 0), ('relu', 2), ('relu', 1), ('identity', 2), ('relu', 2), ('identity', 1), ('sigmoid', 2)], concat=range(1, 9))
Attn_V13 = Genotype(recurrent=[('identity', 0), ('identity', 1), ('identity', 1), ('identity', 0), ('tanh', 2), ('identity', 5), ('relu', 5), ('identity', 6)], concat=range(1, 9))

Attn_N8_3 = Genotype(recurrent=[('identity', 0), ('relu', 1), ('relu', 1), ('identity', 0), ('tanh', 2), ('tanh', 3), ('relu', 5), ('identity', 6)], concat=range(1, 9))
Attn_N8_2 = Genotype(recurrent=[('sigmoid', 0), ('sigmoid', 0), ('tanh', 0), ('tanh', 3), ('relu', 3), ('relu', 0), ('identity', 4), ('relu', 2)], concat=range(1, 9))

Attn_N5 = Genotype(recurrent=[('sigmoid', 0), ('identity', 1), ('relu', 2), ('identity', 1), ('identity', 3)], concat=range(1, 6))
Attn_N4 = Genotype(recurrent=[('relu', 0), ('relu', 0), ('tanh', 2), ('tanh', 3)], concat=range(1, 5))
Attn_darts_like = Genotype(recurrent=[('identity', 0), ('relu', 1), ('relu', 1), ('identity', 0), ('tanh', 2), ('tanh', 3), ('relu', 5), ('identity', 6)], concat=range(1, 9))

DARTS = DARTS_V2

N5_run1_0 = Genotype(recurrent=[('identity', 0), ('identity', 1), ('identity', 0), ('identity', 1), ('identity', 2)], concat=range(1, 6))
N5_run1_1 = Genotype(recurrent=[('identity', 0), ('identity', 1), ('relu', 2), ('identity', 0), ('identity', 3)], concat=range(1, 6))
N5_run1_2 = Genotype(recurrent=[('identity', 0), ('identity', 1), ('relu', 2), ('identity', 1), ('identity', 3)], concat=range(1, 6))
N5_run1_3 = Genotype(recurrent=[('identity', 0), ('identity', 1), ('relu', 2), ('identity', 1), ('identity', 3)], concat=range(1, 6))
N5_run1_4 = Genotype(recurrent=[('identity', 0), ('identity', 1), ('relu', 2), ('identity', 2), ('identity', 3)], concat=range(1, 6))
N5_run1_5 = Genotype(recurrent=[('sigmoid', 0), ('identity', 1), ('relu', 2), ('identity', 1), ('identity', 3)], concat=range(1, 6))

N5_run2_0 = Genotype(recurrent=[('identity', 0), ('tanh', 0), ('identity', 1), ('tanh', 0), ('sigmoid', 2)], concat=range(1, 6))
N5_run2_1 = Genotype(recurrent=[('relu', 0), ('tanh', 1), ('identity', 1), ('sigmoid', 2), ('relu', 1)], concat=range(1, 6))
N5_run2_2 = Genotype(recurrent=[('relu', 0), ('relu', 1), ('identity', 1), ('identity', 1), ('identity', 1)], concat=range(1, 6))
N5_run2_3 = Genotype(recurrent=[('relu', 0), ('relu', 1), ('identity', 1), ('identity', 1), ('identity', 1)], concat=range(1, 6))
N5_run2_4 = Genotype(recurrent=[('relu', 0), ('relu', 1), ('identity', 1), ('identity', 1), ('identity', 1)], concat=range(1, 6))
N5_run2_5 = Genotype(recurrent=[('relu', 0), ('relu', 1), ('identity', 1), ('identity', 1), ('identity', 1)], concat=range(1, 6))

N5_run3_0 = Genotype(recurrent=[('tanh', 0), ('tanh', 1), ('identity', 0), ('identity', 0), ('relu', 0)], concat=range(1, 6))
N5_run3_1 = Genotype(recurrent=[('identity', 0), ('identity', 0), ('identity', 2), ('identity', 2), ('tanh', 2)], concat=range(1, 6))
N5_run3_2 = Genotype(recurrent=[('identity', 0), ('identity', 0), ('identity', 2), ('identity', 2), ('identity', 1)], concat=range(1, 6))
N5_run3_3 = Genotype(recurrent=[('identity', 0), ('identity', 0), ('relu', 2), ('identity', 1), ('identity', 1)], concat=range(1, 6))
N5_run3_4 = Genotype(recurrent=[('tanh', 0), ('identity', 0), ('tanh', 0), ('identity', 1), ('sigmoid', 3)], concat=range(1, 6))
N5_run3_5 = Genotype(recurrent=[('tanh', 0), ('tanh', 1), ('tanh', 0), ('identity', 1), ('sigmoid', 3)], concat=range(1, 6))

N5_random = Genotype(recurrent=[('sigmoid', 0), ('tanh', 1), ('relu', 2), ('sigmoid', 2), ('identity', 2)], concat=range(1, 6))

####################wiki####################
N5_wiki = Genotype(recurrent=[('identity', 0), ('identity', 1), ('relu', 2), ('identity', 2), ('relu', 2)], concat=range(1, 6))
N5_wiki_1 = Genotype(recurrent=[('relu', 0), ('identity', 1), ('identity', 2), ('identity', 2), ('identity', 3)], concat=range(1, 6))
N5_wiki_2 = Genotype(recurrent=[('relu', 0), ('relu', 1), ('identity', 2), ('identity', 2), ('identity', 3)], concat=range(1, 6))
N5_wiki_3 = Genotype(recurrent=[('sigmoid', 0), ('tanh', 1), ('tanh', 2), ('tanh', 3), ('identity', 0)], concat=range(1, 6))
N5_wiki_4 = Genotype(recurrent=[('relu', 0), ('tanh', 0), ('relu', 2), ('relu', 2), ('identity', 3)], concat=range(1, 6))
N5_wiki_5 = Genotype(recurrent=[('relu', 0), ('relu', 1), ('relu', 2), ('identity', 2), ('identity', 3)], concat=range(1, 6))
N5_wiki_6 = Genotype(recurrent=[('relu', 0), ('relu', 1), ('relu', 2), ('identity', 2), ('sigmoid', 3)], concat=range(1, 6))
N5_wiki_7 = Genotype(recurrent=[('relu', 0), ('relu', 1), ('relu', 2), ('tanh', 3), ('identity', 3)], concat=range(1, 6))

N8_wiki = Genotype(recurrent=[('sigmoid', 0), ('relu', 0), ('identity', 1), ('relu', 3), ('relu', 4), ('sigmoid', 3), ('relu', 3), ('relu', 4)], concat=range(1, 9))
N8_wiki_1 = Genotype(recurrent=[('relu', 0), ('relu', 1), ('identity', 2), ('identity', 3), ('identity', 2), ('identity', 2), ('identity', 2), ('identity', 2)], concat=range(1, 9))
N8_wiki_2 = Genotype(recurrent=[('relu', 0), ('relu', 1), ('identity', 1), ('identity', 1), ('identity', 3), ('identity', 2), ('identity', 2), ('identity', 1)], concat=range(1, 9))

N5_mannual_1 = Genotype(recurrent=[('tanh', 0), ('tanh', 0), ('relu', 2), ('identity', 1), ('relu', 1)], concat=range(1, 6))
N5_mannual_2 = Genotype(recurrent=[('sigmoid', 0), ('relu', 0), ('relu', 0), ('relu', 0), ('relu', 2)], concat=range(1, 6))
N5_mannual_3 = Genotype(recurrent=[('relu', 0), ('tanh', 0), ('relu', 2), ('identity', 1), ('identity', 3)], concat=range(1, 6))
N5_mannual_4 = Genotype(recurrent=[('identity', 0), ('tanh', 0), ('relu', 2), ('tanh', 0), ('relu', 1)], concat=range(1, 6))
