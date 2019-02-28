#!/usr/bin/env python
# coding: utf-8
import requests
import numpy as np
import random
import sqlite3
import time
# Variables

database = None
coefs_db = None

alphabet = "10"
lb = 0
length = 100
sos = 10
lb_1 = 1.01
length_1 = 50

# Database Configuration

def _download_db():
    print("Downloading database...")
    db_content = requests.get("http://dev.mtuci.org/database/").content
    print(f"Database size: {len(db_content)} bytes")
    f = open("gg_shit.db", "wb"); f.write(db_content); f.close()

def _execute(command, params=(), commit=True):
    cursor = database.cursor()
    cursor.execute(command, params)
    data = [el for el in cursor]
    if commit: database.commit()
    return data

def _init_db():
    global database
    database = sqlite3.connect("gg_shit.db")

def _setup_coefs(from_db=True):
    global coefs_db
    if from_db:
        coefs_db = [obj[0] for obj in _execute("SELECT crash FROM history")]
    else:
        ...

# Utility Funcs

def get_bool():
    return random.choice(alphabet)

def to_perc(chrom):
    stepen = len(str(chrom))
    x = lb + int(str(chrom), base=2)*(length/((2**stepen)-1))
    ## где lb нижняя граница и length длинна отрезка, все это работает только если chrom двоичная степень миллиона
    return x

def to_coef(chrom):
    stepen = len(str(chrom))
    x = lb_1 + int(str(chrom), base=2)*(length_1/((2**stepen)-1))
    ## где lb нижняя граница и length длинна отрезка, все это работает только если chrom двоичная степень миллиона
    return x

def to_bool(number):
    new = ''
    while number > 0:
        y = str(number % 2)
        new = y + new
        number = int(number / 2)
    return int(new)

def get_fen_data(gen):
    return [round(to_coef(g),2) for g in gen[1]], [round(to_perc(g)/100,2) for g in gen[0]]

"""
def norm_2(X, scores ,lb=0, hb=20):
    X_std = (X - min(scores)) / (max(scores) - min(scores))
    return (X_std * (hb - lb) + lb)
"""

# Genetic ALgorithms

def create_chromosome(size=22):
    c = []
    for i in range(size):
        c.append(get_bool())
    return ''.join(c)

def create_session(size=10):
    proc=[]
    coef=[]
    session = []
    for i in range(size):
        chrome = create_chromosome(22)
        proc.append(chrome)
        coef.append(chrome)
    session.append(proc)
    session.append(coef)
    return session

def create_population(size=20):
    population=[]
    for i in range(size):
        chrome = create_session(sos)
        population.append(chrome)
    return population

def get_score(genotype, the_bank, coefs, proc = 0.4):
    last_i = 0
    counter = 0
    wins = 0
    pred_fen = [round(to_coef(gen),2) for gen in genotype[1]]
    bank_perc = [round(to_perc(gen)/100,2) for gen in genotype[0]]
    for i in range(sos,len(coefs)-sos-1,sos):
        counter += 1
        bank = the_bank
        results = coefs[last_i:i]
        last_i = i
        for n in range(len(pred_fen)):
            our_coef = pred_fen[n]
            real_coef = results[n]
            our_perc = bank_perc[n]
            if our_coef <= real_coef:
                bank += (our_coef*our_perc*bank - our_perc*bank)
            else:
                bank -= our_perc*bank
        if bank - the_bank >= proc*the_bank:
            wins += 1

    return wins

def selection(population, coefs, the_bank):

    mating_pool = []
    scores = []
    for i in range(len(population)):
        genotype = population[i]
        scores.append(get_score(genotype, the_bank, coefs))
    scores = list(map(int, scores))

    reg_coef = np.mean(scores)

    for ind in range(len(scores)):
        score = scores[ind]
        if (score > 17):
            ammount = population[ind]*int((score*score*score*10)/reg_coef)
            mating_pool.extend(ammount)
        if (score > 15):
            ammount = population[ind]*int((score*score*score)/reg_coef)
            mating_pool.extend(ammount)
        elif (score > 10):
            ammount = population[ind]*int((score*score)/reg_coef)
            mating_pool.extend(ammount)
        elif (score > 0):
            ammount = population[ind]*int((score)/reg_coef)
            mating_pool.extend(ammount)
        if len(mating_pool)<1000:
            ammount = population[ind]
            mating_pool.extend(ammount)
    for i in range(int(len(mating_pool)/10)):
        mating_pool.append(create_session(sos))
    print(f'size of mating_pool: {len(mating_pool)}')
    return mating_pool

#population#
# [n] _ n генотипа : [0] _ проц. от банка(при 0) коэф. ставки(при 1): [n] _ номер хромосомы

def crossover(genotype_1, genotype_2): ## на вход принимает генотипы ёпты
    kid = create_session(sos)
    for purpose in range(len(genotype_1)):
        for order in range(len(genotype_1[purpose])):
            cut_point = random.randint(0, len(genotype_2[0][0]))
            half_parent1 = list(genotype_1[purpose][order])[:cut_point]
            half_parent2 = list(genotype_2[purpose][order])[cut_point:]
            child = "".join(half_parent1 + half_parent2)
            kid[purpose][order] = child
    return kid

""" Not used
def crossover_1(genotype_1, genotype_2):
    kid = create_session(10)
    for purpose in range(len(genotype_1)):
        cut_point = random.randint(0,len(genotype_2[0][0])) #### нужно
        half_parent1 = list(genotype_1[purpose])[:cut_point]
        half_parent2 = list(genotype_2[purpose])[cut_point:]
        #  * child = half_parent1 + half_parent2
        child = half_parent1 + half_parent2
        kid[purpose] = child
    return kid
"""

def mutation(genome, gamma = 0.95):
    for purpose in range(len(genome)):
        for order in range(len(genome[purpose])):
            if random.random()>gamma:
                genome[purpose][order] = create_chromosome()
    return genome

def improve_population(population, coefs, bank, top = 0.2, rand = 0.2):
    pop_length = len(population)
    new_pop = []
    mating_pool = selection(population, coefs, bank)

    scores_dict = {}
    for chrom in range(len(population)):
        i_score = get_score(population[chrom], bank, coefs)
        scores_dict[chrom] = i_score
    scores_chrom = sorted(scores_dict, key=scores_dict.get, reverse=True)
    coef_4_best = int(len(scores_chrom)*top)
    new_chromes = scores_chrom[:coef_4_best]
    new_pop = [population[i] for  i in new_chromes]

    #for i in range(int(len(population)*rand)):
    #    new_pop.append(create_session(sos))

    while len(new_pop) != len(population):
        coef_1 = random.randint(0,len(population)-1)
        coef_2 = coef_1
        while coef_1 == coef_2:
            coef_2 = random.randint(0,len(population)-1)
        kid = crossover(population[coef_1],population[coef_2])

        new_pop.append(mutation(kid))
    return new_pop

def natsel(population, all_coefs, bank, gen_num):
    global test
    ## 1.Промотка коэффициентов 20
    ## 2.Эволюция популяции
    ## 3.Оценка популяции ? Возможно не пригодится
    ## 4.Нахождение наилучшего score в текущий популяции
    ## 5.Вывести номер популяции, и стратегию лучшего индивида
    best_population = []
    for n in range(0, gen_num):
        t = time.time()
        # 1.
        start = 200*n
        finish = 200*(n+1)+1
        coefs = all_coefs[start:finish]
        # 2.
        population = improve_population(population, coefs, bank)
        # 4.
        scores = []
        for i in range(len(population)):
            genotype = population[i]
            scores.append(get_score(genotype, bank, coefs))
        best = population[scores.index(max(scores))]
        best_population.append(best)
        # 5

        print(f'gen - {n}, score: {max(scores)}, mean: {np.mean(scores)}, time: {time.time() - t}')
    return best_population

if __name__ == "__main__":
    import pickle
    # _download_db()
    _init_db()
    _setup_coefs()

    pop_amt = 10000
    gen_amt = 7000
    bank = 100
    population = create_population(pop_amt)
    gg = natsel(population, coefs_db, bank, gen_amt)

    # Checking results

    genotype = gg[-1]
    d = get_fen_data(genotype)
    print(d)
    pickle.dump(gg, open("genotype.ga", "wb"))
