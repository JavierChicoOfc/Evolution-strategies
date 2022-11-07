import requests
import random
import numpy as np
import math

IP = "unknown for security reasons"

website="http://"+ IP +"/age/robot" #c1=3.412&c2=2.4&c3=15.42312&c4=-23.412235"

# EE [mu + lambda]

def get_fitness(individuo):
    if n == 4:
        r = requests.get(website+f"4/?c1={individuo[0]}&c2={individuo[1]}&c3={individuo[2]}&c4={individuo[3]}")
    else: # n == 10
        r = requests.get(website+f"10/?c1={individuo[0]}&c2={individuo[1]}&c3={individuo[2]}&c4={individuo[3]}&c5={individuo[4]}&c6={individuo[5]}&c7={individuo[6]}&c8={individuo[7]}&c9={individuo[8]}&c10={individuo[9]}")
    return (float(r.text))

def eval_poblation(poblation):
    """
    Return the minimal score of a poblation
    """
    minimal=(99999999, None)
    for individuo in poblation:
        score = get_fitness(individuo[0])
        if score < minimal[0]:
            minimal = (score, individuo[0])
    return minimal

# [Settings]

n = 10                # Number of variables
mu_padres = 200       # Nunmber of parents
lambda_hijos = 200    # Number of childs
tournament_size = 2   # Tournament size
family_size = 2       # Family size

# [Initialize poblation]
# Individual = [variables, variances]---->[[[variables],[variances]],[[variables],[variances]]...]
mu_poblation = []
for _ in range(mu_padres):
    funcional = []
    varianzas = []
    for _ in range(n):
        funcional.append(0 + np.random.normal(loc=100, scale=200))
        varianzas.append(random.uniform(10,180))
    mu_poblation.append([funcional, varianzas])

ciclo = 0
while ciclo < 1000:
    score_mu_poblation = []
    chosen_poblation = []
    # [Evaluate poblation]
    for individuo in mu_poblation:
        score_mu_poblation.append(get_fitness(individuo[0]))

    minimal = min(score_mu_poblation)
    print(f"Generation: {ciclo+1}\n Fitness: {minimal}\nIndividuo: {mu_poblation[score_mu_poblation.index(minimal)][0]}")

    # [Tournament selection]
    for _ in range(lambda_hijos*2):
        contestants = []
        contestants_score = []
        for _ in range(tournament_size):
            c = random.randrange(0,mu_padres)
            participant = mu_poblation[c]
            contestants.append(participant)
            contestants_score.append(score_mu_poblation[c])

        chosen_poblation.append(contestants[contestants_score.index(min(contestants_score))])
    # [Crossing over]
    lambda_poblation = []
    counter = 0
    for _ in range(lambda_hijos):
        p1_value = chosen_poblation[counter][0]
        p2_value = chosen_poblation[counter+1][0]
        p1_varianza = chosen_poblation[counter][1]
        p2_varianza = chosen_poblation[counter+1][1]
        # [Create childs]
        new_individuo = []
        funcional = []
        # [Functional part]
        for element, _ in enumerate(range(n)):
            funcional.append((1/family_size)*(p1_value[element] + p2_value[element]))
        new_individuo.append(funcional)
        # [Variances]
        new_varianzas = []
        for counter_var, _ in enumerate(range(n)):
            if random.choice((0,1)) == 0:
                new_varianzas.append(p1_varianza[counter_var])
            else:
                new_varianzas.append(p2_varianza[counter_var]) 
        new_individuo.append(new_varianzas)
        counter +=2
        muted_motors = []
        # [Mutation]
        for contador_posicion, _ in enumerate(range(n)):
            # [Functional part mutation]
            new_individuo[0][contador_posicion] = new_individuo[0][contador_posicion] + np.random.normal(loc=0, scale=new_individuo[1][contador_posicion])
            # [Variances part mutation]
            new_individuo[1][contador_posicion] = new_individuo[1][contador_posicion] * math.exp(np.random.normal(loc=0, scale=(1/np.sqrt(2*np.sqrt(lambda_hijos)))))
        # [Add new individuals]
        lambda_poblation.append(new_individuo)

    # [Add lambda poblation to mu poblation]
    total_poblation = mu_poblation + lambda_poblation
    score_lambda_poblation = []
    for individuo in lambda_poblation:
        score_lambda_poblation.append(get_fitness(individuo[0]))

    total_score_poblation = score_mu_poblation + score_lambda_poblation
    final_poblation = []
    
    for _ in range(mu_padres):
        best = (9999999999, None)
        pos = 0
        for score_individuo in total_score_poblation:
            if score_individuo < best[0]:
                best = (score_individuo, pos)
                
            pos +=1
        total_score_poblation[best[1]] = 9999999999
        final_poblation.append(total_poblation[best[1]])

    
    mu_poblation = final_poblation.copy()
    ciclo +=1

e = eval_poblation(final_poblation)
e_value = e[0]


print(f"Generation {ciclo}: {e}")