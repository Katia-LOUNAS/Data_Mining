#  Crée les candidats initiaux, qui sont les éléments individuels, à partir des données de transaction.
def create_initial_candidates(data):
    candidates = set()
    for transaction in data:
        for item in transaction:
            candidates.add(frozenset([item])) 
    return list(candidates)
# Calcule le support d'un itemset en parcourant les transactions et en comptant le nombre de fois où l'itemset apparaît dans chaque transaction.
def calculate_support(data, itemset):
    count = 0
    for transaction in data:
        if itemset.issubset(transaction):
            count += 1
    return count
# Élaguer les candidats qui ne répondent pas au support minimum.
def prune_candidates(candidates, prev_frequent_items, k,data):
    pruned_candidates = []
    for candidate in candidates:
        support = calculate_support(data, candidate)
        if support >= k:
            pruned_candidates.append(candidate)
    return pruned_candidates
# Génère de nouveaux candidats en combinant les candidats fréquents de la dernière étape.
def generate_candidates(prev_candidates, k):
    candidates = set()
    n = len(prev_candidates)
    for i in range(n):
        for j in range(i + 1, n):
            itemset1 = list(prev_candidates[i])
            itemset2 = list(prev_candidates[j])
            if itemset1[:-1] == itemset2[:-1]:
                new_candidate = frozenset(itemset1 + [itemset2[-1]])
                candidates.add(new_candidate)
    return list(candidates)
# L'algorithme Apriori.
def apriori(data, min_support):
    candidates = create_initial_candidates(data)
    k = 1
    frequent_itemsets = []
    while candidates:
        if frequent_itemsets:
            candidates = prune_candidates(candidates, frequent_itemsets[-1], min_support,data)
        frequent_itemsets.extend(candidates)
        k += 1
        candidates = generate_candidates(candidates, k)
    
    return frequent_itemsets
import itertools
import math
def calculate_lift(confidence, support):
    return confidence / support

def calculate_cosine(support_A, support_B, support_AB ):
    cosine_similarity = support_AB / math.sqrt(support_A * support_B)
    return cosine_similarity



def generate_association_rules(L, min_confidence, min_correlation, transactions, correlation = "confidence"):
    association_rules = []

    def calculate_support(itemset, transactions):
        count = 0
        for transaction in transactions:
            if set(itemset).issubset(set(transaction)):
                count += 1
        return count

    for itemset in L:
        itemset_list = list(itemset)
        for i in range(1, len(itemset_list)):
            for combination in itertools.combinations(itemset_list, i):
                A = set(combination)
                B = itemset - A
                support_A = calculate_support(A, transactions)
                support_B = calculate_support(B, transactions)
                support_AB = calculate_support(itemset, transactions)
                confidence = support_AB / support_A

                if correlation == "confidence":
                    if confidence >= min_confidence:
                        association_rules.append((A, B, confidence, support_AB))
                elif correlation == "lift":
                    lift = calculate_lift(confidence, support_AB)
                    if confidence >= min_confidence and lift >= min_correlation:
                        association_rules.append((A, B, confidence, lift, support_AB))
                elif correlation == "cosine":
                    cosine_similarity = calculate_cosine(support_A, support_B, support_AB)
                    if cosine_similarity >= min_correlation:
                        association_rules.append((A, B, confidence, cosine_similarity, support_AB))

    return association_rules




def regles_d_association(data, min_support, min_confidence, min_correlation, correlation):
    frequent_itemsets = apriori(data, min_support)
    nombre_itemsets = len(frequent_itemsets)
    association_rules = generate_association_rules(frequent_itemsets, min_confidence,min_correlation, data, correlation)
    nomber_regles = len(association_rules)
    return association_rules, nombre_itemsets, nomber_regles