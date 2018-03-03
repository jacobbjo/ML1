import monkdata as m
import dtree
import random
import drawtree_qt5
import matplotlib.pyplot as plt
import numpy as np

monks = [m.monk1, m.monk2, m.monk3]

# ------------------- Assignment 1
ent_monk1 = dtree.entropy(monks[0])
ent_monk2 = dtree.entropy(monks[1])
ent_monk3 = dtree.entropy(monks[2])

print("Entropy monk1: " + str(round(ent_monk1, 4)))
print("Entropy monk2: " + str(round(ent_monk2, 4)))
print("Entropy monk3: " + str(round(ent_monk3, 4)))
print("")

# ------------------- Assignment 3
# Table with Monk1-3 on rows and attributes a1-a6 on columns
gains = [[],[],[]]

print("Information gains:")
for i in range(len(monks)):
    for attribute in m.attributes:
        gains[i].append(round(dtree.averageGain(monks[i], attribute),5))
    print(gains[i])
print("")

# ------------------- 5 Building Decision Trees
a5_sub = []
a5_gains = [[],[],[],[]]

# Subsets for selected attribute a5 from monk 1
a5_sub.append(dtree.select(monks[0], m.attributes[4], 1))
a5_sub.append(dtree.select(monks[0], m.attributes[4], 2))
a5_sub.append(dtree.select(monks[0], m.attributes[4], 3))
a5_sub.append(dtree.select(monks[0], m.attributes[4], 4))

# Calculating the gains for all attr
#print(len(a5_sub[0]))
print("LEVEL 1")
for i in range(len(a5_sub)):
    for attribute in m.attributes:
        a5_gains[i].append(round(dtree.averageGain(a5_sub[i], attribute),5))
    print(a5_gains[i])

# Making level 2

# Chosen attributes for level 2

#for subset in a5_sub:
#    for i in range(len(subset)):
#        print(subset[i])
#print("------------")

level2_attr = [m.attributes[0], m.attributes[3], m.attributes[5], m.attributes[0]]

level2_sub = []
a5_gains_sub = []

for j in range(len(level2_attr)):
    attr_sub = []
    for value in level2_attr[j].values:
        print("Value: "+str(value))
        attr_sub.append(dtree.select(a5_sub[j], level2_attr[j], value))

    level2_sub.append(attr_sub)


for sub in level2_sub:
    # sub = one attribute's subset
    sub_gains = [] # The gains for one subset
    for i in range(len(sub)):
        attr_gains = [] # the gains for one attribute
        for attribute in m.attributes:
            attr_gains.append(round(dtree.averageGain(sub[i], attribute),5))
        sub_gains.append(attr_gains)
    a5_gains_sub.append(sub_gains)


print("")
print("LEVEL 2")
# Printing

for attr in a5_gains_sub:
    for sub in attr:
        print(sub)

    print("")

#print("Majority classes")
# Finding the majority class of the subsets
#for subs in level2_sub:
#    for sub in subs:
#        print(dtree.mostCommon(sub))


# ------------------- Assignment 5
# Building the tree with given function

tree0 = dtree.buildTree(monks[0], m.attributes)
#drawtree_qt5.drawTree(tree0)

tree1 = dtree.buildTree(monks[1], m.attributes)
tree2 = dtree.buildTree(monks[2], m.attributes)

print("Monk-1 train: " + str(dtree.check(tree0, monks[0])))
print("Monk-2 train: " + str(dtree.check(tree1, monks[1])))
print("Monk-3 train: " + str(dtree.check(tree2, monks[2])))

print("Monk-1 test: " + str(dtree.check(tree0, m.monk1test)))
print("Monk-2 test: " + str(dtree.check(tree1, m.monk2test)))
print("Monk-3 test: " + str(dtree.check(tree2, m.monk3test)))
print("")

# 6 Pruning

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def pruning(data_set, fraction = 0.6):
    # A function that returns a pruned decision tree from a data set
    data_train, data_val = partition(data_set, fraction)

    # The tree to become pruned
    tree_pruned = dtree.buildTree(data_train, m.attributes)
    err_tree_pru = dtree.check(tree_pruned, data_val)
#    print("Tree before prune:")
#    print(tree_pruned)

    better = True
    while better:
        better = False
        trees_alt = dtree.allPruned(tree_pruned)
        best_prune = None
        err_best = 0

        for alternative in trees_alt:
            err_alternative = dtree.check(alternative, data_val)

            if err_alternative >= err_tree_pru and err_alternative > err_best:
                best_prune = alternative
                err_best = err_alternative
                better = True

        if better:
            tree_pruned = best_prune
            err_tree_pru = err_best

    return tree_pruned


tree_pruned = pruning(m.monk1)
print("Tree after prune:")
print(tree_pruned)
#drawtree_qt5.drawTree(tree_pruned)


# Assignment 7
fractions = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
numLoops = 100


err_prune1 = np.empty([len(fractions), numLoops])
err_prune3 = np.empty([len(fractions), numLoops])


for j in range(len(fractions)):
    for i in range(numLoops):
        tree_pruned1 = pruning(m.monk1, fractions[j])
        tree_pruned3 = pruning(m.monk3, fractions[j])

        err_prune1[j, i] = 1 - dtree.check(tree_pruned1, m.monk1test)
        err_prune3[j, i] = 1 - dtree.check(tree_pruned3, m.monk3test)


mean1 = np.mean(err_prune1, axis=1)
mean3 = np.mean(err_prune3, axis=1)
var1 = np.var(err_prune1, axis=1)
var3 = np.var(err_prune3, axis=1)

fig1 = plt.figure(figsize=(14,7))

ax1 = fig1.add_subplot(121)
ax1.plot(fractions, mean1, c = "r", label="MONK-1")
ax1.plot(fractions, mean3, c = "b", label= "MONK-3")
ax1.scatter(fractions, mean1, c = "k")
ax1.scatter(fractions, mean3, c = "k")
ax1.legend(loc="upper left")
ax1.set_ylabel("Classification error mean")
ax1.set_xlabel("Fraction")
plt.grid()


ax2 = fig1.add_subplot(122)
ax2.plot(fractions, var1, c = "r", label="MONK-1")
ax2.plot(fractions, var3, c = "b", label="MONK-3")
ax2.scatter(fractions, var1, c = "k")
ax2.scatter(fractions, var3, c = "k")
ax2.legend(loc="upper right")
ax2.set_ylabel("Variance")
ax2.set_xlabel("Fraction")
plt.grid()

plt.show()