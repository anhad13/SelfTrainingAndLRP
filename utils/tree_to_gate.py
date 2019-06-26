def tree_to_gates(tree):
    #l : length of sentence, tree is list.
    if len(tree) == 1:
        return [1]
    l = leafno(tree)
    arr = [1] * l
    assign_recursively(tree, arr, 0, max = l+1)
    return arr


def assign_recursively(tree, arr, base = 0, max=10):
	if type(tree) == list:
		assign_recursively(tree[0], arr, base, max-1)
		right_child = tree[1]
		count = 0
		while(1):
			if type(right_child) == str:
				arr[base+leafno(tree[0])] = max
				break
			elif count == 1:
				break
			else:
				right_child = right_child[0]
				count +=1
		assign_recursively(tree[1], arr, base+leafno(tree[0]), max-1)


def leafno(x):
	if type(x)==str or len(x)==1:
		return 1
	else:
		return leafno(x[0]) + leafno(x[1])


if __name__ == '__main__':
	#example: tree:  [[['a', 'b'], 'c'], ['d', 'e']],
	#no of tokens: 5
    print(tree_to_gates([[['a', 'b'], 'c'], ['d', 'e']]))
    print(tree_to_gates([['a','b'],[['c','d'],'e']]))
    print(tree_to_gates([[['a', 'b'], 'c'], 'e']))
    print(tree_to_gates(['a', 'b']))
    print(tree_to_gates(['a']))
