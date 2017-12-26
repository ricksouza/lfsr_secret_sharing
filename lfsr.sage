from sage.all import *
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import isomorphism

class AuthorizedSubSet:

	def __str__(self):
		return str(self.subset)

	def __init__(self, subset):
		self.subset = subset
		
	def contains_participant(self, participant):
		
		return participant in self.subset

def lfsr_sequence_2(prime, f, initial_values, m, n):
	
	F = GF(prime)
	sequence = list()
	
	for i in range (0, n):
		if i < m:
			sequence.append(initial_values[F(i)])
		else:
			next_element = 0
			for j in range (0, m-1):
				next_element = next_element - f[j] * sequence[i - (m-j)]
				#next_element = next_element - F(F(f[j]) * sequence[ F(i) - F((m-j))])
			sequence.append(next_element)
	
	sequence_deque = deque(sequence)
	return sequence_deque
	
def trace_sequence(n, m, prime, f):
	
	R.<x> = PolynomialRing(GF(prime), 'x')
	new_sequence = list()
	for i in range(0, n):
		trace = 0
		for j in range(0, m):
			trace = trace + (x^(i*prime^j)) % f
		new_sequence.append(trace)
		
	#print("Trace sequence")
	#print(new_sequence)
	
	return new_sequence
	
def trace_function(prime, m, exp, f):
	
	R.<x> = PolynomialRing(GF(prime), 'x')
	trace = 0
	for j in range(0, m):
		print((x^(exp*prime^j)))
		print((x^(exp*prime^j)) % f)
		trace = trace + (x^(exp*prime^j)) % f
	
	return trace
	
	
def create_subarray(sequence, prime, m, k):
	
	subarray = list()
	
	sequence_deque = deque(sequence)
	#print(k)
	
	#Creating the first zero line
	zero_line = [0] * int(k)
	
	subarray.append(zero_line)

	for i in range (0, prime^m - 1):
		sequence_deque.rotate(-1)
		seq_slice = list(sequence_deque)[:k]
		print(seq_slice)
		subarray.append(seq_slice)
		
	return subarray
	
def get_linear_dependent_columns(subarray, prime, m, k):
	
	
	ldc = []
	
	for j in range(1, k+1):
	
		index_of_zeroes = list()
		element = subarray[j]
		#print("ELEMENT")
		#print(element)
		for i in range(0, len(element)):
			if(element[i] == 0):
				index_of_zeroes.append(i)
		C = Combinations(index_of_zeroes, 3)
		list_comb = C.list()
		#print("list_comb")
		#print(list_comb)
		for k in range(0, len(list_comb)):
			#if list_comb[k][0] == 0:
			ldc.append(list_comb[k])
		#print(ldc)
	return ldc
	
def get_linear_independent_columns(ldc, k):
	
	list_ind_col_ind = range(0, k)
	
	C = Combinations(list_ind_col_ind, 3)
	
	lic = C.list()
	list_ind_col = list()
	for el in lic:
		if el not in ldc:
			list_ind_col.append(el)
			
	#print(list_ind_col)
	
	return list_ind_col
	
def create_access_structure_graph(access_structure, participants):
	
	#print(access_structure)
	#print(participants)
	
	graph = nx.Graph()
	
	graph.add_nodes_from(participants, color='b')
	#list_of_auth_ac = list()
	
	#for ac in access_structure:
		#auth_ac = AuthorizedSubSet(ac)
	#	str_ac = str(ac)
	#	list_of_auth_ac.append(str_ac)
	#	graph.add_node(str_ac)
		
	for part in participants:
		for ac in access_structure:
			#print(part)
			#print(ac)
			
			if part in ac:
				#if(part == 6 or part == 10 or part == 11):
					#print(part)
					#print(ac)
				str_ac = str(ac)
				graph.add_node(str_ac, size=20, color='b')
				graph.add_edge(part, str_ac, color= 'grey', weight=0.5)
		
	#print(graph.order())
	
	return graph
	
def create_secret_values(prime, m, random_choice, f):

	secret_list = list()
	secret = trace_function(prime, m, random_choice, f)
	secret2 = trace_function(prime, m, random_choice+1, f)
	secret3 = trace_function(prime, m, random_choice+2, f)
	print("Segredo: %s" % (secret))
	
	secret_list.append(secret)
	secret_list.append(secret2)
	secret_list.append(secret3)
	
	return secret_list
			
def main():

	prime = 3
	R.<x> = PolynomialRing(GF(prime), 'x')
	
	m = 3
	f = x^m + 2*x + 1
	initial_values = [0, 1, 2]
	n = prime^m - 1
	
	#Creating lfsr sequence using polynomial
	sequence = lfsr_sequence_2(prime, f, initial_values, m, n)
	#print(sequence)

	#Creating lfsr sequence using trace function
	new_sequence = trace_sequence(n, m, prime, f)
	#print(new_sequence)
	
	#Dividing the sequence in k columns
	k = (prime^m - 1)/(prime-1)
	
	#Creating subarray
	subarray = create_subarray(new_sequence, prime, m, k)
	
	
	#for el in subarray:
	#	print(el)
		
	#Getting linear dependent columns. That is, the combination of columns 
	#that have zero in its values
	ldc = get_linear_dependent_columns(subarray, prime, m, k)
	#print("Linear Dependent Columns: ")
	#print(ldc)
	
	#Getting linear independent columns. That is, the combinations of columns
	#that are not in ldc.
	lic = get_linear_independent_columns(ldc, k)
	#print("Linear Independent Columns: ")
	#print(lic)
	
	participants = [0, 1, 2, 3]
	access_structure = [ [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]

		
	graph2 = create_access_structure_graph(access_structure, participants)
	graph2.add_node(4, color = 'b')
	graph2.add_node(5, color = 'b')
	graph2.add_edge(0, 4,color= 'grey', weight=0.5)
	graph2.add_edge(4, 5, color= 'grey', weight=0.5)
	graph2.add_edge(5, 0, color= 'grey', weight=0.5)
	#print(nx.info(graph2, 0))
	#print(nx.info(graph2, 1))
	#print(nx.info(graph2, 2))
	#print(nx.info(graph2, 3))
	
	list_el = range(0, 13)
	#print(lic)
	graph = create_access_structure_graph(lic, list_el)
	graph.add_node('b', color = 'b')
	graph.add_node('c', color = 'b')
	graph.add_edge(0, 'b', color= 'grey', weight=0.5)
	graph.add_edge('b', 'c', color= 'grey', weight=0.5)
	graph.add_edge('c', 0, color= 'grey', weight=0.5)
	#print(nx.info(graph, 0))
	#print(nx.info(graph, 1))
	#print(nx.info(graph, 2))
	#print(nx.info(graph, 3))
	#print(nx.info(graph, 4))
	#print("<<<<<<<<<<<<<NODES>>>>>>>>>>>>>")
	#print(nx.nodes(graph2))
	#print("<<<<<<<<<<<<<EDGES>>>>>>>>>>>>>")
	#print(nx.edges(graph2))
	
	plt.figure(figsize=(10,10), dpi=500)
	nx.draw_circular(graph2)
	nx.draw_networkx_labels(graph2, pos = nx.circular_layout(graph2))
	plt.savefig("graph2.pdf")
	plt.clf()
	#list_el = range(0, 12)
	#graph2 = create_access_structure_graph(lic, list_el)
	plt.figure(figsize=(10,10), dpi=50)
	plt.xlim(0, 1)
	plt.ylim(0, 1)
	plt.axis('off')
	#pos = nx.spring_layout(graph)
	
	edges = graph.edges()
	colors = [graph[u][v]['color'] for u,v in edges]
	weights = [graph[u][v]['weight'] for u,v in edges]
	nodes = graph.nodes(data=True)
	
	#node_colors = [graph[u][v]['color'] for u,v in nodes]
	
	#nx.draw_networkx_labels(graph2,pos=nx.spring_layout(graph2))
	pos = nx.spring_layout(graph, k=0.08)
	#nx.draw(graph, pos, node_size=20, font_size=8, arrows=False)
	nx.draw_networkx_nodes(graph, pos, node_color='b', node_size=20)
	nx.draw_networkx_nodes(graph, pos, node_color='r', nodelist=[0, 5, 9, 11, 'b', 'c'], node_size=60)
	#nx.draw(graph, pos)
	
	#nx.draw(graph, pos, node_size=20, edges=edges, edge_color=colors, width=weights, node_color=['b', 'r', 'g'])
	nx.draw_networkx_edges(graph, pos, edges=edges, edge_color=colors, width=weights)
	#plt.figure(2, figsize=(1, 1))
	
	#nx.draw_spring(graph)
	#nx.draw_networkx_labels(graph, pos = nx.spring_layout(graph))
	
	plt.savefig("graph.pdf", bbox_inches="tight")
	
	GM = isomorphism.GraphMatcher(graph,graph2)
	print(GM.subgraph_is_isomorphic())
	print(GM.mapping)
	#print(GM.mapping)
	#print(GM.mapping)
	
	columns = list()
	
	mapping = GM.mapping
	for key, value in mapping.items():
		#print(key)
		print(value)
		if value in participants and key != 0:
			columns.append(key)
			
	print("Colunas: ")
	print(columns)
	
	random_choice = randint(0, n)
	print("Linha: ")
	print(random_choice)
	#subarray_line = subarray[random_choice]
	
	secret_list = create_secret_values(prime, m, random_choice, f)
	#trace_values = create_trace_values(prime, m, random_choice, f)
	
	trace_values = list()
	
	for col in columns:
		values = list()
		print("Valores associados a coluna: ")
		#print(subarray_line[col])
		part_value = trace_function(prime, m, random_choice+col, f)
		part_value2 = trace_function(prime, m, random_choice+col+1, f)
		part_value3 = trace_function(prime, m, random_choice+col+2, f)
		print(part_value)
		print(part_value2)
		print(part_value3)
		values.append(part_value)
		values.append(part_value2)
		values.append(part_value3)
		trace_values.append(values)	
		
	matrix_values = [[trace_values[0][0], trace_values[1][0], trace_values[2][0]], 
					[trace_values[0][1], trace_values[1][1], trace_values[2][1]], 
					[trace_values[0][2], trace_values[1][2], trace_values[2][2]]]
	
	M = MatrixSpace(GF(prime),3,3)
	V = VectorSpace(GF(prime),3)
	A = M(matrix_values)
	print(A)
	print(secret_list)
	print(len(secret_list))
	Y = V(secret_list)
	
	X = A.solve_right(Y)
	
	print(X)
	
	print(trace_values[0][0])
	print(trace_values[1][0])
	print(trace_values[2][0])
	
	s = trace_values[0][0]*X[0] + trace_values[1][0]*X[1] + trace_values[2][0]*X[2]
	print("Segredo: ")
	print(s)

def test():
	prime = 3
	R.<x> = PolynomialRing(GF(prime), 'x')
	
	m = 3
	f = x^m + 2*x + 1
	initial_values = [0, 1, 2]
	n = prime^m - 1
	
	#print("1")
	#print(trace_function(prime, m, 1, f))
	print("2")
	print(trace_function(prime, m, 2, f))
	#print("3")
	#print(trace_function(prime, m, 3, f))
	#print("4")
	#print(trace_function(prime, m, 4, f))
	#print(trace_function(prime, m, 9, f))

test()