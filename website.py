import streamlit as st

def main():

    st.title('GREATEST OF THREE NUMBERS')
    st.header('PROGRAM:')
    greatest_of_three_numbers = """
    read_number(Number) :-
        read(Number),
        number(Number).

    main :-
        write('Enter the first number: '),
        read_number(Num1),
        write('Enter the second number: '),
        read_number(Num2),
        Sum is Num1 + Num2,
        format('The sum of ~w and ~w is ~w~n', [Num1, Num2, Sum]).
    """
    st.code(greatest_of_three_numbers, language='prolog')

    # Display the Prolog query
    st.header('QUERIES:')
    query_code = """
    ?- main.
    Enter the first number: 5.
    Enter the second number: 3.
    The sum of 5 and 3 is 8
    true
    """
    st.code(query_code, language='prolog')
###############################################################################
    st.title('Minimum and Maximum of Two Numbers')
    
    # Display the Prolog program for greatest of three numbers
    st.header('PROGRAM:')
    minmax = """
    read_number(Number) :-
        read(Number),
        number(Number).

    max_number(X, Y, Max) :-
        (X >= Y -> Max = X ; Max = Y).

    min_number(X, Y, Min) :-
        (X =< Y -> Min = X ; Min = Y).

    main :-
        write('Enter the first number: '),
        read_number(Num1),
        write('Enter the second number: '),
        read_number(Num2),
        max_number(Num1, Num2, Max),
        min_number(Num1, Num2, Min),
        format('The maximum of ~w and ~w is ~w~n', [Num1, Num2, Max]),
        format('The minimum of ~w and ~w is ~w~n', [Num1, Num2, Min]).
    """
    st.code(minmax, language='prolog')

    # Display the Prolog query
    st.header('QUERIES:')
    query_code1 = """
    ?- main.
    Enter the first number: 7.
    Enter the second number: 3.
    The maximum of 7 and 3 is 7
    The minimum of 7 and 3 is 3
    true.
    """
    st.code(query_code1, language='prolog')
###################################################################################
    st.title('Check if a Number is Odd or even')

    # Display the Prolog program
    st.header('PROGRAM:')
    oddoreven = """
    is_odd(Number) :-
        0 is mod(Number, 2),
        !,
        false.
    is_odd(_).
    """
    st.code(oddoreven, language='prolog')

    # Display the Prolog query
    st.header('QUERIES:')
    query_code2 = """
    ?- is_odd(3).
    true.
    """
    st.code(query_code2, language='prolog')
################################################################################
    st.title('Factorial of a Given Number')

    # Display the Prolog program
    st.header('PROGRAM:')
    fact = """
    read_number(Number) :-
        read(Number),
        number(Number).

    factorial(0, 1).
    factorial(N, Result) :-
        N > 0,
        N1 is N - 1,
        factorial(N1, F1),
        Result is N * F1.

    main :-
        write('Enter a number: '),
        read_number(N),
        factorial(N, Result),
        format('The factorial of ~w is ~w~n', [N, Result]).
    """
    st.code(fact, language='prolog')
    # Display the Prolog query
    st.header('QUERIES:')
    query_code3 = """
    ?- main.
    Enter a number: 5.
    The factorial of 5 is 120
    true.
    """
    st.code(query_code3, language='prolog')
##############################################################################
    st.title('N-Queens Problem(prolog code)')

    # Display the Prolog program
    st.header('PROGRAM:')
    nqueen = """
    :- use_module(library(clpfd)).

    n_queens(N, Qs) :-
        length(Qs, N),
        Qs ins 1..N,
        safe_queens(Qs).

    safe_queens([]).
    safe_queens([Q|Qs]) :-
        safe_queens(Qs, Q, 1),
        safe_queens(Qs).

    safe_queens([], _, _).
    safe_queens([Q|Qs], Q0, D0) :-
        Q0 #\\= Q,
        abs(Q0 - Q) #\\= D0,
        D1 #= D0 + 1,
        safe_queens(Qs, Q0, D1).
    """
    st.code(nqueen, language='prolog')

    # Display the Prolog queries
    st.header('QUERIES:')
    query1_code = """
    ?- n_queens(8, Qs), label(Qs).
    Qs = [1, 5, 8, 6, 3, 7, 2, 4]
    """
    st.code(query1_code, language='prolog')

    query2_code = """
    ?- n_queens(4, Qs), label(Qs).
    Qs = [2, 4, 1, 3]
    """
    st.code(query2_code, language='prolog')

    query3_code = """
    ?- n_queens(2, Qs), label(Qs).
    false
    """
    st.code(query3_code, language='prolog')
####################################################################
    st.title('N-Queens Problem(python code)')

    st.header('PROGRAM:')

    nqueen1="""
            N = 4
ld = [0] * 30
rd = [0] * 30
cl = [0] * 30

def printSolution(board):
    for i in range(N):
        for j in range(N):
            print(" Q " if board[i][j] == 1 else " . ", end="")
        print()
    print()

def solveNQUtil(board, col):
    if col >= N:
        return True
    for i in range(N):
        if (ld[i - col + N - 1] != 1 and rd[i + col] != 1 and cl[i] != 1):
            board[i][col] = 1
            ld[i - col + N - 1] = rd[i + col] = cl[i] = 1
            if solveNQUtil(board, col + 1):
                return True
            board[i][col] = 0
            ld[i - col + N - 1] = rd[i + col] = cl[i] = 0
    return False

def solveNQ():
    board = [[0 for _ in range(N)] for _ in range(N)]
    if not solveNQUtil(board, 0):
        print("Solution does not exist")
        return False
    printSolution(board)
    return True

if __name__ == "__main__":
    solveNQ()

            """
    st.code(nqueen1, language='python')
    ###################################################
    st.title('traveling salesman (using prolog)')

    st.header('PROGRAM:')

    salesman1="""
road(birmingham, bristol, 9).
road(london, birmingham, 3).
road(london, bristol, 6).
road(london, plymouth, 5).
road(plymouth, london, 5).
road(portsmouth, london, 4).
road(portsmouth, plymouth, 8).

get_road(Start, End, Visited, Result) :-
    get_road(Start, End, [Start], 0, Visited, Result).

get_road(Start, End, Waypoints, DistanceAcc, Visited, TotalDistance) :-
    road(Start, End, Distance),
    reverse([End|Waypoints], Visited),
    TotalDistance is DistanceAcc + Distance.

get_road(Start, End, Waypoints, DistanceAcc, Visited, TotalDistance) :-
    road(Start, Waypoint, Distance),
    \+ member(Waypoint, Waypoints),
    NewDistanceAcc is DistanceAcc + Distance,
    get_road(Waypoint, End, [Waypoint|Waypoints], NewDistanceAcc, Visited, TotalDistance).

            """
    st.code(salesman1, language='prolog')

    st.header('QUERIES:')
    saleman1 = """
?- get_road(portsmouth, plymouth, Visited, Distance).
Distance = 8,
Visited = [portsmouth, plymouth]
    """
    st.code(saleman1, language='prolog')

########################################################
    st.title('traveling sales man(python code)')

    st.header('PROGRAM:')

    salesman2="""
            from queue import PriorityQueue

def best_first_search(adj_list, heuristics, start):
    visited = set()
    pq = PriorityQueue()
    pq.put((heuristics[start], start))
    path = []

    while not pq.empty():
        cost, node = pq.get()
        if node not in visited:
            path.append((node, cost))
            visited.add(node)
            if len(visited) == len(adj_list):
                return path
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    pq.put((heuristics[neighbor], neighbor))
    return path

if __name__ == "__main__":
    adj_list = {
        'A': ['D', 'C'],
        'B': ['E'],
        'C': ['B', 'E'],
        'D': ['C'],
        'E': ['D', 'A']
    }
    heuristics = {
        'A': 2,
        'B': 5,
        'C': 4,
        'D': 3,
        'E': 6
    }
    start_node = input("Enter start node: ")
    print("\nAdjacency List : ", adj_list)
    print("\nHeuristics : ", heuristics)
    print("\nStarting node = ", start_node)
    print("\nPath followed :")
    path = best_first_search(adj_list, heuristics, start_node)
    total_cost = 0
    for node, cost in path[:-1]:
        total_cost += heuristics[node]
        print(node, "(", cost, ") - Total Cost =", total_cost)
    last_node, last_cost = path[-1]
    total_cost += heuristics[last_node]
    print(last_node, "(", last_cost, ") - Total Cost =", total_cost)
    print(path[0][0])

            """
    st.code(salesman2, language='python')
#####################################################################
    st.title('A* search')

    st.header('PROGRAM:')

    A="""
           from collections import deque

class Graph:
    def __init__(self, adjacency_list):
        self.adjacency_list = adjacency_list

    def get_neighbors(self, v):
        return self.adjacency_list[v]

    def h(self, n):
        H = {
            'A': 1,
            'B': 1,
            'C': 1,
            'D': 1
        }
        return H[n]

    def a_star_algorithm(self, start_node, stop_node):
        open_list = set([start_node])
        closed_list = set([])

        g = {}
        g[start_node] = 0

        parents = {}
        parents[start_node] = start_node

        while len(open_list) > 0:
            n = None
            for v in open_list:
                if n == None or g[v] + self.h(v) < g[n] + self.h(n):
                    n = v

            if n == None:
                print('Path does not exist!')
                return None

            if n == stop_node:
                reconst_path = []
                while parents[n] != n:
                    reconst_path.append(n)
                    n = parents[n]
                reconst_path.append(start_node)
                reconst_path.reverse()
                print('Path found: {}'.format(reconst_path))
                return reconst_path

            for (m, weight) in self.get_neighbors(n):
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n
                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)

            open_list.remove(n)
            closed_list.add(n)

        print('Path does not exist!')
        return None

adjacency_list = {
    'A': [('B', 1), ('C', 3), ('D', 7)],
    'B': [('D', 5)],
    'C': [('D', 12)]
}
graph1 = Graph(adjacency_list)
graph1.a_star_algorithm('A', 'D')

            """
    st.code(A, language='python')
###########################################################
    st.title('wumpus world')

    st.header('PROGRAM:')

    wumpus="""
            import random

class WumpusWorld:
    def __init__(self, size=4):
        self.size = size
        self.board = [[' ' for _ in range(size)] for _ in range(size)]
        self.agent_pos = [0, 0]
        self.wumpus_pos = self.random_empty_cell()
        self.gold_pos = self.random_empty_cell()
        self.pits = [self.random_empty_cell() for _ in range(size-1)]
        self.board[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        self.board[self.wumpus_pos[0]][self.wumpus_pos[1]] = 'W'
        self.board[self.gold_pos[0]][self.gold_pos[1]] = 'G'
        for pit in self.pits:
            self.board[pit[0]][pit[1]] = 'P'

    def random_empty_cell(self):
        while True:
            cell = [random.randint(0, self.size-1), random.randint(0, self.size-1)]
            if cell != [0, 0] and self.board[cell[0]][cell[1]] == ' ':
                return cell

    def print_board(self):
        for row in self.board:
            print(' | '.join(row))
            print('-' * (self.size * 4 - 1))

    def get_percepts(self):
        percepts = []
        if self.is_adjacent(self.agent_pos, self.wumpus_pos):
            percepts.append('You smell a Wumpus!')
        if self.is_adjacent(self.agent_pos, self.gold_pos):
            percepts.append('You see a glimmer!')
        for pit in self.pits:
            if self.is_adjacent(self.agent_pos, pit):
                percepts.append('You feel a breeze!')
        return percepts

    def is_adjacent(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def move_agent(self, row, col):
        if 0 <= row < self.size and 0 <= col < self.size:
            self.agent_pos = [row, col]

        if self.agent_pos == self.wumpus_pos:
            return 'You were eaten by the Wumpus!'
        if self.agent_pos in self.pits:
            return 'You fell into a pit!'
        if self.agent_pos == self.gold_pos:
            return 'You found the gold and won!'
        return None
    
    def available_moves(self):
        moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for direction in directions:
            new_row = self.agent_pos[0] + direction[0]
            new_col = self.agent_pos[1] + direction[1]
            if 0 <= new_row < self.size and 0 <= new_col < self.size:
                moves.append((new_row, new_col))
        return moves


game = WumpusWorld()
while True:
    game.print_board()
    percepts = game.get_percepts()
    for p in percepts:
        print(p)
    
    available_moves = game.available_moves()
    print("Available moves:", available_moves)
    
    move = input("Enter move (row col): ").split()
    row, col = int(move[0]), int(move[1])
    result = game.move_agent(row, col)
    if result:
        game.print_board()
        print(result)
        break

            """
    st.code(wumpus, language='python')

    ######################################################

    st.title('TIC TAC TOE alpha beta pruning')

    st.header('PROGRAM:')

    tic="""
import math

def init_board():
    return [[' ' for _ in range(3)] for _ in range(3)]

def print_board(board):
    for row in board:
        print("|".join(row))
        print("-" * 5)

def check_winner(board, player):
    for i in range(3):
        if all([board[i][j] == player for j in range(3)]) or all([board[j][i] == player for j in range(3)]):
            return True
    if board[0][0] == board[1][1] == board[2][2] == player or board[0][2] == board[1][1] == board[2][0] == player:
        return True
    return False

def is_full(board):
    return all([cell != ' ' for row in board for cell in row])

def evaluate(board):
    if check_winner(board, 'X'):
        return 1
    elif check_winner(board, 'O'):
        return -1
    else:
        return 0

def minimax(board, depth, is_maximizing, alpha, beta):
    score = evaluate(board)
    if score == 1 or score == -1 or is_full(board):
        return score
    if is_maximizing:
        max_eval = -math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'X'
                    eval = minimax(board, depth + 1, False, alpha, beta)
                    board[i][j] = ' '
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
        return max_eval
    else:
        min_eval = math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'O'
                    eval = minimax(board, depth + 1, True, alpha, beta)
                    board[i][j] = ' '
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
        return min_eval

def best_move(board):
    best_val = -math.inf
    move = (-1, -1)
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = 'X'
                move_val = minimax(board, 0, False, -math.inf, math.inf)
                board[i][j] = ' '
                if move_val > best_val:
                    move = (i, j)
                    best_val = move_val
    return move

board = init_board()
current_player = 'O'
while True:
    print_board(board)
    if current_player == 'O':
        row, col = map(int, input("Enter your move (row col): ").split())
    else:
        row, col = best_move(board)
    if board[row][col] == ' ':
        board[row][col] = current_player
    else:
        print("Invalid move. Try again.")
        continue
    if check_winner(board, current_player):
        print_board(board)
        print(f"Player {current_player} wins!")
        break
    if is_full(board):
        print_board(board)
        print("It's a draw!")
        break
    current_player = 'X' if current_player == 'O' else 'O'


            """
    st.code(tic, language='python')
    ###################################################

    st.title('genetic')

    st.header('PROGRAM:')

    genetic="""
            import random

target = input("Enter the target string: ")
population_size = 100
mutation_rate = 0.01

GENES = ''.join([chr(i) for i in range(32, 127)])

population = [''.join([random.choice(GENES) for _ in range(len(target))]) for _ in range(population_size)]

def calculate_fitness(member):
    return sum([1 for i in range(len(target)) if member[i] == target[i]])

generation = 1

while True:
    best_member = max(population, key=calculate_fitness)
    best_fitness = calculate_fitness(best_member)
    print(f"Generation {generation} - Best member: {best_member}, Fitness: {best_fitness}")

    if best_fitness == len(target):
        print("Target achieved!")
        break

    elite_count = int(0.1 * population_size)
    elite = sorted(population, key=calculate_fitness, reverse=True)[:elite_count]
    new_population = elite[:]

    while len(new_population) < population_size:
        parent1, parent2 = random.choices(population, k=2)
        crossover_point = random.randint(1, len(target) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        child = ''.join([char if random.random() > mutation_rate else random.choice(GENES) for char in child])
        new_population.append(child)
        
    population = new_population
    generation += 1

            """
    st.code(genetic, language='python')
    ####################################################
    st.title('logic evaluator')

    st.header('PROGRAM:')

    evaluator="""
            def evaluate(exp):
    result = eval(exp)
    if result:
        print("The proposition is True.")
    else:
        print("The proposition is False.")

def main():
    print("Enter your expression:")
    exp = input().strip()
    evaluate(exp)

if __name__ == "__main__":
    main()

            """
    st.code(evaluator, language='python')
    ########################################################

    st.title('logic operator evaluator')

    st.header('PROGRAM:')

    operator="""
            P = input("Enter truth value for P (True or False): ").strip()
Q = input("Enter truth value for Q (True or False): ").strip()
s = P and Q
f = P or Q
g = not P
print(f"P AND Q = {s}")
print(f"P OR Q = {f}")
print(f"NOT P = {g}")

            """
    st.code(operator, language='python')

    #######################################################
    st.title('annealing problem')

    st.header('PROGRAM:')

    annealing="""
           from numpy import asarray
from numpy import exp
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed
from matplotlib import pyplot

def objective(x):
    return x[0]**2.0

def simulated_annealing(objective, bounds, n_iterations, step_size, temp):
    best = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    best_eval = objective(best)
    curr, curr_eval = best, best_eval
    scores = list()
    for i in range(n_iterations):
        candidate = curr + randn(len(bounds)) * step_size
        candidate_eval = objective(candidate)
        if candidate_eval < best_eval:
            best, best_eval = candidate, candidate_eval
            scores.append(best_eval)
        diff = candidate_eval - curr_eval
        t = temp / float(i + 1)
        metropolis = exp(-diff / t)
        if diff < 0 or rand() < metropolis:
            curr, curr_eval = candidate, candidate_eval
    return [best, best_eval, scores]

seed(1)
bounds = asarray([[-5.0, 5.0]])
n_iterations = 1000
step_size = 0.1
temp = 10

best, score, scores = simulated_annealing(objective, bounds, n_iterations, step_size, temp)
print('Done!')
print('f(%s) = %f' % (best, score))

pyplot.plot(scores, '.-')
pyplot.xlabel('Improvement Number')
pyplot.ylabel('Evaluation f(x)')
pyplot.show()

            """
    st.code(annealing, language='python')
    
if __name__ == '__main__':
    main()
