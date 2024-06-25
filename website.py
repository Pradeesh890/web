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
    def __init__(self):
        self.size = 4
        self.world = [['' for _ in range(self.size)] for _ in range(self.size)]
        self.agent_pos = (0, 0)
        self.arrow = True
        self.gold_collected = False
        self.place_elements()

    def place_elements(self):
        # Place pits
        for _ in range(3):
            self.place_random('P')
        # Place Wumpus
        self.place_random('W')
        # Place Gold
        self.place_random('G')
        # Add percepts around Pits (Breeze) and Wumpus (Stench)
        self.add_percepts()

    def place_random(self, element):
        while True:
            row = random.randint(0, self.size - 1)
            col = random.randint(0, self.size - 1)
            if self.world[row][col] == '' and (row, col) != (0, 0):
                self.world[row][col] = element
                break

    def add_percepts(self):
        for row in range(self.size):
            for col in range(self.size):
                if self.world[row][col] == 'P':
                    self.add_adjacent_percept(row, col, 'B')
                elif self.world[row][col] == 'W':
                    self.add_adjacent_percept(row, col, 'S')

    def add_adjacent_percept(self, row, col, percept):
        for r, c in self.get_adjacent_cells(row, col):
            if self.world[r][c] == '':
                self.world[r][c] = percept
            elif self.world[r][c] not in ['P', 'W', 'G']:
                self.world[r][c] += percept

    def get_adjacent_cells(self, row, col):
        adjacent = []
        if row > 0: adjacent.append((row - 1, col))
        if row < self.size - 1: adjacent.append((row + 1, col))
        if col > 0: adjacent.append((row, col - 1))
        if col < self.size - 1: adjacent.append((row, col + 1))
        return adjacent

    def print_world(self):
        for row in self.world:
            print(row)

    def move_agent(self, row, col):
        if not (0 <= row < self.size and 0 <= col < self.size):
            print("Invalid move. Try again.")
            return False

        self.agent_pos = (row, col)
        cell = self.world[row][col]

        if 'P' in cell:
            if self.gold_collected:
                print("You were really close but unfortunately you failed!!! Try next time")
            else:
                print("Game over! Sorry, try next time!!!")
            return True

        if 'W' in cell:
            if self.gold_collected:
                print("You were really close but unfortunately you failed!!! Try next time")
            else:
                print("Game over! Sorry, try next time!!!")
            return True

        if 'G' in cell:
            self.gold_collected = True
            print("You found gold!")
            print("You have unlocked next level, move back to your initial position.")
            self.world[row][col] = self.world[row][col].replace('G', '')

        if 'S' in cell:
            print("You came across a stench")
        if 'B' in cell:
            print("You feel a breeze")

        if self.gold_collected and self.agent_pos == (0, 0):
            print("You won!!!")
            return True

        return False

    def get_possible_moves(self):
        row, col = self.agent_pos
        moves = self.get_adjacent_cells(row, col)
        return moves

def main():
    game = WumpusWorld()
    print("Initially agent is at 0,0")
    while True:
        possible_moves = game.get_possible_moves()
        for move in possible_moves:
            print(f"You can go to {move[0]} {move[1]}")

        try:
            row = int(input("Enter input for row: "))
            col = int(input("Enter input for column: "))
        except ValueError:
            print("Invalid input. Please enter integer values.")
            continue

        print(f"Now the agent is at {row},{col}")
        if game.move_agent(row, col):
            break

if __name__ == "__main__":
    main()

            """
    st.code(wumpus, language='python')

    ######################################################

    st.title('TIC TAC TOE alpha beta pruning')

    st.header('PROGRAM:')

    tic="""
            import time

class Game:
    def __init__(self):
        self.initialize_game()

    def initialize_game(self):
        self.current_state = [['.','.','.'],
                              ['.','.','.'],
                              ['.','.','.']]
        # Player X always plays first
        self.player_turn = 'X'

    def draw_board(self):
        for i in range(0, 3):
            for j in range(0, 3):
                print('{}|'.format(self.current_state[i][j]), end=" ")
            print()
        print()

    def is_valid(self, px, py):
        if px < 0 or px > 2 or py < 0 or py > 2:
            return False
        elif self.current_state[px][py] != '.':
            return False
        else:
            return True

    def is_end(self):
        # Vertical win
        for i in range(0, 3):
            if (self.current_state[0][i] != '.' and
                self.current_state[0][i] == self.current_state[1][i] and
                self.current_state[1][i] == self.current_state[2][i]):
                return self.current_state[0][i]
        # Horizontal win
        for i in range(0, 3):
            if (self.current_state[i] == ['X', 'X', 'X']):
                return 'X'
            elif (self.current_state[i] == ['O', 'O', 'O']):
                return 'O'
        # Main diagonal win
        if (self.current_state[0][0] != '.' and
            self.current_state[0][0] == self.current_state[1][1] and
            self.current_state[0][0] == self.current_state[2][2]):
            return self.current_state[0][0]
        # Second diagonal win
        if (self.current_state[0][2] != '.' and
            self.current_state[0][2] == self.current_state[1][1] and
            self.current_state[0][2] == self.current_state[2][0]):
            return self.current_state[0][2]
        # Is whole board full?
        for i in range(0, 3):
            for j in range(0, 3):
                # There's an empty field, we continue the game
                if (self.current_state[i][j] == '.'):
                    return None
        # It's a tie!
        return '.'

    def max(self):
        # Possible values for maxv are:
        # -1 - loss
        # 0 - a tie
        # 1 - win
        # We're initially setting it to -2 as worse than the worst case:
        maxv = -2
        px = None
        py = None
        result = self.is_end()
        # If the game came to an end, the function needs to return
        # the evaluation function of the end. That can be:
        # -1 - loss
        # 0 - a tie
        # 1 - win
        if result == 'X':
            return (-1, 0, 0)
        elif result == 'O':
            return (1, 0, 0)
        elif result == '.':
            return (0, 0, 0)
        for i in range(0, 3):
            for j in range(0, 3):
                if self.current_state[i][j] == '.':
                    # On the empty field player 'O' makes a move and calls Min
                    # That's one branch of the game tree.
                    self.current_state[i][j] = 'O'
                    (m, min_i, min_j) = self.min()
                    # Fixing the maxv value if needed
                    if m > maxv:
                        maxv = m
                        px = i
                        py = j
                    # Setting back the field to empty
                    self.current_state[i][j] = '.'
        return (maxv, px, py)

    def min(self):
        # Possible values for minv are:
        # -1 - win
        # 0 - a tie
        # 1 - loss
        # We're initially setting it to 2 as worse than the worst case:
        minv = 2
        qx = None
        qy = None
        result = self.is_end()
        if result == 'X':
            return (-1, 0, 0)
        elif result == 'O':
            return (1, 0, 0)
        elif result == '.':
            return (0, 0, 0)
        for i in range(0, 3):
            for j in range(0, 3):
                if self.current_state[i][j] == '.':
                    self.current_state[i][j] = 'X'
                    (m, max_i, max_j) = self.max()
                    if m < minv:
                        minv = m
                        qx = i
                        qy = j
                    self.current_state[i][j] = '.'
        return (minv, qx, qy)

    def max_alpha_beta(self, alpha, beta):
        maxv = -2
        px = None
        py = None
        result = self.is_end()
        if result == 'X':
            return (-1, 0, 0)
        elif result == 'O':
            return (1, 0, 0)
        elif result == '.':
            return (0, 0, 0)
        for i in range(0, 3):
            for j in range(0, 3):
                if self.current_state[i][j] == '.':
                    self.current_state[i][j] = 'O'
                    (m, min_i, in_j) = self.min_alpha_beta(alpha, beta)
                    if m > maxv:
                        maxv = m
                        px = i
                        py = j
                    self.current_state[i][j] = '.'
                    # Next two ifs in Max and Min are the only difference between regular algorithm and minimax
                    if maxv >= beta:
                        return (maxv, px, py)
                    if maxv > alpha:
                        alpha = maxv
        return (maxv, px, py)

    def min_alpha_beta(self, alpha, beta):
        minv = 2
        qx = None
        qy = None
        result = self.is_end()
        if result == 'X':
            return (-1, 0, 0)
        elif result == 'O':
            return (1, 0, 0)
        elif result == '.':
            return (0, 0, 0)
        for i in range(0, 3):
            for j in range(0, 3):
                if self.current_state[i][j] == '.':
                    self.current_state[i][j] = 'X'
                    (m, max_i, max_j) = self.max_alpha_beta(alpha, beta)
                    if m < minv:
                        minv = m
                        qx = i
                        qy = j
                    self.current_state[i][j] = '.'
                    if minv <= alpha:
                        return (minv, qx, qy)
                    if minv < beta:
                        beta = minv
        return (minv, qx, qy)

    def play_alpha_beta(self):
        while True:
            self.draw_board()
            self.result = self.is_end()
            if self.result != None:
                if self.result == 'X':
                    print('The winner is X!')
                elif self.result == 'O':
                    print('The winner is O!')
                elif self.result == '.':
                    print("It's a tie!")
                self.initialize_game()
                return
            if self.player_turn == 'X':
                while True:
                    start = time.time()
                    (m, qx, qy) = self.min_alpha_beta(-2, 2)
                    end = time.time()
                    print('Evaluation time: {}s'.format(round(end - start, 7)))
                    print('Recommended move: X = {}, Y = {}'.format(qx, qy))
                    px = int(input('Insert the X coordinate: '))
                    py = int(input('Insert the Y coordinate: '))
                    qx = px
                    qy = py
                    if self.is_valid(px, py):
                        self.current_state[px][py] = 'X'
                        self.player_turn = 'O'
                        break
                    else:
                        print('The move is not valid! Try again.')
            else:
                (m, px, py) = self.max_alpha_beta(-2, 2)
                self.current_state[px][py] = 'O'
                self.player_turn = 'X'

    def play(self):
        while True:
            self.draw_board()
            self.result = self.is_end()
            # Printing the appropriate message if the game has ended
            if self.result != None:
                if self.result == 'X':
                    print('The winner is X!')
                elif self.result == 'O':
                    print('The winner is O!')
                elif self.result == '.':
                    print("It's a tie!")
                self.initialize_game()
                return
            # If it's player's turn
            if self.player_turn == 'X':
                while True:
                    start = time.time()
                    (m, qx, qy) = self.min()
                    end = time.time()
                    print('Evaluation time: {}s'.format(round(end - start, 7)))
                    print('Recommended move: X = {}, Y = {}'.format(qx, qy))
                    px = int(input('Insert the X coordinate: '))
                    py = int(input('Insert the Y coordinate: '))
                    (qx, qy) = (px, py)
                    if self.is_valid(px, py):
                        self.current_state[px][py] = 'X'
                        self.player_turn = 'O'
                        break
                    else:
                        print('The move is not valid! Try again.')
            # If it's AI's turn
            else:
                (m, px, py) = self.max()
                self.current_state[px][py] = 'O'
                self.player_turn = 'X'

def main():
    g = Game()
    g.play()

if __name__ == "__main__":
    main()

            """
    st.code(tic, language='python')
    ###################################################

    st.title('genetic')

    st.header('PROGRAM:')

    genetic="""
            import random

# Number of individuals in each generation
POPULATION_SIZE = 100

# Valid genes
GENES = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP
QRSTUVWXYZ 1234567890, .-;:_!"#%&/()=?@${[]}'''

# Target string to be generated
TARGET = "I love GeeksforGeeks"

class Individual(object):
    '''
    Class representing individual in population
    '''
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.cal_fitness()

    @classmethod
    def mutated_genes(cls):
        '''
        Create random genes for mutation
        '''
        global GENES
        gene = random.choice(GENES)
        return gene

    @classmethod
    def create_gnome(cls):
        '''
        Create chromosome or string of genes
        '''
        global TARGET
        gnome_len = len(TARGET)
        return [cls.mutated_genes() for _ in range(gnome_len)]

    def mate(self, par2):
        '''
        Perform mating and produce new offspring
        '''
        # Chromosome for offspring
        child_chromosome = []
        for gp1, gp2 in zip(self.chromosome, par2.chromosome):
            # Random probability
            prob = random.random()
            # If prob is less than 0.45, insert gene from parent 1
            if prob < 0.45:
                child_chromosome.append(gp1)
            # If prob is between 0.45 and 0.90, insert gene from parent 2
            elif prob < 0.90:
                child_chromosome.append(gp2)
            # Otherwise insert random gene (mutate), for maintaining diversity
            else:
                child_chromosome.append(self.mutated_genes())
        
        # Create new Individual (offspring) using generated chromosome for offspring
        return Individual(child_chromosome)

    def cal_fitness(self):
        '''
        Calculate fitness score, it is the number of characters in string which differ from target string.
        '''
        global TARGET
        fitness = 0
        for gs, gt in zip(self.chromosome, TARGET):
            if gs != gt: 
                fitness += 1
        return fitness

# Driver code
def main():
    global POPULATION_SIZE
    
    # Current generation
    generation = 1
    found = False
    population = []

    # Create initial population
    for _ in range(POPULATION_SIZE):
        gnome = Individual.create_gnome()
        population.append(Individual(gnome))

    while not found:
        # Sort the population in increasing order of fitness score
        population = sorted(population, key=lambda x: x.fitness)

        # If the individual having lowest fitness score i.e. 0 then we know that we have reached the target and break the loop
        if population[0].fitness <= 0:
            found = True
            break

        # Otherwise generate new offsprings for new generation
        new_generation = []

        # Perform Elitism, that means 10% of fittest population goes to the next generation
        s = int((10 * POPULATION_SIZE) / 100)
        new_generation.extend(population[:s])

        # From 50% of fittest population, individuals will mate to produce offspring
        s = int((90 * POPULATION_SIZE) / 100)
        for _ in range(s):
            parent1 = random.choice(population[:50])
            parent2 = random.choice(population[:50])
            child = parent1.mate(parent2)
            new_generation.append(child)

        population = new_generation

        print("Generation: {}\tString: {}\tFitness: {}".format(
            generation, "".join(population[0].chromosome), population[0].fitness))

        generation += 1

    print("Generation: {}\tString: {}\tFitness: {}".format(
        generation, "".join(population[0].chromosome), population[0].fitness))

if __name__ == "__main__":
    main()

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
