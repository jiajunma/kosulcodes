from itertools import combinations, permutations
from math import comb, factorial
from typing import Set, Tuple, List
from enum import Enum
from RB import *


def generate_R(m, n):
    """
    Generate all rook boards (partial matchings) between two sets.
    A rook board is a set of pairs (i,j) where:
    - i ∈ [1..m] and j ∈ [1..n]
    - each i appears at most once
    - each j appears at most once
    
    This generates all possible rook boards, including the empty board.
    
    Args:
        m: Size of the first set (i in range [1, m])
        n: Size of the second set (j in range [1, n])
    
    Yields:
        Each rook board as a set of tuples (i, j)
    """
    
    # Generate empty rook board
    yield set()
    
    # For each k from 1 to min(m, n)
    for k in range(1, min(m, n) + 1):
        # Take k elements from m
        for m_subset in combinations(range(1, m + 1), k):
            # Take k elements from n
            for n_subset in combinations(range(1, n + 1), k):
                # Generate all permutations of the k elements in n
                for n_perm in permutations(n_subset):
                    # Pair elements in m with elements in n
                    matching = set(zip(m_subset, n_perm))
                    yield matching



def count_R(m, n):
    """
    Directly compute the number of all rook boards (partial matchings) between two sets.
    
    The formula is based on the permanent of a matrix, but can be computed more efficiently.
    For rook boards, we sum over all possible matching sizes k:
    - For each k from 0 to min(m, n):
      - Choose k elements from m: C(m, k)
      - Choose k elements from n: C(n, k)
      - Match them in all possible ways: k!
    
    Total count = Σ(k=0 to min(m,n)) C(m,k) * C(n,k) * k!
    
    Args:
        m: Size of the second set (j in range [1, m])
        n: Size of the first set (i in range [1, n])
    
    Returns:
        The total number of rook boards
    """
    
    total = 0
    for k in range(0, min(m, n) + 1):
        # C(m, k) * C(n, k) * k!
        count = comb(m, k) * comb(n, k) * factorial(k)
        total += count
    return total


def maze_to_rook_board(grid):
    """
    Convert a maze grid back to a rook board by extracting positions of 'N' entries.
    The inverse operation of generate_maze.
    
    Args:
        grid: A 2D list (matrix) representing the maze
    
    Returns:
        A set of tuples (i, j) representing the rook board (1-indexed)
    """
    if not grid:
        return set()
    
    m = len(grid)
    n = len(grid[0]) if m > 0 else 0
    
    rook_board = set()
    
    # Scan through the grid and find all 'N' entries
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 'S':
                # Convert to 1-indexed coordinates
                rook_board.add((i + 1, j + 1))
    
    return rook_board


def generate_maze(m, n, rook_board):
    # Define a function to compute the NW envelope which is the union of all indices (i,j) 
    # such that there exists (a,b) in X with i<=a and j<=b
    def compute_nw_envelope(X):
        """
        Compute the Northwest envelope of a rook board as a matrix.
        The NW envelope is represented as a (m+2) x (n+2) matrix where
        each cell contains 1 if the position (i,j) is in the envelope, 0 otherwise.
        A position (i,j) is in the envelope if there exists at least one rook 
        at position (a,b) in X where i <= a and j <= b.
        
        Args:
            X: Set of rook positions (tuples of (row, col))
            m: Number of rows in the grid
            n: Number of columns in the grid
        
        Returns:
            List of lists (matrix) where envelope[i][j] = 1 if (i,j) is in the envelope, 0 otherwise
            Matrix dimensions are (m+2) x (n+2) with 0-based indexing
        """
        # Initialize (m+2) x (n+2) matrix with zeros
        nw_envelope = [[0 for _ in range(n + 2)] for _ in range(m + 2)]
        
        # For each position (i,j) in the grid (using 1-based indexing)
        for i in range(1, m + 2):
            for j in range(1, n + 2):
                # Check if there exists any rook (a,b) in X such that i <= a and j <= b
                for a, b in X:
                    if i <= a and j <= b:
                        nw_envelope[i][j] = 1
                        break  # Found at least one, no need to check further
        
        return nw_envelope

    def remove_boundary_points(X, nw_envelope):
        """
        Remove boundary points from the augmented rook board.
        Non-boundary points (i,j) are those (i+1,j), (i+1,j+1) and (i,j+1) are all marked by 1  
         
        Args:
            X: Set of rook positions (tuples of (row, col)) including boundary points
            nw_envelope: The Northwest envelope matrix computed from X
            m: Number of rows in the original grid
            n: Number of columns in the original grid
        
        Returns:
            A new set containing only the non-boundary rook positions
        """
        non_boundary = set()
        
        for i, j in X:
            # Check if (i,j) is a non-boundary point
            # A point is non-boundary if (i+1,j), (i+1,j+1), and (i,j+1) are all in the envelope
            # Make sure indices are within bounds

            # Check if all three adjacent positions exist and are in the envelope
            # If a position doesn't exist, we don't check it
            check_right = j + 1 < n + 2
            check_down = i + 1 < m + 2
            check_diagonal = check_right and check_down
            
            # Determine if this is a non-boundary point
            is_non_boundary = True
            
            # Check (i+1, j) if it exists
            if check_down and nw_envelope[i + 1][j] != 1:
                is_non_boundary = False
            
            # Check (i, j+1) if it exists
            if check_right and nw_envelope[i][j + 1] != 1:
                is_non_boundary = False
            
            # Check (i+1, j+1) if it exists
            if check_diagonal and nw_envelope[i + 1][j + 1] != 1:
                is_non_boundary = False
            
            if is_non_boundary:
                non_boundary.add((i, j))
        
        return non_boundary


    def determine_cell_type(grid, i, j):
        """
        Determine the type of entry at position (i, j) in the maze grid.

        Cell types are defined as follows:
        - 'S' (SE corner): grid[i+1][j] = grid[i][j]-1 and grid[i][j+1] = grid[i][j]-1
        - 'N' (NW corner): grid[i+1][j+1] = grid[i][j]-1 and grid[i][j+1] = grid[i+1][j] = grid[i][j]
        - 'H' (Horizontal edge -): grid[i+1][j] = grid[i][j]-1 and grid[i][j+1] = grid[i][j]
        - 'V' (Vertical edge |): grid[i][j+1] = grid[i][j]-1 and grid[i+1][j] = grid[i][j]

        Args:
            grid: A 2D list (matrix) representing the maze
            i: Row index (0-based)
            j: Column index (0-based)

        Returns:
            A character ('N', 'S', 'H', 'V', or ' ') indicating the type of the cell
        """
        m = len(grid)
        n = len(grid[0]) if m > 0 else 0

        # Check bounds - if any adjacent cell is out of bounds, return space

        current = grid[i][j]
        down = grid[i + 1][j] if i+1 < m else grid[i][j]-1
        right = grid[i][j + 1] if j+1 < n else grid[i][j]-1
        diagonal = grid[i + 1][j + 1] if i+1<m and j+1<n else grid[i][j]-1

        # Check for SE corner: both down and right are current-1
        if down == current - 1 and right == current - 1:
            return 'S'

        # Check for NW corner: diagonal is current-1, and both down and right equal current
        if diagonal == current - 1 and down == current and right == current:
            return 'N'

        # Check for Horizontal edge: down is current-1, right equals current
        if down == current - 1 and right == current:
            return 'H'

        # Check for Vertical edge: right is current-1, down equals current
        if right == current - 1 and down == current:
            return 'V'

        # Default case: empty space
        return ' '

    # Step 1: Augment the rook set X1
    # X stores coordinates as (row, col) using 1-based indexing from the text
    X = set(rook_board)
    
    # Initialize (m+2) x (n+2) grid with empty strings
    # indices are 0..m+1 and 0..n+1
    grid = [[0 for _ in range(n+2)] for _ in range(m+2)]

    # Fill the augmented M+1 row and N+1 column
    cols_with_rooks = {c for r, c in X}
    for c in range(1, n + 1):
        if c not in cols_with_rooks:
            X.add((m + 1, c))
    rows_with_rooks = {r for r, c in X}
    for r in range(1, m + 1):
        if r not in rows_with_rooks:
            X.add((r, n + 1))


    def add_nw_env_to_grid(nw_env):
        """
        Add the Northwest envelope to the grid.
        For each position (i,j) in the envelope, mark it in the grid.
        
        Args:
            grid: The (m+2) x (n+2) grid to be modified
            nw_env: The Northwest envelope matrix
            m: Number of rows in the original grid
            n: Number of columns in the original grid
        """
        for i in range(m + 2):
            for j in range(n + 2):
                grid[i][j] += nw_env[i][j] 

    while X : 
        # Compute the NW envelope
        nw_env = compute_nw_envelope(X)
        add_nw_env_to_grid(nw_env)
        X = remove_boundary_points(X, nw_env)


    # Only take the 1..m, 1..n parts of the grid and return
    result=[]
    for i in range(1, m + 1):
        row = [determine_cell_type(grid, i, j) for j in range(1,n+1)] 
        result.append(row)

    # Check that all 'N' entries in the result correspond to elements in the original rook_board
    RB = maze_to_rook_board(result)
    assert RB == set(rook_board)
    return result


def maze_involution(grid):
    """
    Apply the involution on a maze by reversing the grid and swapping 'S' with 'N'.
    The involution maps grid[i][j] to grid[-i-1][-j-1] and changes 'S' to 'N' and vice versa.
    
    Args:
        grid: A 2D list (matrix) representing the maze
    
    Returns:
        A new 2D list representing the involution of the maze
    """
    if not grid:
        return []
    
    m = len(grid)
    n = len(grid[0]) if m > 0 else 0
    
    # Create a new grid with reversed indices
    result = [['' for _ in range(n)] for _ in range(m)]
    
    for i in range(m):
        for j in range(n):
            # Map grid[i][j] to result[m-1-i][n-1-j]
            cell = grid[i][j]
            
            # Swap 'S' and 'N'
            if cell == 'S':
                result[m-1-i][n-1-j] = 'N'
            elif cell == 'N':
                result[m-1-i][n-1-j] = 'S'
            else:
                result[m-1-i][n-1-j] = cell
    return result



def str_maze(grid):
    """
    Convert the maze grid to a string representation.
    Output the maze using the numbers in the grid directly.
    
    Args:
        grid: A 2D list (matrix) representing the maze
    
    Returns:
        A string representation of the maze
    """
    if not grid:
        return "Empty grid\n"
    
    m = len(grid)
    n = len(grid[0]) if m > 0 else 0
    
    # ANSI escape code for blue color
    BLUE = '\033[94m'
    RESET = '\033[0m'
    
    lines = []
    # Build each row with numbers from the grid
    for i in range(m):
        row_str = ""
        for j in range(n):
            if grid[i][j] == 'S':
                row_str += f"{BLUE}{grid[i][j]}{RESET}"
            else:
                row_str += f"{grid[i][j]}"
        lines.append(row_str.rstrip())
    
    return "\n".join(lines) 


def print_maze(grid):
    """
    Print the maze grid in a readable format.
    Output the maze using the numbers in the grid directly.
    
    Args:
        grid: A 2D list (matrix) representing the maze
    """
    print(str_maze(grid), end="")





def str_maze_by_type(grid):
    """
    Convert the maze grid to a string representation using cell type symbols.
    
    Args:
        grid: A 2D list (matrix) representing the maze
    
    Returns:
        A string representation of the maze with box-drawing characters
    """
    if not grid:
        return "Empty grid\n"
    
    m = len(grid)
    n = len(grid[0]) if m > 0 else 0
    
    # Mapping from cell type to box-drawing character
    symbol_map = {
        'N': '┌',
        'S': '┘',
        'H': '─',
        'V': '│',
        ' ': ' '
    }

    symbol_map = {
        'N': '\u250F',
        'S': '\u251B',
        'H': '\u2501',
        'V': '\u2503',
        ' ': '\u0020'
    }
    
    # ANSI escape code for blue color
    BLUE = '\033[94m'
    RESET = '\033[0m'
    
    lines = []
    # Build each row with box-drawing symbols
    for i in range(m):
        row_str = ""
        for j in range(n):
            if grid[i][j] == 'S':
                row_str += f"{BLUE}{symbol_map[grid[i][j]]}{RESET}"
            else:
                row_str += symbol_map[grid[i][j]]
        lines.append(row_str)
    
    return "\n".join(lines) + "\n"


def print_maze_by_type(grid):
    """
    Print the maze using cell type symbols.
    
    Args:
        grid: A 2D list (matrix) representing the maze
    """
    print(str_maze_by_type(grid), end="")


def concat_str_blocks(str1, str2, separator="  "):
    """
    Concatenate two blocks of strings side by side.
    This is used to print two mazes side by side for comparison.
    
    Args:
        str1: First string block (can contain multiple lines)
        str2: Second string block (can contain multiple lines)
        separator: String to separate the two blocks (default: "  ")
    
    Returns:
        A string with both blocks concatenated line by line
    """
    lines1 = str1.split('\n')
    lines2 = str2.split('\n')
    
    # Ensure both have the same number of lines by padding with empty strings
    max_lines = max(len(lines1), len(lines2))
    while len(lines1) < max_lines:
        lines1.append('')
    while len(lines2) < max_lines:
        lines2.append('')
    
    # Find the maximum width of lines in the first block
    max_width1 = max(len(line) for line in lines1) if lines1 else 0
    
    # Concatenate line by line
    result_lines = []
    for line1, line2 in zip(lines1, lines2):
        # Pad the first line to align the separator
        padded_line1 = line1.ljust(max_width1)
        result_lines.append(f"{padded_line1}{separator}{line2}")
    
    return '\n'.join(result_lines)



def concat_str_blocks_list(str_list, separator="  "):
    """
    Concatenate a list of string blocks side by side.
    Each block can contain multiple lines separated by newlines.
    
    Args:
        str_list: A list of string blocks to concatenate
        separator: String to separate the blocks (default: "  ")
    
    Returns:
        A string with all blocks concatenated line by line
    """
    import re
    
    def visible_length(s):
        """Calculate the visible length of a string, excluding ANSI escape codes."""
        # Remove ANSI escape sequences
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return len(ansi_escape.sub('', s))
    
    if not str_list:
        return ""
    
    # Split each block into lines
    all_lines = [block.split('\n') for block in str_list]
    
    # Find the maximum number of lines across all blocks
    max_lines = max(len(lines) for lines in all_lines)
    
    # Pad each block to have the same number of lines
    for lines in all_lines:
        while len(lines) < max_lines:
            lines.append('')
    
    # Find the maximum visible width for each block
    max_widths = []
    for lines in all_lines:
        max_width = max(visible_length(line) for line in lines) if lines else 0
        max_widths.append(max_width)
    
    # Concatenate line by line
    result_lines = []
    for line_idx in range(max_lines):
        result_line = ""
        for block_idx, lines in enumerate(all_lines):
            line = lines[line_idx]
            
            # Pad all blocks except the last one
            if block_idx < len(all_lines) - 1:
                visible_len = visible_length(line)
                padding_needed = max_widths[block_idx] - visible_len
                padded_line = line + ' ' * padding_needed + separator
                result_line += padded_line
            else:
                result_line += line
        
        result_lines.append(result_line)
    
    return '\n'.join(result_lines) + '\n'


def check_R_count(m, n):
    """
    Verify that the generate_R function produces the correct number
    of rook boards by comparing the actual count with the theoretical count.
    
    Args:
        m: Size of the second set (j in range [1, m])
        n: Size of the first set (i in range [1, n])
    
    Returns:
        A tuple (generated_count, expected_count, is_correct) where:
        - generated_count: actual number of rook boards generated
        - expected_count: theoretical number computed by count_R
        - is_correct: True if counts match, False otherwise
    """
    
    # Generate all rook boards and count them
    generated_count = sum(1 for _ in generate_R(m, n))
    
    # Compute expected count using the formula
    expected_count = count_R(m, n)
    
    # Check if they match
    is_correct = (generated_count == expected_count)
    
    return generated_count, expected_count, is_correct


def format_R(rook_board):
    """
    Format a rook board for printing by sorting pairs by their first element.
    
    Args:
        rook_board: A set of tuples (i, j) representing a rook board
    
    Returns:
        A string representation with pairs sorted by i (first element)
    """
    sorted_pairs = sorted(rook_board, key=lambda pair: pair[0])
    return str(sorted_pairs)



def reverse_pair(rook_board):
    """
    Reverse the pairs in a rook board by swapping the row and column indices.
    This operation transposes the rook board, converting (i, j) pairs to (j, i) pairs.
    
    Args:
        rook_board: A set of tuples (i, j) representing a rook board
    
    Returns:
        A new set of tuples with reversed pairs (j, i)
    """
    return {(j, i) for i, j in rook_board}

def Fourier(rook_board,m,n):
    """
    Apply the Fourier transform to a rook board.
    
    Args:
        rook_board: A set of tuples (i, j) representing a rook board
        m: Size of the second set (j in range [1, m])
        n: Size of the first set (i in range [1, n])
    
    Returns:
        A new set of tuples representing the Fourier transform of the rook board
    """
    maze = generate_maze(m, n, rook_board)
    invmaze = maze_involution(maze)
    res = maze_to_rook_board(invmaze)
    res = reverse_pair(res)
    return res


def size_of_maze(maze):
    m = len(maze)
    n = len(maze[0]) if m > 0 else 0
    return m, n

def trace_wall(maze,i,j):
    """
    Trace all possible paths in a maze starting from a given cell.
    
    Args:
        maze: A 2D list (matrix) representing the maze
        i: Starting row index
        j: Starting column index
    
    Returns:
        A set of tuples of the points in the wall 
    """
    m, n = size_of_maze(maze)
    res = []
    while j>=0 and i<m: 
        res.append((i,j,maze[i][j]))
        if maze[i][j] in ['S','H']:
            j = j-1
        elif maze[i][j] in ['N','V']:
            i = i+1
        else: 
            break    
    return res


def graph_to_matching(X,n=None):
    """
    Convert a graph (edge set) to a matching (edge set) by selecting one edge per vertex.
    
    Args:
        X: A set of tuples (i, j) representing the graph edges
        n: Number of vertices in the graph
    
    Returns:
        A set of tuples (i, j) representing the matching edges
    """
    assert X, "X must be non-empty."
    if not n:
        n = max(i for i,j in X)
    w = [0  for _ in range(n)]
    for i,j in X:
        w[i-1] = j
    return w

def matching_to_graph(w):
    G = set( (i+1, w[i]) for i in range(len(w)))
    return G


def maze_to_RB(maze):
    """
    Convert a maze grid to a rook board by extracting positions of 'S' entries.
    Only works for square mazes where m == n.

    This is [FGT (before) Lemma~4.2.5]
    
    Args:
        maze: A 2D list (matrix) representing the maze
    
    Returns:
        A set of tuples (i, j) representing the rook board (1-indexed)
   
    Raises:
        ValueError: If the maze is not square (m != n)
    """
    if not maze:
        return set(),set()
    
    m = len(maze)
    n = len(maze[0]) if m > 0 else 0

    # Check if the maze is square
    if m != n:
        raise ValueError(f"Maze must be square, but got m={m}, n={n}")
    res = set() 
    beta = set() 
    for j in range(n):
        path = trace_wall(maze,0,j)
        if path and path[0][2] in ['V','S']:
            for i,j,t  in path:
                if t == 'S':
                    res.add((i+1,j+1))
                    beta.add(j+1)
    for i in range(m):
        path = trace_wall(maze,i,n-1)
        if path and path[0][2] in ['H','N']:
            for i,j,t  in path:
                if t == 'N':
                    res.add((i+1,j+1))
    res = graph_to_matching(res)
    return (res,beta)


    
def RB_to_maze(w,beta,m,n):
    """
    Send RB_N to Maze_N,N. 
    This is [FGT Lemma 4.2.5]
    """
    def compute_nw_envelope(X):
        """
        Compute the Northwest envelope of a rook board as a matrix.
        The NW envelope is represented as a (m+2) x (n+2) matrix where
        each cell contains 1 if the position (i,j) is in the envelope, 0 otherwise.
        A position (i,j) is in the envelope if there exists at least one rook 
        at position (a,b) in X where i <= a and j <= b.
        
        Args:
            X: Set of rook positions (tuples of (row, col))
            m: Number of rows in the grid
            n: Number of columns in the grid
        
        Returns:
            List of lists (matrix) where envelope[i][j] = 1 if (i,j) is in the envelope, 0 otherwise
            Matrix dimensions are (m+2) x (n+2) with 0-based indexing
        """
        # Initialize (m+2) x (n+2) matrix with zeros
        nw_envelope = [[0 for _ in range(n+2)] for _ in range(m+2)]
        
        # For each position (i,j) in the grid (using 1-based indexing)
        for i in range(0, m + 1):
            for j in range(0, n + 1):
                # Check if there exists any rook (a,b) in X such that i <= a and j <= b
                for a, b in X:
                    if i <= a and j <= b:
                        nw_envelope[i][j] = 1
                        break  # Found at least one, no need to check further
        return nw_envelope


    def determine_nw_edge(grid, i, j):
        """
        Determine the type of entry at position (i, j) in the maze grid.

        Cell types are defined as follows:
        - 'S' (SE corner): grid[i+1][j] = grid[i][j]-1 and grid[i][j+1] = grid[i][j]-1
        - 'N' (NW corner): grid[i+1][j+1] = grid[i][j]-1 and grid[i][j+1] = grid[i+1][j] = grid[i][j]
        - 'H' (Horizontal edge -): grid[i+1][j] = grid[i][j]-1 and grid[i][j+1] = grid[i][j]
        - 'V' (Vertical edge |): grid[i][j+1] = grid[i][j]-1 and grid[i+1][j] = grid[i][j]

        Args:
            grid: A 2D list (matrix) representing the maze
            i: Row index 1..m 
            j: Column index 1..n 

        Returns:
            A character ('N', 'S', 'H', 'V', or ' ') indicating the type of the cell
        """
        assert 0 < i and i <= m, f"Row index i={i} is out of bounds for m={m}"
        assert 0 < j and j <= n, f"Column index j={j} is out of bounds for n={n}"

        # Check bounds - if any adjacent cell is out of bounds, return space
        current = grid[i][j]
        down = grid[i + 1][j] 
        right = grid[i][j + 1] 
        diagonal = grid[i + 1][j + 1] 
        # Check for SE corner: both down and right are current-1
        if down == current - 1 and right == current - 1:
            return 'S'
        # Check for NW corner: diagonal is current-1, and both down and right equal current
        if diagonal == current - 1 and down == current and right == current:
            return 'N'
        # Check for Horizontal edge: down is current-1, right equals current
        if down == current - 1 and right == current:
            return 'H'
        # Check for Vertical edge: right is current-1, down equals current
        if right == current - 1 and down == current:
            return 'V'
        # Default case: empty space
        return ' '

    def compute_se_envelope(X):
        """
        Compute the Southeast envelope of a rook board as a matrix.
        The SE envelope is represented as a (m+2) x (n+2) matrix where
        each cell contains 1 if the position (i,j) is in the envelope, 0 otherwise.
        A position (i,j) is in the envelope if there exists at least one rook 
        at position (a,b) in X where i >= a and j >= b.
        
        Args:
            X: Set of rook positions (tuples of (row, col))
            m: Number of rows in the grid
            n: Number of columns in the grid
        
        Returns:
            List of lists (matrix) where envelope[i][j] = 1 if (i,j) is in the envelope, 0 otherwise
            Matrix dimensions are (m+2) x (n+2) with 0-based indexing
        """
        # Initialize (m+2) x (n+2) matrix with zeros
        se_envelope = [[0 for _ in range(n+2 )] for _ in range(m+2)]
        
        # For each position (i,j) in the grid (using 1-based indexing)
        for i in range(1, m + 2):
            for j in range(1, n + 2):
                # Check if there exists any rook (a,b) in X such that i <= a and j <= b
                for a, b in X:
                    if i >= a and j >= b:
                        se_envelope[i][j] = 1
                        break  # Found at least one, no need to check further
        return se_envelope


    def determine_se_edge(grid, i, j):
        """
        Args:
            grid: A 2D list (matrix) representing the maze
            i: Row index 1..m 
            j: Column index 1..n 

        Returns:
            A character ('N', 'S', 'H', 'V', or ' ') indicating the type of the cell
        """
        assert 0 < i and i <= m, f"Row index i={i} is out of bounds for m={m}"
        assert 0 < j and j <= n, f"Column index j={j} is out of bounds for n={n}"

        # Check bounds - if any adjacent cell is out of bounds, return space
        current = grid[i][j]
        up = grid[i - 1][j] 
        left = grid[i][j - 1] 
        diagonal = grid[i - 1][j - 1] 
        # Check for SE corner: both down and right are current-1
        if up == current - 1 and left == current - 1:
            return 'N'
        # Check for NW corner: diagonal is current-1, and both down and right equal current
        if diagonal == current - 1 and up == current and left == current:
            return 'S'
        # Check for Horizontal edge: down is current-1, right equals current
        if up == current - 1 and left == current:
            return 'H'
        # Check for Vertical edge: right is current-1, down equals current
        if left == current - 1 and up == current:
            return 'V'
        # Default case: empty space
        return ' '

    # Initialize (m+2) x (n+2) grid with empty strings
    # indices are 0..m+1 and 0..n+1
    grid = [[' ' for _ in range(n+2)] for _ in range(m+2)]

    w = matching_to_graph(w)

    # All blue positions
    XNW = {(i,j) for i,j in w if j in beta}
    XSE = {(i,j) for i,j in w if not j in beta}
    while XNW:
        nw = compute_nw_envelope(XNW)
        # update grid
        for i in range(1,m+1):
            for j in range(1,n+1):
                c = determine_nw_edge(nw, i, j)
                if c != ' ':
                    grid[i][j] = c
                    # remove element on the boundary 
                    XNW.discard((i,j))
                
    while XSE:
        se = compute_se_envelope(XSE)
        # update grid
        for i in range(1,m+1):
            for j in range(1,n+1):
                c = determine_se_edge(se, i, j)
                if c != ' ':
                    grid[i][j] = c
                    # remove element on the boundary 
                    XSE.discard((i,j))
    
    # trim grid to m x n grid. 
    grid = [row[1:-1] for row in grid[1:-1]]
    return grid   
        
def print_all_R(m, n):
    """
    Print all rook boards (partial matchings) between two sets.
    
    Args:
        m: Size of the second set (j in range [1, m])
        n: Size of the first set (i in range [1, n])
    
    Returns:
        The total number of rook boards printed
    """
    count = 0
    print(f"All rook boards/mazes for m={m}, n={n}:")
    print("-" * 80)

    maze_to_RB_test = 0
    
    for rook_board in generate_R(m, n):
        count += 1
        maze = generate_maze(m, n, rook_board)
        strblocks = []
        strblocks.append(str_maze(maze))
        strblocks.append(str_maze_by_type(maze))
        invmaze = maze_involution(maze)
        strblocks.append(str_maze_by_type(invmaze))
        strblocks.append(str_maze(invmaze))
        fourier = reverse_pair(maze_to_rook_board(invmaze))
        print(f"{count}. {format_R(rook_board)} --> {format_R(fourier)}")
        print(concat_str_blocks_list(strblocks))

        # if m=n, test Maze to RB and RB to Maze
        if m == n:
            w,beta = maze_to_RB(maze)
            sigma = beta_to_sigma(w,beta)
            maze2 = RB_to_maze(w,beta,m,n)
            rook_board2 = RB_to_R(w,beta)

            print(f"{str_colored_partition(w,beta)}")
            print(f"{str_colored_partition(w,sigma)}")
            if maze2 == maze and frozenset(rook_board2) == frozenset(rook_board): 
                print("✓")
            else:
                maze_to_RB_test += 1
                print(f"{str_maze(maze2)}")
                print(f"{str_maze_by_type(maze2)}")
                print("✗")

    
    print("-" * 80)
    print(f"Total: {count} rook boards")
    
    # Verify the count is correct
    expected_count = count_R(m, n)
    if count == expected_count:
        print(f"✓ Count verification: PASSED ({count} == {expected_count})")
    else:
        print(f"✗ Count verification: FAILED ({count} != {expected_count})")

    if maze_to_RB_test > 0:
        print(f"✗ maze_to_RB test: FAILED ({maze_to_RB_test} cases)")
    else:
        print(f"✓ maze_to_RB test: PASSED")
    return count

def str_R(rook_board):
    """
    Convert a rook board to a string representation.
    The output is in two lines convention. 
    If rook_board = {(1,2),(3,4),(5,6)}, the output is:
    (1 3 5) 
    (2 4 6)
    """
    if not rook_board:
        return "()\n()"

    sorted_pairs = sorted(rook_board, key=lambda pair: pair[0])
    first_line = " ".join(str(i) for i, _ in sorted_pairs)
    second_line = " ".join(str(j) for _, j in sorted_pairs)
    return f"({first_line})\n({second_line})"


def RB_to_R(w,beta):
    """
    Convert a red-blue permutation to a rook board of size n.
    This is [FGT Lemma 4.2.7]: RB --> maze --> R
    """
    n = len(w)
    maze = RB_to_maze(w,beta,n,n)
    rook_board = frozenset(maze_to_rook_board(maze))
    return rook_board


def query_from_colored_permutation(w, beta, show_maze=False):
    """
    Query the rook board (and optional maze) from a colored permutation (w, beta).
    """
    n = len(w)
    if set(w) != set(range(1, n + 1)):
        raise ValueError(f"w must be a permutation of 1..{n}")
    if any(i < 1 or i > n for i in beta):
        raise ValueError(f"beta must be a subset of 1..{n}")

    rook_board = RB_to_R(w, beta)
    sigma = beta_to_sigma(w, beta)
    print(f"(w,beta) = {str_colored_partition(w, beta)}")
    print(f"(w,sigma) = {str_colored_partition(w, sigma)}")
    print(str_R(rook_board))

    if show_maze:
        maze = RB_to_maze(w, beta, n, n)
        print(str_maze(maze))
        print(str_maze_by_type(maze))

    return rook_board



def count_RB_to_rook_board_bijection(n, verbose=False, max_print=5):
    """
    Count sizes to show RB -> maze -> rook_board is bijective for size n.
    Print the map RB --> maze --> rook_board.
    Returns True if counts match and the map hits all rook boards.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    rook_from_rb = set()
    rb_count = 0
    for w in generate_permutations(n):
        for beta in generate_all_beta(w):
            rb_count += 1
            maze = RB_to_maze(w, beta, n, n)
            rook_from_rb.add(frozenset(maze_to_rook_board(maze)))
            sigma = beta_to_sigma(w, beta)
            print(f"(w,beta)={str_colored_partition(w, beta)} --> (w,sigma)={str_colored_partition(w, sigma)} --> rook_board\n{str_R(frozenset(maze_to_rook_board(maze)))}")
            print(str_maze(maze))

    rook_count = count_R(n, n)
    image_count = len(rook_from_rb)

    ok = (rb_count == rook_count == image_count)

    if verbose or not ok:
        print(f"RB elements count: {rb_count}")
        print(f"Rook boards count: {rook_count}")
        print(f"Image size (RB -> maze -> R): {image_count}")
        print("Count check:", "PASS" if ok else "FAIL")
        if not ok and max_print > 0:
            missing = [rb for rb in generate_R(n, n) if frozenset(rb) not in rook_from_rb]
            if missing:
                print(f"Missing rook boards (showing up to {max_print}):")
                for rb in missing[:max_print]:
                    print(f"  {format_R(rb)}")

    return ok


if __name__ == "__main__":
    import argparse

    def _parse_int_list(value):
        value = value.strip()
        if not value or value in {"[]", "()", "{}"}:
            return []
        if any(sep in value for sep in [",", " "]):
            parts = [p for p in value.replace(",", " ").split(" ") if p]
            return [int(p) for p in parts]
        return [int(ch) for ch in value]

    parser = argparse.ArgumentParser(
        description="Rook boards, mazes, and queries from colored permutations."
    )
    parser.add_argument("-m", type=int, help="Size of the second set")
    parser.add_argument("-n", type=int, help="Size of the first set")
    parser.add_argument(
        "--w",
        type=str,
        help="Permutation w as '2,1,3' or '213'.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--beta",
        type=str,
        help="Beta positions as '1,3' or '13'.",
    )
    group.add_argument(
        "--sigma",
        type=str,
        help="Sigma positions as '1,3' or '13'.",
    )
    parser.add_argument(
        "--show-maze",
        action="store_true",
        help="Also print the maze representations.",
    )
    parser.add_argument(
        "--check-bijection",
        action="store_true",
        help="Check RB -> maze -> rook_board bijection for size n.",
    )
    args = parser.parse_args()

    if args.check_bijection:
        if args.n is None:
            parser.error("--check-bijection requires -n")
        count_RB_to_rook_board_bijection(args.n, verbose=True)
    elif args.w:
        w = _parse_int_list(args.w)
        if not w:
            raise ValueError("w must be non-empty")
        if args.beta:
            beta = set(_parse_int_list(args.beta))
            assert is_beta_on_subset(w, beta), f"beta={beta} is not a valid beta subset for w={w}"
        elif args.sigma:
            sigma = set(_parse_int_list(args.sigma))
            assert is_decreasing_on_subset(w, sigma), f"sigma={sigma} is not a valid sigma subset for w={w}"
            beta = sigma_to_beta(w, sigma)
        else:
            raise ValueError("Provide --beta or --sigma with --w")
        query_from_colored_permutation(w, beta, show_maze=args.show_maze)
    else:
        if args.m is None or args.n is None:
            parser.error("Provide -m and -n to list rook boards, or --w to query.")
        if args.m <= 0 or args.n <= 0:
            raise ValueError("m and n must be positive integers")
        print_all_R(args.m, args.n)
