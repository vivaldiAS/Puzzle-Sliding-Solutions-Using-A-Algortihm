from flask import Flask, render_template, request, jsonify
import random
from PIL import Image, ImageEnhance
import pytesseract
import numpy as np
import cv2
from heapq import heappop, heappush


app = Flask(__name__)

# Konfigurasi lokasi Tesseract (sesuaikan jika perlu)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Fungsi untuk membuat grid puzzle yang solvable
def create_puzzle_grid(size):
    while True:
        numbers = list(range(1, size * size)) + [0]  # 0 adalah ruang kosong
        random.shuffle(numbers)
        flat_grid = numbers
        if is_solvable(flat_grid, size):
            # Ubah menjadi grid 2D
            grid = [flat_grid[i * size:(i + 1) * size] for i in range(size)]
            return grid
        
def generate_puzzle_grid(size):
    # Buat puzzle grid dinamis
    import random
    puzzle = list(range(size * size))
    random.shuffle(puzzle)
    return [puzzle[i * size:(i + 1) * size] for i in range(size)]

# Fungsi untuk menghitung inversi
def count_inversions(flat_grid):
    inversions = 0
    flat_grid = [num for num in flat_grid if num != 0]  # Hapus angka 0
    for i in range(len(flat_grid)):
        for j in range(i + 1, len(flat_grid)):
            if flat_grid[i] > flat_grid[j]:
                inversions += 1
    return inversions

# Fungsi untuk mengecek apakah puzzle solvable
def is_solvable(flat_grid, size):
    inversions = count_inversions(flat_grid)
    if size % 2 == 1:  # Untuk ukuran ganjil, inversi harus genap
        return inversions % 2 == 0
    else:  # Untuk ukuran genap
        zero_row = flat_grid.index(0) // size
        return (inversions + zero_row) % 2 == 0

# Kelas untuk menyelesaikan puzzle menggunakan A*
class PuzzleSolver:
    def __init__(self, initial, size):
        self.initial = initial
        self.size = size
        self.goal = list(range(1, size * size)) + [0]  # Goal state

    def get_neighbors(self, state):
        idx = state.index(0)
        neighbors = []
        row, col = divmod(idx, self.size)

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.size and 0 <= new_col < self.size:
                new_idx = new_row * self.size + new_col
                new_state = state[:]
                new_state[idx], new_state[new_idx] = new_state[new_idx], new_state[idx]
                neighbors.append(new_state)
        return neighbors

    def heuristic(self, state):
        distance = 0
        for i, value in enumerate(state):
            if value == 0:
                continue
            goal_row, goal_col = divmod(value - 1, self.size)
            curr_row, curr_col = divmod(i, self.size)
            distance += abs(goal_row - curr_row) + abs(goal_col - curr_col)
        return distance

    def solve(self):    
        if not is_solvable(self.initial, self.size):
            return None

        frontier = [(self.heuristic(self.initial), 0, self.initial, [])]
        visited = set()

        while frontier:
            total_cost, current_cost, current_state, path = heappop(frontier)

            if current_state == self.goal:
                solution_steps = []
                for step in path + [current_state]:
                    space_idx = step.index(0)
                    row, col = divmod(space_idx, self.size)
                    solution_steps.append({
                        'step': step,
                        'space_position': (row, col)
                    })
                return solution_steps

            visited.add(tuple(current_state))

            for neighbor in self.get_neighbors(current_state):
                if tuple(neighbor) not in visited:
                    new_cost = current_cost + 1
                    heappush(frontier, (new_cost + self.heuristic(neighbor), new_cost, neighbor, path + [current_state]))
        return None

# Fungsi untuk preprocessing gambar
def preprocess_image(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, thresholded = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
    pil_image = Image.fromarray(thresholded)
    enhancer = ImageEnhance.Contrast(pil_image)
    enhanced_image = enhancer.enhance(2)
    return enhanced_image

# Fungsi untuk ekstrak angka dari gambar
def extract_digits_only(image):
    preprocessed_image = preprocess_image(image)
    extracted_text = pytesseract.image_to_string(preprocessed_image, config='--psm 6')
    digits = []
    for line in extracted_text.splitlines():
        line = ''.join(filter(str.isdigit, line))
        if line:
            digits.append(line)
    return digits


@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/generate-puzzle', methods=['POST'])
def generate_puzzle():
    grid_size = int(request.json.get('grid_size', 3))  # Ambil ukuran grid (default 3)
    puzzle = generate_puzzle_grid(grid_size)
    return jsonify({'grid_size': grid_size, 'puzzle': puzzle})

    
@app.route('/puzzleangka')
def puzzle_angka():
    return render_template('puzzleangka.html')

@app.route('/get_puzzle', methods=['POST'])
def get_puzzle():
    try:
        size = int(request.json.get('size', 3))
        if size < 2:
            raise ValueError("Puzzle size must be at least 2.")
        puzzle_grid = create_puzzle_grid(size)
        flat_grid = [cell for row in puzzle_grid for cell in row]
        if not is_solvable(flat_grid, size):
            raise ValueError("Generated puzzle is not solvable. Please try again.")
        return jsonify({'grid': puzzle_grid, 'size': size})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/solve_puzzle', methods=['POST'])
def solve_puzzle():
    try:
        image = request.files['puzzle_image']
        img = Image.open(image)

        extracted_text_lines = extract_digits_only(img)
        grid = [list(map(int, line)) for line in extracted_text_lines]

        size = len(grid)
        if any(len(row) != size for row in grid):
            raise ValueError("Puzzle grid is not square.")

        flat_grid = [cell for row in grid for cell in row]
        solver = PuzzleSolver(flat_grid, size)
        solution = solver.solve()

        if solution is None:
            return render_template('solution.html', error="Puzzle is not solvable.", extracted_text_lines=extracted_text_lines)
        return render_template('solution.html', grid=grid, extracted_text_lines=extracted_text_lines, solution_steps=solution)
    except Exception as e:
        return render_template('solution.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)