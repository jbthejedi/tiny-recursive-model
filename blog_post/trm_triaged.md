What to know about the datasets:
* Sudoku-Extreme is extremely difficult Sudoku puzzles. 423K samples for testing
* Maze-Hard is 30x30 procedurally generated mazes. Train/test are both of size 1000
* ARC-AGI-1/ARC-AGI-2 are geometric puzzles
    * Each puzzle task contains 2-3 example solutions and 1-2 test puzzles to be solved
    * Easy for humans, hard for AI
    * ARC-AGI data also augmented with 160 tasks from ConceptARC, a closely related dataset
* Datasets are small so augmentation is used: Sudoku-Extreme, Maze-Hard, and ARC-AGI all use shuffling or data transformation per example 