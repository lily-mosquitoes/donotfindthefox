use std::iter::zip;

use rand::Rng;

fn str_says_fox(slice: &[char]) -> bool {
    let word: String = slice.iter().collect();
    word == "fox" || word == "xof"
}

fn grid_3x3_says_fox(grid: [[char; 3]; 3]) -> bool {
    let mut diagonal_a = Vec::new();
    let mut diagonal_b = Vec::new();
    for (x, y) in zip(0..3, (0..3).rev()) {
        if str_says_fox(&grid[x]) {
            return true;
        };
        let column: Vec<char> = (0..3).map(|y| grid[y][x]).collect();
        if str_says_fox(&column) {
            return true;
        };
        diagonal_a.push(grid[x][x]);
        diagonal_b.push(grid[x][y]);
    }
    str_says_fox(diagonal_a.as_slice()) || str_says_fox(diagonal_b.as_slice())
}

fn grid_nxn_says_fox<const N: usize>(grid: [[char; N]; N]) -> bool {
    for window_x in (0..N).collect::<Vec<usize>>().windows(3) {
        for window_y in (0..N).collect::<Vec<usize>>().windows(3) {
            let mut grid3x3 = [['-'; 3]; 3];
            for x in 0..3 {
                for y in 0..3 {
                    grid3x3[x][y] = grid[window_x[x]][window_y[y]];
                }
            }
            if grid_3x3_says_fox(grid3x3) {
                return true;
            }
        }
    }
    false
}

fn random_tries(charset: &str, n: usize) -> (usize, usize) {
    let mut rng = rand::thread_rng();

    let result = (0..n).map(|_| {
        let mut charset: Vec<char> = charset.chars().collect();
        let mut f = || charset.remove(rng.gen_range(0..charset.len()));

        let grid = [
            [f(), f(), f(), f()], //
            [f(), f(), f(), f()], //
            [f(), f(), f(), f()], //
            [f(), f(), f(), f()], //
        ];
        grid_nxn_says_fox(grid) as usize
    });

    (result.sum(), n)
}

#[derive(Debug, Clone)]
struct PermutationsUnique<I: Iterator> {
    is_first_iter: bool,
    next: Option<Vec<I::Item>>,
}

impl<I> Iterator for PermutationsUnique<I>
where
    I: Iterator,
    I::Item: Clone + Ord,
{
    type Item = Vec<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut current = match (&self.is_first_iter, &self.next) {
            (true, _) => {
                self.is_first_iter = false;
                return self.next.clone();
            }
            (_, None) => return None,
            (false, Some(vec)) => vec.clone(),
        };

        // loop from n-2 to 0 to find the first sorted element
        let mut j = current.len() - 2;
        loop {
            if current[j] < current[j + 1] {
                break;
            }

            if j == 0 {
                // we looped through the whole sequence without breaking,
                // it is already reversed (we're finished)
                self.next = None;
                return None;
            }

            j -= 1;
        }

        // find m such that array[m] is the first value larger than array[j]
        // when searching backwards in the array from n-1 to j+1
        // think of the sequence as two: [0..=j], [j+1..n]
        let mut m = current.len() - 1;
        while m > j + 1 {
            if current[j] < current[m] {
                break;
            }
            m -= 1;
        }
        // swap array[j] and array[m]
        current.swap(j, m);

        // construct new array with the second portion reversed
        // i.e. [0..=j], [n-1..=j+1]
        let mut next = Vec::new();
        next.extend_from_slice(&current[..=j]);
        let tail = &mut current[(j + 1)..];
        tail.reverse();
        next.extend_from_slice(&tail);
        self.next = Some(next);

        self.next.clone()
    }
}

trait Permutations: Iterator {
    fn permutations_unique(self) -> PermutationsUnique<Self>
    where
        Self: Sized,
        Self::Item: Clone + Ord,
    {
        let mut next: Vec<Self::Item> = self.collect();
        next.sort();
        PermutationsUnique {
            is_first_iter: true,
            next: Some(next),
        }
    }
}

impl<T> Permutations for T where T: Iterator + ?Sized {}

fn permutate(charset: &str) -> (usize, usize) {
    let permutations = charset.chars().permutations_unique();
    let total = permutations.clone().count();

    let mut foxes = 0;
    let mut step = 0;
    for p in permutations {
        step += 1;
        print!("\r({step}) {:.6} %", step as f64 / (total as f64 / 100.0));
        let grid = [
            [p[0], p[1], p[2], p[3]],     //
            [p[4], p[5], p[6], p[7]],     //
            [p[8], p[9], p[10], p[11]],   //
            [p[12], p[13], p[14], p[15]], //
        ];
        foxes += grid_nxn_says_fox(grid) as usize;
    }
    println!("");

    (foxes, total)
}

fn main() {
    let charset = "fffffooooooxxxxx";
    println!("{charset}");

    let (foxes, games) = random_tries(charset, 2_018_016);
    println!(
        "Random win rate = {:.2} %",
        100.0 - (foxes as f64 / (games as f64 / 100.0))
    );

    let (foxes, games) = permutate(charset);
    println!(
        "Total win rate = {:.2} %",
        100.0 - (foxes as f64 / (games as f64 / 100.0))
    );
}

#[cfg(test)]
mod test {
    use crate::grid_3x3_says_fox;
    use crate::Permutations;

    #[test]
    fn test_permutations_unique() {
        let expected = vec![
            vec!['1', '1', '2', '2'], //
            vec!['1', '2', '1', '2'], //
            vec!['1', '2', '2', '1'], //
            vec!['2', '1', '1', '2'], //
            vec!['2', '1', '2', '1'], //
            vec!['2', '2', '1', '1'], //
        ];

        let charset = "1122";
        let result: Vec<Vec<char>> = charset.chars().permutations_unique().collect();

        assert_eq!(result, expected);
    }

    static GRIDS_WHICH_SAY_FOX: &[[[char; 3]; 3]] = &[
        [
            ['f', 'o', 'x'], //
            ['-', '-', '-'], //
            ['-', '-', '-'], //
        ],
        [
            ['-', '-', '-'], //
            ['f', 'o', 'x'], //
            ['-', '-', '-'], //
        ],
        [
            ['-', '-', '-'], //
            ['-', '-', '-'], //
            ['f', 'o', 'x'], //
        ],
        [
            ['x', 'o', 'f'], //
            ['-', '-', '-'], //
            ['-', '-', '-'], //
        ],
        [
            ['-', '-', '-'], //
            ['x', 'o', 'f'], //
            ['-', '-', '-'], //
        ],
        [
            ['-', '-', '-'], //
            ['-', '-', '-'], //
            ['x', 'o', 'f'], //
        ],
        [
            ['f', '-', '-'], //
            ['o', '-', '-'], //
            ['x', '-', '-'], //
        ],
        [
            ['-', 'f', '-'], //
            ['-', 'o', '-'], //
            ['-', 'x', '-'], //
        ],
        [
            ['-', '-', 'f'], //
            ['-', '-', 'o'], //
            ['-', '-', 'x'], //
        ],
        [
            ['x', '-', '-'], //
            ['o', '-', '-'], //
            ['f', '-', '-'], //
        ],
        [
            ['-', 'x', '-'], //
            ['-', 'o', '-'], //
            ['-', 'f', '-'], //
        ],
        [
            ['-', '-', 'x'], //
            ['-', '-', 'o'], //
            ['-', '-', 'f'], //
        ],
        [
            ['f', '-', '-'], //
            ['-', 'o', '-'], //
            ['-', '-', 'x'], //
        ],
        [
            ['x', '-', '-'], //
            ['-', 'o', '-'], //
            ['-', '-', 'f'], //
        ],
        [
            ['-', '-', 'x'], //
            ['-', 'o', '-'], //
            ['f', '-', '-'], //
        ],
        [
            ['-', '-', 'f'], //
            ['-', 'o', '-'], //
            ['x', '-', '-'], //
        ],
    ];

    static GRIDS_WHICH_DO_NOT_SAY_FOX: &[[[char; 3]; 3]] = &[
        [
            ['f', 'x', 'o'], //
            ['o', 'o', 'x'], //
            ['f', 'x', 'f'], //
        ], //
    ];

    #[test]
    fn detects_fox() {
        for grid in GRIDS_WHICH_SAY_FOX.iter() {
            assert!(grid_3x3_says_fox(*grid));
        }
    }

    #[test]
    fn detects_not_fox() {
        for grid in GRIDS_WHICH_DO_NOT_SAY_FOX.iter() {
            assert!(!grid_3x3_says_fox(*grid));
        }
    }
}
