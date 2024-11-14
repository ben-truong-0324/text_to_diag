import ro_baseline
import ro_by_problem_size
import ro_validation_curves
import ro_monte_carlo
from config import *



if __name__ == "__main__":
    #set randomization seed for reproducibility
    np.random.seed(GT_ID)

    ro_baseline.run_ro_baseline_experiment()
    ro_by_problem_size.run_ro_by_problem_size()
    ro_monte_carlo.run_ro_monte_carlo()
    ro_validation_curves.run_ro_validation_curves()
    