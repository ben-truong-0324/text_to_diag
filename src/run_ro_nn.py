import nn_baseline
import nn_op_train_size
import nn_validation
import nn_cv
from config import *

if __name__ == "__main__":
    #set randomization seed for reproducibility
    np.random.seed(GT_ID)

    nn_baseline.run_nn_baseline_experiment()
    nn_op_train_size.run_nn_by_train_size()
    nn_validation.run_nn_validation_experiment()
    nn_cv.run_nn_cross_validation()
    