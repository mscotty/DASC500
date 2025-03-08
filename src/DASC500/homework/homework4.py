import os

from DASC500.classes.DataAnalysis import DataAnalysis
from DASC500.utilities.get_top_level_module import get_top_level_module_path
import statsmodels.api as sm

folder = get_top_level_module_path()
data_file = os.path.join(folder, "../../data/homework4/mtcars.csv")
output_folder = os.path.join(folder, "../../outputs/homework4")
output_logger = os.path.join(output_folder, 'homework4_results.txt')

data_obj = DataAnalysis(data_file)
predictor_vars = ["cyl", "disp", "hp", "drat", "wt", "qsec", "vs", "am", "gear", "carb"]
response_var = "mpg"

def problem1():
    print("Problem 1")
    print(f"Total Number of Elements: {len(data_obj.df)}")
    data_obj.downsample_dataframe(frac=0.7, random_state=42)
    print(f"Current Train DF Elements: {len(data_obj.df)}")
    print(f"Current Test DF Elements: {len(data_obj.df_test)}")

def problem2():
    print("Problem 2")
    
    data_obj.build_linear_regression_model(response_var, predictor_vars)
    print(data_obj.lin_reg_model)

def problem3():
    print("Problem 3")
    print("Best Predictor: cyl")

def problem4():
    print("Problem 4")
    print("mpg = beta_0 + beta_1 * cyl")
    print(f"mpg = {data_obj.lin_reg_model.loc[0, 'β0']} + {data_obj.lin_reg_model.loc[0, 'β1']} * cyl")

def problem5():
    print("Problem 5")
    data_obj.build_mult_linear_regression_model(response_var, predictor_vars)

def problem6():
    print("Problem 6")

def extra_credit():
    print("Extra Credit")
    data_obj.build_stepwise_parsimonious_regression_model(response_var, predictor_vars)
    print(data_obj.parsimonious_model['used_vars'])
    # Dictionary for storing simple models
    simple_models = {
        "wt": sm.OLS(data_obj.df["mpg"], sm.add_constant(data_obj.df["wt"])).fit(),
        "cyl": sm.OLS(data_obj.df["mpg"], sm.add_constant(data_obj.df["cyl"])).fit()
    }
    data_obj.vis_reg_models("mpg", 
                            simple_models, 
                            data_obj.mult_lin_reg_model, 
                            data_obj.parsimonious_model,
                            output_dir=output_folder)

if __name__ == "__main__":
    problem1()
    problem2()
    problem3()
    problem4()
    problem5()
    problem6()
    extra_credit()