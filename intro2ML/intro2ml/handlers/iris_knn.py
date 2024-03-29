import sys
import pickle
import numpy as np

# utils
target_names = {0: "setosa", 1: "versicolor", 2: "virginica"}


def __convert_to_float(args):
    # convert input into type float
    return list(map(float, args))


def __decode_name(d: int):
    # decode the output and match it with a target names
    return target_names[d]


def convert_to_list(string):
    # split the arguments into element bu comma
    li = list(string.split(","))
    # convert all element into type float
    li = __convert_to_float(li)
    return li


# prediction
def model_predict(a: list) -> str:
    # load the model
    model_load = pickle.load(open("exported_models/check_iris_model.pkl", "rb"))

    # load the features into a 2D np array
    new_flower = np.array([a])
    # predict the name of the flower with the feature data
    result = model_load.predict(new_flower)
    # get the name by decode the result
    output = map(__decode_name, result)
    # return the outcome
    return output


if __name__ == "__main__":
    # get the 1st arguments from the cli
    array = sys.argv[1]
    # convert the arguments into lists
    array = convert_to_list(array)
    # determine the result by invoking the models
    results = model_predict(array)
    # print the result in a list type
    print(list(results))
