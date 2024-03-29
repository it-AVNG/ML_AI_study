import sys
import pickle
import numpy as np


def __convert_to_float(args):
    return list(map(float, args))


def convert_to_list(string):
    li = list(string.split(","))
    li = __convert_to_float(li)
    return li


def model_predict(a: list) -> str:
    model_load = pickle.load(open("exported_models/check_iris_model.pkl", "rb"))

    new_flower = np.array([a])
    result = model_load.predict(new_flower)

    return str(result)


if __name__ == "__main__":
    array = sys.argv[1]
    array = convert_to_list(array)
    print(model_predict(array))
