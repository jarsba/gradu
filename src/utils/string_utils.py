def epsilon_str_to_float(epsilon_str: str):
    epsilon = float(f"{epsilon_str[0]}.{epsilon_str[1]}")
    return epsilon


def epsilon_float_to_str(epsilon: float):
    epsilon_str = str(epsilon).replace(".", "")
    return epsilon_str
