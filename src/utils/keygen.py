import random
import string


def get_key():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
