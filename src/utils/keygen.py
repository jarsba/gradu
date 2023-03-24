import random
import string
import uuid


def generate_experiment_id() -> str:
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))


def get_hash(query_dataset: str, epsilon: float, algo: str):
    uuid_long = uuid.uuid3(uuid.NAMESPACE_DNS, f"{query_dataset}{epsilon}{algo}")
    hash = str(uuid_long)[-12:]
    return hash
