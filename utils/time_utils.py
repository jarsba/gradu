from datetime import datetime

DATETIME_FORMAT = '%Y%m%d_%H%M%S%f'


def get_formatted_datetime():
    now = datetime.now()
    now_str = now.strftime(DATETIME_FORMAT)
    return now_str
