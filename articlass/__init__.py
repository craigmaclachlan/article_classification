import logging

logging_config = {
    'version': 1,
    'formatters': {
        'f': {
            'format': '%(asctime)s - %(name)s - %(levelname)s: %(message)s',
            'datefmt': "%Y-%m-%d %H:%M:%S"}
        },
    'handlers': {
        'h': {'class': 'logging.StreamHandler',
              'formatter': 'f',
              'level': logging.INFO}
        },
    'root': {
        'handlers': ['h'],
        'level': logging.INFO,
        },
}
