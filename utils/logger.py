import logging


class Logger:
    COLORS = {
        'RESET': '\033[0m',
        'RED': '\033[91m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'BLUE': '\033[94m',
        'PURPLE': '\033[95m',
        'CYAN': '\033[96m'
    }

    class ColoredFormatter(logging.Formatter):
        def format(self, record):
            colors = Logger.COLORS
            color_map = {
                'DEBUG': colors['CYAN'],
                'INFO': colors['GREEN'],
                'WARNING': colors['YELLOW'],
                'ERROR': colors['RED'],
                'CRITICAL': colors['PURPLE']
            }
            color = color_map.get(record.levelname, colors['RESET'])
            message = super().format(record)
            return f"{color}{message}{colors['RESET']}"

    @staticmethod
    def setup(name, verbose="WARNING"):
        logger = logging.getLogger(name)

        if logger.hasHandlers():
            return logger

        try:
            level = getattr(logging, verbose.upper())
            logger.setLevel(level)
        except AttributeError:
            print(f"Invalid log level '{verbose}'. Defaulting to WARNING")
            logger.setLevel(logging.WARNING)

        handler = logging.StreamHandler()
        formatter = Logger.ColoredFormatter('%(name)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger
