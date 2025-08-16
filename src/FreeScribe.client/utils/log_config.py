import os
import io
import sys
import logging
from collections import deque
import utils.file_utils
from utils.AESCryptoUtils import AESCryptoUtilsClass


MAX_BUFFER_SIZE = 2500

# Configure logging
if os.environ.get("FREESCRIBE_DEBUG"):
    LOG_LEVEL = logging.DEBUG
else:
    LOG_LEVEL = logging.INFO

class AESEncryptedFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        # Define your visible format (not shown, but used for encryption)
        super().__init__(fmt=fmt or "[%(asctime)s] | %(levelname)s | %(name)s | %(threadName)s | [%(filename)s:%(lineno)d in %(funcName)s] | %(message)s", datefmt=datefmt)

    def format(self, record):
        # First format the entire line (all metadata)
        plain_line = super().format(record)

        return AESCryptoUtilsClass.encrypt(plain_line)


class BufferHandler(logging.Handler):
    """Custom handler that maintains an in-memory buffer of log records.
    
    This handler stores log records in a deque buffer with a maximum capacity,
    allowing efficient access to recent log history.
    
    :param capacity: Maximum number of records to store (default: 2500)
    :type capacity: int
    """

    def __init__(self, capacity=MAX_BUFFER_SIZE):
        super().__init__()
        self.buffer = deque(maxlen=capacity)

    def emit(self, record):
        """Store the log record in the buffer.
        
        :param record: The log record to store
        :type record: logging.LogRecord
        :raises Exception: If record cannot be stored, handled by handleError
        """
        try:
            self.buffer.append(record)
        except Exception:
            self.handleError(record)

    def get_buffer_content(self):
        """Get all buffered records as formatted strings.
        
        :return: Complete buffer contents as single string with newline separators
        :rtype: str
        :note: Records are formatted using the handler's formatter
        """
        return '\n'.join(self.format(record) for record in self.buffer)


class LoggingStream(io.StringIO):
    """A stream that logs messages to a specified logger level.
    
    This class implements the io.StringIO interface while redirecting
    writes to a logger at the specified level.
    
    :param level: Logging level to use for messages (e.g., logging.INFO)
    :type level: int
    """

    def __init__(self, level):
        super().__init__()
        self.level = level
        # Store encoding as instance variable since we can't set the class attribute
        self._encoding = 'utf-8'

    def write(self, message):
        """Write message to logger, splitting multi-line messages.
        
        :param message: The message to write/log
        :type message: str or bytes
        :return: Length of the processed message
        :rtype: int
        :note: Empty messages are ignored, multi-line messages are split
        """
        # Handle bytes input by decoding with UTF-8
        if isinstance(message, bytes):
            try:
                message = message.decode(self._encoding)
            except UnicodeDecodeError:
                # Fallback to replace invalid characters
                message = message.decode(self._encoding, errors='replace')

        message = message.strip()
        if message:
            for line in message.splitlines():
                try:
                    logger.log(self.level, line)
                except UnicodeEncodeError:
                    # If encoding fails, replace problematic characters
                    logger.log(self.level, line.encode(self._encoding, errors='replace').decode(self._encoding))
        return len(message)

    def flush(self):
        """No-op flush to satisfy stream interface.
        
        This method exists to maintain compatibility with the stream interface
        but performs no actual operations.
        """
        pass


def addLoggingLevel(levelName, levelNum, methodName=None):
    """Add a new logging level to the logging module.
    
    :param levelName: Name of the new level (e.g., 'DIAG')
    :type levelName: str
    :param levelNum: Numeric value for the new level
    :type levelNum: int
    :param methodName: Optional method name to add to logger (defaults to levelName.lower())
    :type methodName: str or None
    :raises AttributeError: If level or method name already exists
    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
        raise AttributeError("{} already defined in logging module".format(levelName))
    if hasattr(logging, methodName):
        raise AttributeError("{} already defined in logging module".format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
        raise AttributeError("{} already defined in logger class".format(methodName))

    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)

def add_file_handler(log, format:logging.Formatter, file_name:str = "freescribe.log", level:int = logging.DEBUG):
    """Add a file handler to the logger.
    
    This function creates a file handler for logging and sets its level and formatter.
    The log file is named 'freescribe.log' and is located in the current working directory.
    """
    log_file = utils.file_utils.get_resource_path(file_name)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(format)
    log.addHandler(file_handler)
    log.info(f"File Log Path: {log_file}")

def remove_file_handler(log, file_name:str = "freescribe.log"):
    """Remove the file handler from the logger.
    
    This function removes the file handler associated with the specified log file name.
    """
    log_file = utils.file_utils.get_resource_path(file_name)
    for handler in log.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == log_file:
            log.removeHandler(handler)
            break

# Define custom level
DIAGNOSE_LEVEL = 99
addLoggingLevel("DIAG", DIAGNOSE_LEVEL)

LOG_FORMAT = '[%(asctime)s] | %(levelname)s | %(name)s | %(threadName)s | [%(filename)s:%(lineno)d in %(funcName)s] | %(message)s'

formatter = logging.Formatter(LOG_FORMAT)

# When running a PyInstaller-built application with --windowed mode, there's no console,
# so sys.stdout and sys.stderr are set to None.
# Since Python's logging module tries to write to sys.stdout (or another stream handler),
# it fails with AttributeError: 'NoneType' object has no attribute 'write'.
if sys.stderr or sys.stdout:
    console_handler = logging.StreamHandler(sys.stderr or sys.stdout)
else:
    console_handler = logging.NullHandler()
console_handler.setLevel(LOG_LEVEL)
console_handler.setFormatter(formatter)

buffer_handler = BufferHandler(capacity=MAX_BUFFER_SIZE)
buffer_handler.setLevel(LOG_LEVEL)
buffer_handler.setFormatter(formatter)

# root logger settings
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[console_handler, buffer_handler],
    format=LOG_FORMAT
)
logger = logging.getLogger("freescribe")
logger.setLevel(LOG_LEVEL)

sys.stdout = LoggingStream(logging.INFO)
sys.stderr = LoggingStream(DIAGNOSE_LEVEL)
