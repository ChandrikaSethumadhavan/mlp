import sys #to control the exceptions                         #refer to python custom exception handling documentation
from src.logger import logging #importing the logger module to log all errors and exceptions into a text file

def error_message_detail(error, error_detail:sys):
    """
    This function takes an error and its details and returns a string with the error message and details.
    """
    _, _, exc_tb = error_detail.exc_info()  # exc_info() returns in which line the error occurred # exc_tb is a traceback object that contains information about the error
    file_name = exc_tb.tb_frame.f_code.co_filename # tb_frame returns the frame object for the current exception, f_code returns the code object for the current frame, co_filename returns the name of the file in which the error occurred
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in script: [{file_name}] at line number: [{line_number}] error message: [{str(error)}]"
    return error_message



class CustomException(Exception):
    """
    This class is a custom exception that inherits from the built-in Exception class.
    It takes an error and its details as arguments and returns a string with the error message and details.
    """
    def __init__(self, error, error_detail:sys):
        super().__init__(error) # call the constructor of the parent class
        self.error_message = error_message_detail(error, error_detail=error_detail) # call the error_message_detail function from above to get the error message

    def _str_(self):
        return self.error_message   # TO print the error message
    


if __name__ == "__main__":
    try:
        a = 1/0 # this will raise a ZeroDivisionError
    except Exception as e:
        logging.info("Logging has been configured.")
        raise CustomException(e, sys) # raise the custom exception with the error and its details
    
