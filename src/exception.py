import sys
import logger
def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(file_name,exc_tb.tb_lineno,str(error))
    return error_message

class CustomException(Exception):
    
    def __init__(self,error_message,error_detail:sys) -> None:
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)
    
    def __str__(self) -> str:
        return self.error_message
    
# try:
#     c=1/0
# except Exception as e:
#     custom_error=CustomException(e,sys)
#     logger.logging.error(custom_error)  # it is not required to mention logger, since when import logger the logger.py runs
#     raise custom_error from None # Stops program and shows clean message (without Tracebacks)