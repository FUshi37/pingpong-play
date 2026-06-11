import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/yangzhe/Project/pingpong_controller/install/pingpong_controller'
