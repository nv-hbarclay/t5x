from ctypes import cdll, c_char_p, c_int, c_uint64, c_void_p
libcudart = cdll.LoadLibrary('libcudart.so')

def cudaProfilerStart():
  libcudart.cudaProfilerStart()
def cudaProfilerStop():
  libcudart.cudaProfilerStop()

