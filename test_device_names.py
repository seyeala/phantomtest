import nidaqmx
from nidaqmx.system import System

system = System.local()
for d in system.devices:
    print(d.name, d.product_type)
