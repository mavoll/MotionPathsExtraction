import configparser
import os, sys
import subprocess
import multiprocessing
from tkinter import messagebox

import detect_and_track


class BulkProcessor(object):

    def __init__(self):
        
        num_nvidia_gpus = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')
        
        config = configparser.ConfigParser()
        config.sections()
        config.read('bulk_config.ini')
        
        self.gpu_ids = config.items( "Gpus" )
        self.configs = config.items( "Configs" )
        self.inputs = config.items( "Input" )
        self.num_instances = config.items( "Instances_per_gpu" )
        self.file_types = config.items( "File_type" )
        
        if len(self.gpu_ids) < 1 or len(self.gpu_ids) > num_nvidia_gpus:
        	sys.exit('Number of processors to use must be greater than 0 and smaller than gpus available')
        
        if len(self.gpu_ids) != len(self.configs):
        	sys.exit('len(self.gpu_ids) != len(self.configs) != len(self.inputs) != len(self.outputs)')
        
    def process(self, i, j, count):
        
        print("Starting process %d on GPU %d" % (j, int(self.gpu_ids[i][1]))) 
        
        try:
            process = detect_and_track.App(True,
                                           True,
                                           int(self.gpu_ids[i][1]),
                                           j, 
                                           str(self.configs[i][1]), 
                                           str(self.inputs[count][1]), 
                                           str(self.file_types[i][1]))
            process.start_bulk()

            print("Process on GPU %d stopped" % int(self.gpu_ids[i][1]))
        
        except Exception:
    
                e = sys.exc_info()[0] + '' + sys.exc_info()[1]
                print("Error: " + str(e))
                raise       
           
if __name__ == '__main__':

    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    procs = []
    bulk = BulkProcessor()           
           
    count = 0
    for i in range(len(bulk.gpu_ids)):  
        for j in range(int(bulk.num_instances[i][1])):
            p = multiprocessing.Process(target=bulk.process, args=(i, j, count))
            procs.append(p)
            p.start()
            count += 1
            
    for proc in procs:
        proc.join()