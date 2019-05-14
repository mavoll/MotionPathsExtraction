import configparser
import sys
import subprocess
import multiprocessing
import time
import glob
import os

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
        
    def process(self, i, j, count, file):
        
        print("Starting process %d on GPU %d" % (j, int(self.gpu_ids[i][1]))) 
        
        try:
            process = detect_and_track.App(True,
                                           True,
                                           int(self.gpu_ids[i][1]),
                                           j, 
                                           str(self.configs[i][1]), 
                                           str(self.inputs[count][1]), 
                                           str(file))
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
    
    bulk = BulkProcessor()                    
           
    proc_dict = {}
    count = 0
               
    for i in range(len(bulk.gpu_ids)):  
        for j in range(int(bulk.num_instances[i][1])):            
            
            glob_list = glob.glob(bulk.inputs[count][1] + '/**/*.' + bulk.file_types[i][1], recursive=True)            
            procs = []
            
            for index, vid_file_name in enumerate(glob_list):
                file_name = os.path.splitext(vid_file_name)[0] + "_tracks.csv"            
                if os.path.isfile(file_name) is not True:
                    p = multiprocessing.Process(target=bulk.process, args=(i, j, count, vid_file_name))
                    p.daemon = True
                    procs.append(p)
            
            if len(procs) > 0:
                proc_dict[str(i) + str(j)] = procs.copy()
                proc = procs[0]
                proc.start()
                time.sleep(30) 
            
            count += 1
    
    while len(proc_dict) > 0:  
        tmp_dict = proc_dict.copy()
        for key in tmp_dict.keys():
            procs = tmp_dict[key]
            proc = procs[0]
            if not proc.is_alive():
                del procs[0]
                
                if len(procs) > 0:
                    tmp_dict[key] = procs.copy()     
                    next_proc = procs[0]                                
                    time.sleep(30) 
                    next_proc.start()                                   
                    time.sleep(30)
                else:
                    del tmp_dict[key]
                    
        proc_dict = tmp_dict.copy()
                    
        time.sleep(1)