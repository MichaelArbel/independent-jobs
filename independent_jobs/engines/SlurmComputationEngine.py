
import os
import time

from independent_jobs.engines.BatchClusterComputationEngine import BatchClusterComputationEngine


class SlurmComputationEngine(BatchClusterComputationEngine):
    def __init__(self, batch_parameters, check_interval=10, do_clean_up=False):
        BatchClusterComputationEngine.__init__(self,
                                               batch_parameters=batch_parameters,
                                               check_interval=check_interval,
                                               submission_cmd="sbatch",
                                               do_clean_up=do_clean_up,
                                               submission_delay=0.01)

    def create_batch_script(self, job_name, dispatcher_string):
        command = "nice -n 10 " + dispatcher_string
        
        walltime = time.strftime('%H:%M:%S', time.gmtime(self.batch_parameters.max_walltime))
        
        num_nodes = str(self.batch_parameters.nodes)
        # note memory is in megabyes
        memory = str(self.batch_parameters.memory)
        workdir = self.get_job_foldername(job_name)
        
        output = workdir + os.sep + "output.txt"
        error = workdir + os.sep + "error.txt"
        
        job_string = """#!/bin/bash
#SBATCH -J %s
#SBATCH --time=%s
#SBATCH -n %s
#SBATCH --mem=%s
#SBATCH --output=%s
#SBATCH --error=%s
cd %s
%s""" % (job_name, walltime, num_nodes, memory, output, error, workdir,
         command)
        
        return job_string