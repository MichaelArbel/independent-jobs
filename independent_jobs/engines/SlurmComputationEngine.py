import os
import popen2

from independent_jobs.engines.BatchClusterComputationEngine import BatchClusterComputationEngine
from independent_jobs.tools.Log import logger
from independent_jobs.tools.Time import Time

import time


class SlurmComputationEngine(BatchClusterComputationEngine):
    def __init__(self, batch_parameters, check_interval=10, do_clean_up=False, partition=None, additional_input= ''):
        BatchClusterComputationEngine.__init__(self,
                                               batch_parameters=batch_parameters,
                                               check_interval=check_interval,
                                               submission_cmd="sbatch",
                                               do_clean_up=do_clean_up,
                                               submission_delay=0.01,
                                               max_jobs_in_queue=2000)
        
        self.partition=partition
        self.additional_input = additional_input

    def _infer_slurm_qos(self, max_walltime, nodes):
        if max_walltime <= 60 * 60 and \
           nodes <= 90:
            qos = "short"
        elif max_walltime <= 60 * 60 * 24 and \
             nodes <= 70:
            qos = "normal"
        elif max_walltime <= 60 * 60 * 72 and \
             nodes <= 20:
            qos = "medium"
        elif max_walltime <= 60 * 60 * 168 and \
             nodes <= 10:
            qos = "long"
        else:
            logger.warning("Unable to infer slurm qos. Setting to normal")
            qos = "normal"
            
        return qos

    def create_batch_script(self, job_name, dispatcher_string, walltime, memory, nodes):
        command = "nice -n 10 " + dispatcher_string
        
        qos = self._infer_slurm_qos(walltime,
                                    nodes)
        
        days, hours, minutes, seconds = Time.sec_to_all(walltime)
        walltime = '%d-%d:%d:%d' % (days, hours, minutes, seconds)
        
        num_nodes = str(nodes)
        # note memory is in megabyes
        memory = str(memory)
        workdir = self.get_job_foldername(job_name)
        
        output = workdir + os.sep + BatchClusterComputationEngine.output_filename
        error = workdir + os.sep + BatchClusterComputationEngine.error_filename
        
        job_strings = ["#!/bin/bash"]
        job_strings += ["#SBATCH -J %s" % job_name]
        job_strings += ["#SBATCH --time=%s" % walltime]
        job_strings += ["#SBATCH --qos=%s" % qos]
        job_strings += ["#SBATCH -n %s" % num_nodes]
        job_strings += ["#SBATCH --mem=%s" % memory]
        job_strings += ["#SBATCH --output=%s" % output]
        job_strings += ["#SBATCH --error=%s" % error]
        
        if self.partition is not None:
            job_strings += ["#SBATCH --partition=%s" % self.partition]

        job_strings += ["%s" %self.additional_input]
        
        #job_strings += ["cd %s" % workdir]
        job_strings += ["%s" % command]
        
        return os.linesep.join(job_strings)

    def submit_to_batch_system(self, job_string):
        # send job_string to batch command
        num_max_trials = 10
        i =0
        outpipe, inpipe = popen2.popen2(self.submission_cmd)
        inpipe.write(job_string + os.linesep)
        inpipe.close()
        job_id = outpipe.read().strip().split(" ")[-1]
        outpipe.close()
        while job_id == "" and i < num_max_trials:
            time.sleep(2.)
            outpipe, inpipe = popen2.popen2(self.submission_cmd)
            inpipe.write(job_string + os.linesep)
            inpipe.close()
            job_id = outpipe.read().strip().split(" ")[-1]
            outpipe.close()
            i += 1
        
        return job_id