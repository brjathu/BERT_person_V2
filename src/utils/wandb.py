import wandb
import time
import os
import sys




if __name__ == "__main__":
    wait_time  = int(sys.argv[1])
    min_folder = int(sys.argv[2])
    log_dir    = "/private/home/jathushan/3D/BERT_person_hydra/logs/"

    # infine loop
    while True:

        list_of_runs = [i for i in os.listdir(log_dir) if ("test" not in i and len(i)<=5 and int(i[:4])>=int(min_folder))]

        for runs in list_of_runs:
            if(runs=="1958"):
                os.system("wandb sync -p BERT_person_hydra --sync-tensorboard " + log_dir + "/" + str(runs) + "/tensorboard/lightning_logs/version_0/" + " --id " + str(runs) + "_1")
            else:
                os.system("wandb sync -p BERT_person_hydra --sync-tensorboard " + log_dir + "/" + str(runs) + "/tensorboard/lightning_logs/version_0/" + " --id " + str(runs))
                
            
        time.sleep(wait_time)
