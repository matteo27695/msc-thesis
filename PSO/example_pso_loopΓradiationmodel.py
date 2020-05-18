import os
import pickle
import sys
from csv import DictWriter, writer

import matplotlib.pyplot as plt

from pso_Γradiationmodel import OdeModel, PrepareData, RegressionMetrics, Swarm

if __name__ == '__main__':
    def append_list_as_row(file_name, list_of_elem):
        # Open file in append mode
        with open(file_name, 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj, delimiter=';')
            # Add contents of list as last row in the csv file
            csv_writer.writerow(list_of_elem)

    def append_dict_as_row(file_name, dict_of_elem, field_names):
        # Open file in append mode
        with open(file_name, 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            dict_writer = DictWriter(
                write_obj, fieldnames=field_names, delimiter=';')
            # Add dictionary as wor in the csv
            dict_writer.writerow(dict_of_elem)

    if sys.platform == "darwin":
        path = r"path cleandictpatients.pkl"
        path_results = r"path save plot"
    else:
        path = r"path cleandictpatients.pkl"
        path_results = r"path save plot"
        path_name = r"path cleanpatientsname.pkl"

    pkl_file = open(path, 'rb')
    dictpatients = pickle.load(pkl_file)
    pkl_file.close()
    pkl_file = open(path_name, 'rb')
    patients = pickle.load(pkl_file)
    pkl_file.close()

    for patient in patients:

        dates = dictpatients[patient]["dates"]
        volumes = dictpatients[patient]["volumes"]
        start_date = dictpatients[patient]["start_date"]
        end_date = dictpatients[patient]["end_date"]
        tot_dose = dictpatients[patient]["tot_dose"]
        if len(volumes)>1:
            ratio = 10

            t_radio = 25
            timestep = "minuto"

            ut, t, days, nt, ut_min = PrepareData.inputode(start_date, end_date, tot_dose=tot_dose,
                                                           t_radio=t_radio, timestep=timestep)
            realdata, time = PrepareData.realdata(start_date, end_date, dates,
                                                  volumes, timestep=timestep)

            odemodel = OdeModel(
                days=days, z0=[volumes[0], 0], ut=ut, b=volumes[0] * 2, ratio=ratio, timestep="minuto")
            swarm = Swarm(num_particles=40, n_iterations=1000, dimspace=2,
                          realdata=realdata, mode="Parallel", odemodel=odemodel)
            swarm.run()

            odemodel.param(
                a=swarm.gbest_pos[0], alpha=swarm.gbest_pos[1], ratio=ratio, b=volumes[0] * 2)
            odemodel.solve()

            real = realdata[realdata != 0]
            estimated = odemodel.x[realdata != 0]

            rss = RegressionMetrics.metric_rss(real, estimated)
            mse = RegressionMetrics.metric_mse(real, estimated)
            msep = RegressionMetrics.metric_msep(real, estimated) * 100
            rmse = RegressionMetrics.metric_rmse(real, estimated)
            rmsep = RegressionMetrics.metric_rmsep(real, estimated) * 100
            mae = RegressionMetrics.metric_mae(real, estimated)
            mape = RegressionMetrics.metric_mape(real, estimated) * 100
            rse = RegressionMetrics.metric_rse(real, estimated)
            rae = RegressionMetrics.metric_rae(real, estimated)

            note = "Fitnesss: MAPE"
            modello = "Gamma-LQ"

            performance = {"Modello": modello, "Paziente": patient, "num_particles": swarm.num_particles,
                           "n_iterations": swarm.n_iterations, "tolerance": swarm.tolerance, "timestep": timestep,
                           "t_radio": t_radio, "ratio": odemodel.ratio, "a": swarm.gbest_pos[0],
                           "alpha": swarm.gbest_pos[1], "rss": rss, "mse": mse, "msep": msep,
                           "rmse": rmse, "rmsep": rmsep, "mae": mae, "mape": mape, "rse": rse,
                           "rae":  rae, "run_time": swarm.run_time, "dates": dates,
                           "volumes": volumes, "note": note}

            field_names = ["Modello", "Paziente", "num_particles", "n_iterations", "tolerance",
                           "timestep", "t_radio", "ratio", "a", "alpha", "rss", "mse", "msep",
                           "rmse", "rmsep", "mae", "mape", "rse", "rae", "run_time", "dates",
                           "volumes", "note"]

            append_dict_as_row("performances-LQmodel.csv",
                               performance, field_names)

            plt.rcParams['figure.dpi'] = 150
            plt.rcParams['figure.figsize'] = [10, 5.5]
            plt.figure()
            plt.plot(time, volumes, "*", label='Real Data')
            plt.plot(odemodel.t, odemodel.x, label='Estimation by PSO Algorithm')
            plt.legend()
            plt.xlabel('time [day]')
            plt.ylabel("volumes [cc]")
            plt.savefig(os.path.join(path_results, f"{patient}.png"))
            plt.close()
