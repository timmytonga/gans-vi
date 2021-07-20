import matplotlib.pyplot as plt
import wandb

api = wandb.Api()

svrg_run_paths = {"adapegsvrg_optimistic": api.run("optimproject/optimproj/3k93dg4d"),
                  "adapegsvrg_nooptimistic": api.run("optimproject/optimproj/2vag1nlz"),
                  "optimistic_svrg": api.run("optimproject/optimproj/vw1rtpeg"),
                  "extraadam_svrg": api.run("optimproject/optimproj/j0o8oyqs")}

fig, axs = plt.subplots(2, sharex=True)

for run_name, run in svrg_run_paths.items():
    if run.state == "finished":
        gen_data = []
        dis_data = []
        if run_name == "adapegsvrg_optimistic":
            for i, row in run.history().iterrows():
                if (i+1) % 20 == 0:
                    gen_data.append(row["GEN_VARIANCE"])
                    dis_data.append(row["DIS_VARIANCE"])
        else:
            for i, row in run.history().iterrows():
                if str(row["GEN_VARIANCE"]) != "nan":
                    gen_data.append(row["GEN_VARIANCE"])
                if str(row["DIS_VARIANCE"]) != "nan":
                    dis_data.append(row["DIS_VARIANCE"])

        print(len(gen_data))

        axs[0].plot(list(range(len(gen_data))), gen_data, label=run_name)
        axs[1].plot(list(range(len(dis_data))), dis_data, label=run_name)


axs[0].legend()
axs[0].set_title("GEN_VARIANCE")

axs[1].legend()
axs[1].set_title("DIS_VARIANCE")

plt.savefig("fig1.png")




