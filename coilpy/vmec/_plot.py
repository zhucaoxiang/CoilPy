import matplotlib.pyplot as plt
import numpy as np


def plot(self, plot_name="none", ax=None):
    """Plot various VMEC quantities.

    Args:
        plot_name (str, optional): The quantity to be plotted, should be one of
                                    iota, q, pressue, <Buco>, <Bvco>, <jcuru>, <jcurv>,
                                    <j.B>, LPK, none. Defaults to 'none'.
        ax (Matplotlib axis, optional): The Matplotlib axis to be plotted on. Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    if plot_name == "none":
        print("You can plot: iota, q, pressue, <Buco>, <Bvco>, <jcuru>, <jcurv>, ")
        print("               <j.B>, LPK")
    elif plot_name == "iota":
        ax.plot(self.data["nflux"], self.wout["iotaf"].values)
        ax.set_xlabel("Normalized Flux")
        ax.set_ylabel("iota")
        ax.set_title("Rotational Transform")
        # ax.set(xlabel='s',ylabel='iota',aspect='square')
    elif plot_name == "q":
        ax.plot(self.data["nflux"], 1.0 / self.wout["iotaf"].values)
        ax.set_xlabel("Normalized Flux")
        ax.set_ylabel("q")
        ax.set_title("Safety Factor")
    elif plot_name == "pressure":
        ax.plot(self.data["nflux"], self.wout["presf"].values / 1000)
        ax.set_xlabel("Normalized Flux")
        ax.set_ylabel("Pressure [kPa]")
        ax.set_title("Pressure Profile")
    elif plot_name == "<Buco>":
        ax.plot(self.data["nflux"], self.wout["buco"].values)
        ax.set_xlabel("Normalized Flux")
        ax.set_ylabel("<B^u> [T]")
        ax.set_title("Flux surface Averaged B^u")
    elif plot_name == "<Bvco>":
        ax.plot(self.data["nflux"], self.wout["bvco"].values)
        ax.set_xlabel("Normalized Flux")
        ax.set_ylabel("<B^v> [T]")
        ax.set_title("Flux surface Averaged B^v")
    elif plot_name == "<jcuru>":
        ax.plot(self.data["nflux"], self.wout["jcuru"].values / 1000)
        ax.set_xlabel("Normalized Flux")
        ax.set_ylabel("<j^u> [kA/m^2]")
        ax.set_title("Flux surface Averaged j^u")
    elif plot_name == "<jcurv>":
        ax.plot(self.data["nflux"], self.wout["jcurv"].values / 1000)
        ax.set_xlabel("Normalized Flux")
        ax.set_ylabel("<j^v> [kA/m^2]")
        ax.set_title("Flux surface Averaged j^v")
    elif plot_name == "<j.B>":
        ax.plot(self.data["nflux"], self.wout["jdotb"].values / 1000)
        ax.set_xlabel("Normalized Flux")
        ax.set_ylabel("<j.B> [T*kA/m^2]")
        ax.set_title("Flux surface Averaged j.B")
    elif plot_name == "LPK":
        self.surface[-1].plot(zeta=0, color="red", label=r"$\phi=0$")
        self.surface[-1].plot(
            zeta=0.5 * np.pi / self.data["nfp"], color="green", label=r"$\phi=0.25$"
        )
        self.surface[-1].plot(
            zeta=np.pi / self.data["nfp"], color="blue", label=r"$\phi=0.5$"
        )
        ax.set_title("LPK Plot")
    elif plot_name[0] == "-":
        print(plot_name)
    else:
        return
